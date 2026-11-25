#!/usr/bin/env python3
"""Verify completed native Train epochs while the original fit continues."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tools.launch_native_pyramid_study import verify_fit
from tools.native_pyramid_train_data import schedule_proof
from tools.native_pyramid_training_loop import state_sha256


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def process(pid):
    path = Path('/proc') / str(pid)
    if not path.exists():
        return dict(pid=pid, alive=False)
    stat, command = (path / 'stat').read_text(), (path / 'cmdline').read_bytes()
    fields = stat[stat.rfind(')')+2:].split()
    environment = (path / 'environ').read_bytes().split(b'\0')
    return dict(pid=pid, alive=True, kernel_state=fields[0], start_ticks=int(fields[19]),
        command=command.replace(b'\0', b' ').decode().strip(),
        command_sha256=hashlib.sha256(command).hexdigest(), physical_gpu=next(
            (v.decode().split('=', 1)[1] for v in environment if v.startswith(b'CUDA_VISIBLE_DEVICES=')), None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-id', action='append', required=True)
    parser.add_argument('--through-epoch', type=int, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or args.through_epoch < 1:
        raise ValueError('CPU-only verification of a positive completed prefix is required')
    started = time.monotonic()
    planpath = ROOT / 'configs/perceptual_mft_20261122.json'
    studypath = ROOT / 'configs/perceptual_native_pyramid_study_20261124.json'
    plan, study = read(planpath), read(studypath)
    work = Path(plan['work_root'])
    regpath = work / 'native_pyramid_pipeline_registered.json'
    registration, state = read(regpath), read(work / 'native_pyramid_study_state.json')
    files = {str(p): sha(p) for p in [planpath, studypath, regpath, Path(__file__)]}
    for name, digest in registration['source_hashes'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Original native registered source changed: ' + name)
        files[str(ROOT / name)] = digest
    fits = {r['id']: r for r in study['normal_fits']}
    if len(set(args.fit_id)) != len(args.fit_id) or not set(args.fit_id) <= set(fits):
        raise ValueError('Select distinct already registered normal fits')
    rows, captured = [], {}
    for name in args.fit_id:
        folder = work / 'native_pyramid_fits' / name
        path = folder / 'progress.json'
        # Capture a single complete JSON inode; later epochs may update the live file.
        payload = path.read_bytes()
        progress = json.loads(payload)
        verify_fit(work, name, args.through_epoch)
        if progress['completed_full_Train_epochs'] < args.through_epoch:
            raise ValueError('The captured report does not cover the requested epoch')
        metadata = progress['metadata']
        if (metadata['seed'] != 0 or metadata['cue'] != fits[name]['cue'] or
                metadata['registry_sha256'] != sha(regpath) or metadata['study_sha256'] != sha(studypath) or
                metadata['source_hashes'] != registration['source_hashes'] or
                metadata['source_RGB_sha256'] != study['source_RGB_sha256'] or
                not progress['frozen_RGB_exact_unchanged']):
            raise ValueError('Original model, source, seed or frozen RGB contract differs')
        snapshots = []
        for epoch in range(1, args.through_epoch+1):
            history = progress['history'][epoch-1]
            expected = schedule_proof(study['Train_delivery']['lengths'], epoch)
            delivery = history['delivery']
            if (history['epoch'] != epoch or not history['frozen_RGB_exact_unchanged'] or
                    any(delivery[k] != v for k, v in expected.items()) or
                    not delivery['complete_all_Train_frames'] or delivery['delivered_clips'] != 6814 or
                    delivery['actual_address_sha256'] != expected['address_sha256'] or
                    delivery['frames'] != 27251 or delivery['repeated_frames'] != 0 or
                    not all(math.isfinite(v) for v in history['mean_clip_losses'].values())):
                raise ValueError('Actual complete Train coverage, frozen state or loss differs')
            checkpoint = folder / f'epoch_{epoch:02d}.pt'
            digest = sha(checkpoint)
            value = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if (value['metadata'] != metadata or value['completed_epochs'] != epoch or
                    value['history'] != progress['history'][:epoch] or
                    not value['source_RGB_only_by_reference'] or sha(checkpoint) != digest):
                raise ValueError('An immutable epoch snapshot differs from its original journal')
            if any(not bool(torch.isfinite(v).all()) for v in value['module'].values() if v.is_floating_point()):
                raise ValueError('An actual native checkpoint contains nonfinite tensors')
            module_sha = state_sha256(value['module'])
            files[str(checkpoint)] = digest
            snapshots.append(dict(epoch=epoch, checkpoint=str(checkpoint), checkpoint_sha256=digest,
                bytes=checkpoint.stat().st_size, actual_module_sha256=module_sha,
                module_differs_from_initial=module_sha != metadata['initial_native_sha256'],
                all_native_tensors_finite=True, history=history))
            del value
        reference = metadata['input_content_reference']
        required_entries = 54502 if metadata['cue'] == 'ir' else 27251
        if reference['expected_input_entries'] != required_entries:
            raise ValueError('Expected on-read Train image set differs')
        for original in reference['reference_successful_IR_content_manifests'].values():
            if sha(original['path']) != original['sha256']:
                raise ValueError('Original fixed-image reference changed')
            files[original['path']] = original['sha256']
        phase = 'NORMAL_TRAIN_' + str(progress['registered_through_epoch']).zfill(2)
        job = state['phases'][phase][name]
        actual = process(job['pid'])
        if job['status'] == 'RUNNING' and (not actual['alive'] or actual['kernel_state'] == 'Z' or
                actual['command'] != ' '.join(job['command']) or actual['physical_gpu'] != job['physical_gpu']):
            raise ValueError('The claimed original running worker is not actually alive')
        rows.append(dict(fit_id=name, cue=metadata['cue'], audited_complete_epochs=args.through_epoch,
            captured_total_complete_epochs=progress['completed_full_Train_epochs'],
            original_registered_stage_epochs=progress['registered_through_epoch'],
            captured_progress_sha256=hashlib.sha256(payload).hexdigest(), metadata=metadata,
            snapshots=snapshots, actual_original_worker=actual, actual_original_job_status=job['status'],
            on_read_image_SHA_guard_bound_to_successful_reference=True,
            native_per_image_consumption_list_not_saved_in_this_epoch_journal=True,
            IR_formal_Train_used=metadata['cue'] == 'ir'))
        captured[name] = payload
    if any(sha(path) != digest for path, digest in files.items()) or torch.cuda.is_initialized():
        raise ValueError('Static source/snapshot changed, or this audit initialized CUDA')
    report = dict(status='COMPLETE_NATIVE_TRAIN_EPOCH_PREFIX_AUDIT_NOT_COMPLETION_QUALIFICATION',
        source_hashes=files, rows=rows, original_controller=process(state['controller_pid']),
        original_phase=state['status'], audited_complete_epochs_each=args.through_epoch,
        requested_fits=args.fit_id, full_study_completion_claimed=False, new_GPU_or_NN_work=False,
        new_fit_or_optimizer_update=False, new_Dev_or_Test_accuracy=False, fixed_seed=0,
        additional_seed_work=False, goal_met=False, supported_innovations=0,
        seconds=time.monotonic()-started,
        peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        scope='Actual completed prefix of original already-running native fits,not entire6/9/12epoch stage. '
              'Full27251frame/6814clip/no-padding delivery recomputed;immutable checkpoints and frozenRGB journal checked. '
              'Pinned on-read imageSHA assertions bound by executed source,not a retroactively fabricated per-image list. '
              'No IR training or modality benefit is inferred for RGB/gray controls.No stop,resume,restart or Test selection.')
    if report['peak_process_RSS_GiB'] > 8:
        raise RuntimeError('CPU verification exceeded8GiB')
    output = args.output_directory
    output.mkdir(exist_ok=False)
    for name, payload in captured.items():
        (output / (name+'_original_progress.json')).write_bytes(payload)
    (output / 'complete.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], fits=args.fit_id, full_epochs_each=args.through_epoch,
                         seconds=report['seconds'], peak_process_RSS_GiB=report['peak_process_RSS_GiB'])))


if __name__ == '__main__':
    main()
