#!/usr/bin/env python3
"""Inspect finished native fits without a forward pass or optimizer update.

AdamW's second moment is separate from decoupled weight decay. A positive
stored second moment therefore establishes historical nonzero gradients,
whereas changed weights alone do not. Neither establishes useful predictions.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
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

from tools.native_pyramid_fit import learning_rate_factor
from tools.native_pyramid_training_loop import state_sha256
from yolox.models.native_pyramid_fusion import NativePyramidFusion


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def process(pid):
    folder = Path('/proc') / str(pid)
    try:
        stat = (folder / 'stat').read_text().rsplit(') ', 1)[1].split()
        command = (folder / 'cmdline').read_bytes()
        return dict(pid=pid, present=True, kernel_state=stat[0],
                    start_ticks=int(stat[19]),
                    raw_argv_sha256=hashlib.sha256(command).hexdigest(),
                    command=command.replace(b'\0', b' ').decode().strip())
    except FileNotFoundError:
        return dict(pid=pid, present=False)


def group_name(name):
    parts = name.split('.')
    return '.'.join(parts[:3] if parts[0] == 'encoder' else parts[:2])


def parameter_row(name, initial, trained, optimizer_state):
    old, new = initial.detach().double(), trained.detach().double()
    delta = new - old
    n = old.numel()
    initial_l2_squared = float(old.square().sum())
    delta_l2_squared = float(delta.square().sum())
    moment = optimizer_state.get('exp_avg')
    second = optimizer_state.get('exp_avg_sq')
    if moment is not None and (moment.shape != old.shape or not torch.isfinite(moment).all()):
        raise ValueError('Invalid first moment: ' + name)
    if second is not None and (second.shape != old.shape or not torch.isfinite(second).all() or
                               bool((second < 0).any())):
        raise ValueError('Invalid second moment: ' + name)
    step = optimizer_state.get('step')
    if step is not None:
        step = float(step)
        if not math.isfinite(step) or step < 0 or step != int(step):
            raise ValueError('Invalid optimizer step: ' + name)
        step = int(step)
    positive_second = int((second > 0).sum()) if second is not None else 0
    return dict(name=name, group=group_name(name), shape=list(trained.shape),
        elements=n, trained_dtype=str(trained.dtype),
        changed_elements=int((trained != initial).sum()),
        trained_nonzero_elements=int((trained != 0).sum()),
        initial_L2_squared=initial_l2_squared, delta_L2_squared=delta_l2_squared,
        initial_RMS=math.sqrt(initial_l2_squared / n),
        trained_RMS=math.sqrt(float(new.square().sum()) / n),
        delta_RMS=math.sqrt(delta_l2_squared / n), delta_abs_max=float(delta.abs().max()),
        delta_to_initial_L2_ratio=(math.sqrt(delta_l2_squared / initial_l2_squared)
                                   if initial_l2_squared else None),
        optimizer_state_present=bool(optimizer_state), optimizer_steps=step,
        optimizer_first_moment_nonzero_elements=(int((moment != 0).sum()) if moment is not None else 0),
        optimizer_first_moment_RMS=(math.sqrt(float(moment.double().square().mean()))
                                    if moment is not None else None),
        optimizer_second_moment_positive_elements=positive_second,
        optimizer_second_moment_positive_fraction=positive_second / n,
        optimizer_second_moment_mean=(float(second.double().mean()) if second is not None else None),
        historical_nonzero_gradient_established=positive_second > 0)


def summarize_groups(rows):
    result = []
    for name in dict.fromkeys(row['group'] for row in rows):
        members = [row for row in rows if row['group'] == name]
        n = sum(row['elements'] for row in members)
        base = sum(row['initial_L2_squared'] for row in members)
        delta = sum(row['delta_L2_squared'] for row in members)
        result.append(dict(group=name, parameter_tensors=len(members), elements=n,
            changed_elements=sum(row['changed_elements'] for row in members),
            trained_nonzero_elements=sum(row['trained_nonzero_elements'] for row in members),
            delta_RMS=math.sqrt(delta / n),
            delta_to_initial_L2_ratio=math.sqrt(delta / base) if base else None,
            tensors_with_historical_nonzero_gradient=sum(
                row['historical_nonzero_gradient_established'] for row in members),
            optimizer_second_moment_positive_elements=sum(
                row['optimizer_second_moment_positive_elements'] for row in members),
            optimizer_steps=sorted({row['optimizer_steps'] for row in members},
                                   key=lambda value: -1 if value is None else value)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-id', action='append', required=True)
    parser.add_argument('--through-epoch', type=int, choices=[6, 9, 12], required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('This diagnostic requires CUDA_VISIBLE_DEVICES empty and uninitialized CUDA')
    started = time.monotonic()
    args.output_directory.mkdir(exist_ok=False)
    plan_path = ROOT / 'configs/perceptual_mft_20261122.json'
    study_path = ROOT / 'configs/perceptual_native_pyramid_study_20261124.json'
    plan, study = read(plan_path), read(study_path)
    work = Path(plan['work_root'])
    registry_path = work / 'native_pyramid_pipeline_registered.json'
    registration = read(registry_path)
    state = read(work / 'native_pyramid_study_state.json')
    files = {str(p): sha(p) for p in [plan_path, study_path, registry_path, Path(__file__)]}
    for name, expected in registration['source_hashes'].items():
        path = ROOT / name
        if sha(path) != expected:
            raise ValueError('Registered native source changed: ' + name)
        files[str(path)] = expected
    fits = {row['id']: row for row in study['normal_fits']}
    if len(set(args.fit_id)) != len(args.fit_id) or not set(args.fit_id) <= set(fits):
        raise ValueError('Select distinct registered normal fits only')
    rows = []
    for name in args.fit_id:
        phase = state['phases']['NORMAL_TRAIN_' + str(args.through_epoch).zfill(2)]
        job = phase[name]
        actual = process(job['pid'])
        if (job['status'] != 'COMPLETE' or job.get('exit_code') != 0 or
                (actual['present'] and actual['kernel_state'] != 'Z' and
                 actual['command'] == ' '.join(job['command']))):
            raise ValueError('Only finished original fits with recorded actual exit0 may be inspected')
        fit_path = ROOT / 'configs/perceptual_native_pyramid_fit_cases_20261124' / (name + '.json')
        config = read(fit_path)
        files[str(fit_path)] = sha(fit_path)
        folder = work / 'native_pyramid_fits' / name
        progress_path, latest_path = folder / 'progress.json', folder / 'latest.pt'
        snapshot_path = folder / ('epoch_' + str(args.through_epoch).zfill(2) + '.pt')
        captured_progress = progress_path.read_bytes()
        progress = json.loads(captured_progress)
        before = {str(path): sha(path) for path in [latest_path, snapshot_path]}
        latest = torch.load(latest_path, map_location='cpu', weights_only=False)
        snapshot = torch.load(snapshot_path, map_location='cpu', weights_only=False)
        metadata = progress['metadata']
        updates = args.through_epoch * study['Train_delivery']['clips_per_epoch']
        if (progress['status'] != 'COMPLETE_REGISTERED_TRAIN_STAGE' or
                progress['completed_full_Train_epochs'] != args.through_epoch or
                progress['actual_optimizer_updates'] != updates or
                latest['completed_epochs'] != args.through_epoch or latest['prefix'] is not None or
                latest['actual_optimizer_updates'] != updates or
                latest['delivered_clips_total'] != updates or
                latest['metadata'] != metadata or snapshot['metadata'] != metadata or
                latest['history'] != progress['history'] or snapshot['history'] != progress['history'] or
                metadata['fit_id'] != name or metadata['seed'] != 0 or
                metadata['fit_config_sha256'] != sha(fit_path) or
                metadata['source_hashes'] != registration['source_hashes'] or
                metadata['registry_sha256'] != sha(registry_path) or
                metadata['study_sha256'] != sha(study_path) or
                state_sha256(latest['module']) != state_sha256(snapshot['module']) or
                progress['checkpoint_sha256'] != before[str(snapshot_path)]):
            raise ValueError('Original completed-fit lineage, snapshots or update counts differ: ' + name)
        torch.manual_seed(0)
        initial_module = NativePyramidFusion(**config['module_kwargs']).float()
        if state_sha256(initial_module.state_dict()) != metadata['initial_native_sha256']:
            raise ValueError('Reconstructed seed0 initialization differs from original training: ' + name)
        parameters = list(initial_module.named_parameters())
        groups = latest['optimizer']['param_groups']
        if len(groups) != 1 or len(groups[0]['params']) != len(parameters):
            raise ValueError('AdamW parameter order no longer matches pinned module.parameters()')
        if groups[0].get('amsgrad') or groups[0].get('maximize'):
            raise ValueError('Unregistered AdamW variant')
        expected_lr = config['fit']['learning_rate'] * learning_rate_factor(
            updates - 1, config['fit']['horizon_updates'], config['fit']['warmup_updates'],
            config['fit']['minimum_lr_fraction'])
        if (groups[0]['weight_decay'] != config['fit']['weight_decay'] or
                not math.isclose(groups[0]['lr'], expected_lr, rel_tol=1e-12)):
            raise ValueError('Stored optimizer recipe differs: ' + name)
        parameter_rows = []
        for identifier, (parameter_name, initial) in zip(groups[0]['params'], parameters):
            trained = snapshot['module'][parameter_name]
            if trained.shape != initial.shape or not torch.isfinite(trained).all():
                raise ValueError('Invalid native parameter: ' + parameter_name)
            parameter_rows.append(parameter_row(parameter_name, initial, trained,
                                  latest['optimizer']['state'].get(identifier, {})))
        n = sum(row['elements'] for row in parameter_rows)
        if n != study['architecture'][metadata['mode'] + '_parameters']:
            raise ValueError('Actual parameter count differs')
        if (progress_path.read_bytes() != captured_progress or
                any(sha(path) != expected for path, expected in before.items())):
            raise ValueError('A completed fit changed during observation')
        files.update(before)
        files[str(progress_path)] = hashlib.sha256(captured_progress).hexdigest()
        projection = [row for row in parameter_rows if row['name'].startswith('output.')]
        rows.append(dict(fit_id=name, mode=metadata['mode'], cue=metadata['cue'],
            original_job_status=job['status'], original_recorded_exit_code=job['exit_code'],
            original_worker_observation=actual, completed_full_Train_epochs=args.through_epoch,
            actual_optimizer_updates=updates, native_parameters=n,
            initial_native_sha256=metadata['initial_native_sha256'],
            trained_native_sha256=state_sha256(snapshot['module']),
            source_RGB_sha256=metadata['source_RGB_sha256'],
            immutable_epoch_checkpoint=str(snapshot_path), optimizer_latest=str(latest_path),
            optimizer_latest_may_be_replaced_by_future_registered_continuation=True,
            optimizer_state_binary_not_copied=True,
            stored_optimizer_lr=groups[0]['lr'], stored_optimizer_weight_decay=groups[0]['weight_decay'],
            parameter_dtypes=dict(Counter(row['trained_dtype'] for row in parameter_rows)),
            tensors_with_historical_nonzero_gradient=sum(
                row['historical_nonzero_gradient_established'] for row in parameter_rows),
            total_parameter_tensors=len(parameter_rows),
            tensors_without_established_nonzero_gradient=[row['name'] for row in parameter_rows
                if not row['historical_nonzero_gradient_established']],
            optimizer_step_counts=sorted({row['optimizer_steps'] for row in parameter_rows},
                                         key=lambda value: -1 if value is None else value),
            nonzero_output_projection_elements=sum(row['trained_nonzero_elements'] for row in projection),
            total_output_projection_elements=sum(row['elements'] for row in projection),
            mean_clip_losses=[dict(epoch=h['epoch'], **h['mean_clip_losses'])
                              for h in progress['history']],
            groups=summarize_groups(parameter_rows), parameters=parameter_rows))
        del latest, snapshot, initial_module
    if torch.cuda.is_initialized() or any(sha(path) != expected for path, expected in files.items()):
        raise ValueError('CUDA initialized or an inspected source/input changed')
    memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    if memory > 2:
        raise ValueError('CPU diagnostic exceeded its 2GiB RSS limit')
    report = dict(status='COMPLETE_NATIVE_PARAMETER_LEARNING_DIAGNOSTIC_NOT_ACCURACY',
        UTC=datetime.now(timezone.utc).isoformat(), source_hashes=files, rows=rows,
        seconds=time.monotonic() - started, peak_process_RSS_GiB=memory,
        fixed_seed=0, additional_seed_work=False,
        NN_forward_passes=0, backward_passes=0, optimizer_updates=0, GPU_calls=0,
        image_reads=0, image_feature_or_weight_copies=0, Dev_rows=0, Test_rows=0,
        training_signals_sent=False, fitted_recipe_changed=False,
        new_Dev_or_Test_accuracy=False, supported_innovations=0, goal_met=False,
        scope=('Read only original completed native fits and optimizer states. '
               'Positive AdamW second moments establish historical nonzero gradients separately '
               'from decoupled weight decay, not sustained gradient size, effective feature or '
               'prediction changes, learned useful cue information, modality benefit or convergence. '
               'FP16 inference corrections require actual forward observations during registered Dev. '
               'Optimizer binary states are hashed but not copied; future registered continuation '
               'may replace latest.pt. No checkpoint selection or full-study completion claim.'))
    output = args.output_directory / 'complete.json'
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(status=report['status'], fits=args.fit_id, seconds=report['seconds'],
                         peak_process_RSS_GiB=memory)))


if __name__ == '__main__':
    main()
