#!/usr/bin/env python3
"""Preserve the input-guard failure, reuse completed controls and redo failed IR.

Successful work keeps its exact checkpoints. The failed partial IR fit is not
resumed: its lost RAM bytes cannot be reconstructed. Seed0, all addresses,
optimizer, losses and the original fitter bytecode remain unchanged.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from tools.launch_structural_full_train_fits import draw_digests
from tools.run_pseudo_ir_comparison import dump, sha256

REGISTRY_NAME = 'structural_content_guard_recovery_registered.json'
ARCHIVE_NAME = 'structural_input_metadata_guard_failure_20261124'


def read(path):
    return json.loads(Path(path).read_text())


def context():
    plan = read(ROOT / 'configs/perceptual_mft_20261122.json')
    work = Path(plan['work_root'])
    study_path = ROOT / 'configs/perceptual_structural_fulltrain_study_20261123.json'
    study = read(study_path)
    assert sha256(ROOT / study['configuration']) == study['configuration_sha256']
    parent_path = work / 'structural_full_train_pipeline_registered.json'
    parent = read(parent_path)
    assert all(sha256(ROOT / p) == h for p, h in parent['source_hashes'].items())
    schedule = read(work / 'structural_full_train_schedule/complete.json')
    assert sha256(schedule['address_file']['path']) == schedule['address_file']['sha256']
    with np.load(schedule['address_file']['path']) as saved:
        addresses = saved['addresses']
    return plan, work, study_path, study, parent, schedule, addresses


def verify_complete(work, name, expected, study, study_path):
    path = work / 'structural_full_train_fit' / name / 'full_train_complete.json'
    proof = read(path)
    assert proof['status'] == 'COMPLETE_THREE_FULL_TRAIN_TRAVERSALS'
    assert proof['epochs'] == 72 and proof['successful_steps_this_invocation'] == 20736
    assert proof['logical_Train_epochs_completed'] == 3 and proof['full_frame_address_delivery_verified']
    assert proof['full_encoder_frozen_no_grad'] and proof['post_loop_source_verified']
    assert proof['initial_raw_RGB_head_bit_exact'] and proof['trainable_parameters'] == 145600
    assert proof['full_train_study_sha256'] == sha256(study_path)
    assert proof['full_train_address_sha256'] == study['semantic_address_sha256']
    assert [r['sample_digest'] for r in proof['history']] == expected
    assert [r['actual_frame_draw_sha256'] for r in proof['live_frame_passes']] == expected
    assert all(r['complete_input_pass'] for r in proof['live_frame_passes'])
    assert all(sha256(ROOT / p) == h for p, h in proof['source_hashes'].items())
    snapshots = {}
    for epoch in [24, 48, 72]:
        weights = path.parent / f'epoch{epoch:02d}.pt'
        saved = torch.load(weights, map_location='cpu', weights_only=False)
        assert saved['completed_epochs'] == epoch and saved['config_sha256'] == study['configuration_sha256']
        assert saved['metadata']['seed'] == 0 and saved['metadata']['variant'] == proof['variant']
        assert saved['metadata']['cue'] == proof['cue']
        assert sum(t.numel() for t in saved['adapter'].values()) == 145600
        assert all(torch.isfinite(t).all() for t in saved['adapter'].values())
        snapshots[str(epoch)] = dict(sha256=sha256(weights), bytes=weights.stat().st_size)
    return dict(full_train_complete_sha256=sha256(path), snapshots=snapshots,
        updates=20736, logical_Train_epochs=3, all72_actual_address_digests_exact=True)


def register():
    plan, work, study_path, study, parent, schedule, addresses = context()
    path = work / REGISTRY_NAME
    assert not path.exists()
    failed = read(work / 'structural_full_train_fit_state.json')
    assert failed['status'] == 'FAILED_PRESERVE_FULL_TRAIN_STUDY'
    assert failed['jobs']['ir_111_s0']['fit_exit_code'] == 1
    assert not (Path('/proc') / str(failed['controller_pid'])).exists()
    for job in failed['jobs'].values():
        if job.get('pid'):
            assert not (Path('/proc') / str(job['pid'])).exists()
    proof_path = work / 'content_verified_train_reader_CPU_proof.json'
    proof = read(proof_path)
    assert proof['status'] == 'COMPLETE_CONTENT_VERIFIED_READER_CPU_PROOF'
    assert proof['unchanged_real_Train_pixels_exact'] and proof['changed_bytes_rejected']
    assert all(sha256(ROOT / p) == h for p, h in proof['source_hashes'].items())
    expected = draw_digests(schedule['sequence_order'], addresses)
    reused = {name: verify_complete(work, name, expected, study, study_path)
              for name in ['rgb_111_s0', 'gray_111_s0']}
    archive = work / ARCHIVE_NAME
    archive.mkdir(exist_ok=False)
    records = {}
    for name in ['structural_full_train_fit_state.json', 'structural_full_train_dev_state.json',
        'structural_full_train_next_stage_state.json', 'structural_legacy_backfill_state.json',
        'native_pyramid_study_state.json']:
        source = work / name
        value = read(source)
        assert 'FAIL' in value['status'] and not (Path('/proc') / str(value['controller_pid'])).exists()
        if name == 'native_pyramid_study_state.json':
            assert not value['phases'], 'No native children may be discarded'
        shutil.copy2(source, archive / name)
        records[name] = sha256(source)
    for name in ['structural_full_train_pipeline_registered.json', 'native_pyramid_pipeline_registered.json',
        'native_pyramid_CPU_workflow_tests.json', 'native_pyramid_CPU_workflow_tests.log',
        'launch_structural_full_train_fits.log', 'launch_structural_full_train_dev.log',
        'ir_111_s0.structural_full_train_fit.log', 'gray_111_s0.structural_full_train_fit.log']:
        source = work / name
        if source.exists():
            shutil.copy2(source, archive / name)
            records[name] = sha256(source)
    partial = work / 'structural_full_train_fit/ir_111_s0'
    failure = read(partial / 'failure.json')
    assert 'Original Train image changed during RAM reuse' in failure['error']
    saved = torch.load(partial / 'latest.pt', map_location='cpu', weights_only=False)
    assert saved['completed_epochs'] == 5 and len(saved['history']) == 5
    assert [r['sample_digest'] for r in saved['history']] == expected[:5]
    shutil.move(str(partial), str(archive / 'failed_ir_111_s0'))
    sources = {name: sha256(ROOT / name) for name in ['tools/content_verified_train_reader.py',
        'tools/train_structural_content_verified.py', 'tools/recover_full_train_content_guard.py',
        'tests/test_content_verified_train_reader.py']}
    registry = dict(status='REGISTERED_CONTENT_GUARD_RECOVERY_WITH_UNCHANGED_ORIGINAL_FITTER',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), archive=str(archive), archived_sha256=records,
        parent_pipeline_sha256=sha256(work / 'structural_full_train_pipeline_registered.json'),
        reader_CPU_proof_sha256=sha256(proof_path), source_hashes=sources, reused_complete_fits=reused,
        source_failed_state=failed, full_train_study_sha256=sha256(study_path),
        failed_IR_preserved_completed_macro_passes=5, failed_IR_logged_successful_updates_at_least=1536,
        failed_IR_successful_updates_at_most=1567, partial_failed_IR_not_resumed=True,
        fixed_seed=0, additional_seed_work=False, actual_failed_attempt_cost_preserved=True,
        scope='RGB/gray completed artifacts independently verified. IR interrupted by a metadata guard; '
            'its exact historical cached bytes and offending file were not recorded. Redo only unfinishedIR '
            'from seed0 using exact byte-content validation, not a model-selection retry. '
            'Original files/readers, plans, loss/optimizer/fitter bytecode, schedules and completed controls unchanged. '
            'No image dataset copies and no changed bytes accepted. Other formal queues restore after registration.',
        goal_met=False, supported_innovations=0)
    dump(path, registry)
    print(json.dumps(dict(status=registry['status'], reused=list(reused), archive=str(archive))), flush=True)


def run():
    plan, work, study_path, study, parent, schedule, addresses = context()
    registry_path = work / REGISTRY_NAME
    registry = read(registry_path)
    assert all(sha256(ROOT / p) == h for p, h in registry['source_hashes'].items())
    path = work / 'structural_full_train_fit_state.json'
    lock = path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert sha256(path) == registry['archived_sha256'][path.name]
    expected = draw_digests(schedule['sequence_order'], addresses)
    smoke_digest = draw_digests(schedule['sequence_order'], addresses[:1], limit=20)[0]
    jobs = {f"{r['cue']}_{r['components']}_s0": dict(r, status='PENDING') for r in study['fits']}
    for name, proof in registry['reused_complete_fits'].items():
        job = registry['source_failed_state']['jobs'][name]
        jobs[name].update(job, status='COMPLETE', fit_complete_sha256=proof['full_train_complete_sha256'],
            independent_completion_proof_verified=True,
            exit_code_scope='Original RGB exit0 recorded; gray exit code unavailable after parent failure; complete artifacts verified.')
    state = dict(status='FITTING_SIX_MATCHED_FULL_TRAIN_CONFIGURATIONS', controller_pid=os.getpid(), jobs=jobs,
        source_hashes=dict(parent['source_hashes'], **registry['source_hashes']),
        full_train_study_sha256=sha256(study_path), recovery_registry_sha256=sha256(registry_path),
        fixed_seed=0, additional_seed_work=False, logical_Train_epochs_each=3, updates_each=20736,
        goal_met=False, supported_innovations=0)
    dump(path, state)
    active = {}
    try:
        while True:
            assert all(sha256(ROOT / p) == h for p, h in state['source_hashes'].items())
            for gpu, (name, phase, process) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[gpu]
                job = jobs[name]
                job[phase+'_exit_code'] = code
                assert code == 0, job['log']
                folder = work / 'structural_full_train_fit' / name
                if phase == 'smoke':
                    proof_path = folder / 'smoke_20/full_train_complete.json'
                    proof = read(proof_path)
                    assert proof['status'] == 'FULL_TRAIN_SMOKE20_VERIFIED'
                    assert proof['successful_steps_this_invocation'] == 20
                    assert proof['history'][0]['sample_digest'] == smoke_digest
                    assert proof['initial_raw_RGB_head_bit_exact'] and proof['full_encoder_frozen_no_grad']
                    assert proof['source_image_content_manifest']['sha256'] == sha256(proof['source_image_content_manifest']['path'])
                    job.update(status='SMOKE_PASSED', smoke_complete_sha256=sha256(proof_path))
                else:
                    verify_complete(work, name, expected, study, study_path)
                    job.update(status='COMPLETE', fit_complete_sha256=sha256(folder / 'full_train_complete.json'))
            for gpu in ['4', '5']:
                if gpu in active:
                    continue
                pending = [(name, row) for name, row in jobs.items() if row['physical_gpu'] == gpu
                           and row['status'] in ['PENDING', 'SMOKE_PASSED']]
                if not pending:
                    continue
                name, job = pending[0]
                phase = 'smoke' if job['status'] == 'PENDING' else 'fit'
                assert shutil.disk_usage(work).free >= 40*2**30
                used = int(subprocess.check_output(['du', '-s', '-B1', str(work)], text=True).split()[0])
                assert used < 64*2**30
                command = [sys.executable, '-u', 'tools/train_structural_content_verified.py',
                    '--cue', job['cue'], '--components', job['components'], '--seed', '0']
                if phase == 'smoke':
                    command += ['--smoke-steps', '20']
                log = work / (name+'.content_guard_recovery_'+phase+'.log')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OPENBLAS_NUM_THREADS='1',
                    OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MFT26IR_ROOT=plan['data_root'])
                with log.open('xb', buffering=0) as stream:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = name, phase, process
                job.update(status=phase.upper()+'_RUNNING', pid=process.pid, command=command, log=str(log))
            if all(row['status'] == 'COMPLETE' for row in jobs.values()):
                state['status'] = 'COMPLETE'
            dump(path, state)
            if state['status'] == 'COMPLETE':
                break
            time.sleep(5)
    except Exception as error:
        state.update(status='FAILED_PRESERVE_FULL_TRAIN_STUDY', error=repr(error))
        dump(path, state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['register', 'run'])
    register() if parser.parse_args().action == 'register' else run()
