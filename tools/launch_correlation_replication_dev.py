#!/usr/bin/env python3
"""All additional matched-seed Dev curves, with complete seed0 anchor parity."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from tools.run_pseudo_ir_comparison import dump, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    spec_path = ROOT/'configs/perceptual_correlation_replication_eval_20261123.json'
    spec = json.loads(spec_path.read_text())
    fit_plan = ROOT/'configs/perceptual_correlation_replication_20261123.json'
    assert sha256(fit_plan) == spec['replication_plan_sha256']
    source = ROOT/'tools/extract_correlation_policies.py'
    state_path = work/'correlation_replication_dev_state.json'
    lock = state_path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not state_path.exists()
    folder = ROOT/'configs/perceptual_correlation_replication_dev_20261123'
    folder.mkdir(exist_ok=False)
    jobs = {}
    for row in spec['models']:
        if row.get('anchor'):
            continue
        for policy in ['priority', 'raw']:
            key = row['id'] + ('_raw' if policy == 'raw' else '')
            config = dict(row, id=key, proposal_key=key, output_policy=policy,
                          producer_directory='correlation_replication_predictions',
                          fusion_plan_sha256=spec['config_sha256'], specification_sha256=sha256(spec_path),
                          registration='All matched seeds1/2, four checkpoints, both output policies; reuse all original seed0 curves. Complete Dev only, no Test choice or established contribution.')
            path = folder/(key+'.json'); dump(path, config)
            jobs[key] = dict(status='PENDING', config=str(path), sha256=sha256(path))
    assert len(jobs) == 48
    state = dict(status='WAITING_FOR_ALL6_MATCHED_SEED_FITS', jobs=jobs, producers={},
                 specification_sha256=sha256(spec_path), source_sha256=sha256(source), Test_read=False)
    dump(state_path, state)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', MFT26IR_ROOT=plan['data_root'])
    while True:
        fit = json.loads((work/'correlation_replication_fit_state.json').read_text())
        if fit['status'] == 'FAILED':
            state['status'] = 'FIT_DEPENDENCY_FAILED'; dump(state_path, state); return
        if fit['status'] == 'COMPLETE':
            break
        time.sleep(5)
    assert sha256(spec_path) == state['specification_sha256'] and sha256(source) == state['source_sha256']
    used = sum(p.lstat().st_size for p in work.rglob('*') if not p.is_symlink() and p.is_file())
    if used > 64*2**30 or shutil.disk_usage(work).free < 40*2**30:
        raise RuntimeError('User work-cache cap or independent-disk reserve exceeded')
    state['status'] = 'SOURCE_CHECKS_BEFORE_FULL_INFERENCE'; dump(state_path, state)
    for sequence, physical in [('BT-005', '2'), ('SN-013', '3')]:
        command = [sys.executable, '-u', str(source), '--sequence', sequence, '--source-check', '8',
                   '--producer-name', 'correlation_replication_predictions']
        log = work/(sequence+'.replication_source_check.log')
        with log.open('ab', buffering=0) as handle:
            code = subprocess.call(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=physical),
                                   stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
        assert code == 0, str(log)
        check = json.loads((work/'correlation_replication_predictions/dev'/sequence/'source_check_8/complete.json').read_text())
        assert check['source_sha256'] == state['source_sha256'] and len(check['outputs']) == 7
        assert all(r['array_exact'] for r in check['fresh_RGB_parity'])
    state['status'] = 'PRODUCING_ALL55_DEV_PROPOSAL_STREAMS'; active = {}
    for sequence, physical in [('BT-005', '2'), ('SN-013', '3')]:
        command = [sys.executable, '-u', str(source), '--sequence', sequence, '--specification', str(spec_path),
                   '--producer-name', 'correlation_replication_predictions']
        log = work/(sequence+'.correlation_replication_full_dev.log')
        with log.open('ab', buffering=0) as handle:
            proc = subprocess.Popen(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=physical),
                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        active[sequence] = proc
        state['producers'][sequence] = dict(status='RUNNING', pid=proc.pid, physical_gpu=physical, command=command, log=str(log))
    dump(state_path, state)
    while active:
        for key, proc in list(active.items()):
            code = proc.poll()
            if code is not None:
                state['producers'][key].update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code)
                del active[key]
        dump(state_path, state); time.sleep(5)
    if any(p['status'] != 'COMPLETE' for p in state['producers'].values()):
        state['status'] = 'PRODUCER_FAILED'; dump(state_path, state); return
    proofs = {}; source_proofs = {}
    for sequence in ['BT-005', 'SN-013']:
        base = work/'correlation_replication_predictions/dev'/sequence
        meta = json.loads((base/'complete.json').read_text())
        assert meta['source_sha256'] == state['source_sha256'] and meta['specification_sha256'] == state['specification_sha256']
        proofs[sequence] = []
        for row in spec['models']:
            if not row.get('anchor'):
                continue
            if row.get('baseline'):
                new_path = base/(row['id']+'.npz')
                reference = Path(plan['frozen_detection_sources']['rgb_dev'])/'detections'/(sequence+'.npz')
                old_audit = json.loads((work/'correlation_RGB_parity'/(sequence+'.json')).read_text())
                allowed = {r['frame']: r for r in old_audit['mismatches']}; mismatches = []
                with np.load(new_path) as new, np.load(reference) as old:
                    for field in ['frame_ids', 'offsets']:
                        np.testing.assert_array_equal(new[field], old[field])
                    for frame in new['frame_ids']:
                        lo, hi = map(int, new['offsets'][frame-1:frame+1])
                        a = np.c_[new['boxes'][lo:hi], new['scores'][lo:hi]]
                        b = np.c_[old['boxes'][lo:hi], old['scores'][lo:hi]]
                        if np.array_equal(a, b):
                            continue
                        assert int(frame) in allowed
                        previous = allowed[int(frame)]
                        assert float(abs(a[:, :4]-b[:, :4]).max()) <= previous['max_box_difference']+1e-7
                        assert float(abs(a[:, 4]-b[:, 4]).max()) <= previous['max_score_difference']+1e-7
                        for threshold in [.6, .7]:
                            np.testing.assert_array_equal(a[:, 4] > threshold, b[:, 4] > threshold)
                        mismatches.append(int(frame))
                source_proofs[sequence] = dict(known_tail_mismatch_frames=mismatches,
                    preexisting_full_audit_sha256=sha256(work/'correlation_RGB_parity'/(sequence+'.json')),
                    reference_sha256=sha256(reference), new_sha256=sha256(new_path),
                    scope='Counts/offsets exact all frames; only pre-existing tail-batch numerical differences allowed, bounded by original audit and no0.6/0.7 membership changes.')
                continue
            suffixes = [''] if row.get('baseline') else ['', '_raw']
            for suffix in suffixes:
                new_path = base/(row['id']+suffix+'.npz')
                old_path = work/'correlation_policy_predictions/dev'/sequence/(row['reference_model_id']+suffix+'.npz')
                with np.load(new_path) as new, np.load(old_path) as old:
                    for field in ['frame_ids', 'offsets', 'boxes', 'scores']:
                        np.testing.assert_array_equal(new[field], old[field])
                proofs[sequence].append(dict(id=row['id']+suffix, all4_arrays_exact=True,
                                             new_sha256=sha256(new_path), reference_sha256=sha256(old_path)))
        assert len(proofs[sequence]) == 6
    state['seed0_anchor_exact_all6_streams_both_full_videos'] = proofs
    state['source_RGB_full_numerical_compatibility'] = source_proofs
    state['status'] = 'RUNNING'; active = {}; dump(state_path, state)
    while True:
        for key, proc in list(active.items()):
            code = proc.poll()
            if code is not None:
                jobs[key].update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code); del active[key]
        for key, job in jobs.items():
            if len(active) >= 2 or any(j['status'] == 'FAILED' for j in jobs.values()):
                break
            if job['status'] != 'PENDING':
                continue
            assert sha256(job['config']) == job['sha256']
            command = [sys.executable, '-u', 'tools/replay_correlation_observations.py', '--config', job['config'],
                       '--output', str(work/'correlation_replication_dev'/key)]
            log = work/(key+'.correlation_replication_replay.log')
            with log.open('ab', buffering=0) as handle:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            active[key] = proc; job.update(status='RUNNING', pid=proc.pid, command=command, log=str(log))
        if not active:
            if all(j['status'] == 'COMPLETE' for j in jobs.values()):
                state['status'] = 'COMPLETE'
            elif any(j['status'] == 'FAILED' for j in jobs.values()):
                state['status'] = 'FAILED'
        dump(state_path, state)
        if state['status'] != 'RUNNING':
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
