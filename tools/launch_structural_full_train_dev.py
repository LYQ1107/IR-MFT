#!/usr/bin/env python3
"""Full49-stream source audits, then all50 complete-Dev tracking evaluations."""
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
from tools.correlation_geometry_checks import compare_packets, fresh_RGB_audit
from tools.run_pseudo_ir_comparison import dump, sha256


def proofs(base, sequence, plan, specification, frames=None):
    canonical = Path(plan['frozen_detection_sources']['rgb_dev']) / 'detections' / (sequence + '.npz')
    result = {}
    count = 0
    for row in specification['models']:
        if row.get('baseline'):
            continue
        name = row['id']
        if row.get('intervention') == 'unavailable':
            result[name + '_raw'] = compare_packets(base / (name + '_raw.npz'), base / 'CG3_source_RGB.npz', frames)
            result[name + '_priority'] = compare_packets(base / (name + '.npz'), canonical, frames)
        with np.load(base / (name + '_raw.npz')) as z:
            assert len(z['boxes']) == len(z['anchor_indices']) == len(z['rgb_reference_scores'])
            assert ((z['anchor_indices'] >= 0) & (z['anchor_indices'] < 23625)).all()
            assert ((z['rgb_reference_scores'] >= 0) & (z['rgb_reference_scores'] <= 1)).all()
        count += 1
    assert count == 24 and len(result) == 6
    result['dense_RGB_geometry'] = dict(streams=count, every_raw_selected_anchor_checked_before_publication=True)
    return result


def main():
    plan = json.loads((ROOT / 'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    spec_path = ROOT / 'configs/perceptual_structural_fulltrain_dev_20261123.json'
    specification = json.loads(spec_path.read_text())
    state_path = work / 'structural_full_train_dev_state.json'
    lock = state_path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not state_path.exists()
    sources = ['tools/launch_structural_full_train_dev.py', 'tools/extract_structural_full_train.py',
        'tools/replay_structural_full_train.py', 'tools/extract_structural_relative.py',
        'tools/structural_relative_inference.py', 'tools/structural_grid.py',
        'tools/structural_correspondence.py', 'tools/build_correlation_cache.py',
        'tools/indexed_postprocess.py', 'tools/correlation_geometry_checks.py',
        'tools/structural_gate_observer.py', 'yolox/models/structural_factorial_repair.py',
        'yolox/models/structural_relative_repair.py', 'yolox/models/frozen_head_repair.py',
        'yolox/models/yolo_head.py', 'yolox/tracker/foveated_regions.py',
        'yolox/tracker/priority_observations.py', 'yolox/tracker/byte_tracker.py']
    config_dir = ROOT / 'configs/perceptual_structural_fulltrain_dev_cases_20261123'
    jobs = {}
    for path in sorted(config_dir.glob('*.json')):
        case = json.loads(path.read_text())
        assert case['specification_sha256'] == sha256(spec_path) and case['seed'] == 0
        jobs[path.stem] = dict(status='PENDING', config=str(path), sha256=sha256(path))
    assert len(jobs) == 50
    state = dict(status='WAITING_FOR_SIX_FULL_TRAIN_FITS', controller_pid=os.getpid(), jobs=jobs,
        producers={}, source_hashes={name: sha256(ROOT / name) for name in sources},
        specification_sha256=sha256(spec_path), fixed_seed=0, new_Test_accuracy_used=False,
        goal_met=False, supported_innovations=0)
    dump(state_path, state)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MFT26IR_ROOT=plan['data_root'])
    try:
        while True:
            fit_path = work / 'structural_full_train_fit_state.json'
            fits = json.loads(fit_path.read_text()) if fit_path.exists() else {}
            if 'FAIL' in fits.get('status', ''):
                raise RuntimeError('Preserve failed full-Train fitting')
            if fits.get('status') == 'COMPLETE':
                break
            time.sleep(5)
        assert sha256(spec_path) == state['specification_sha256']
        assert all(sha256(ROOT / name) == digest for name, digest in state['source_hashes'].items())
        assert shutil.disk_usage(work).free >= 40 * 2**30
        assert int(subprocess.check_output(['du', '-s', '-B1', str(work)], text=True).split()[0]) < 64 * 2**30
        base_command = [sys.executable, '-u', 'tools/extract_structural_full_train.py']
        state['status'] = 'CHECKING_SOURCE17_AND_TRAINED49_PREFIXES'
        dump(state_path, state)
        for sequence, physical in [('BT-005', '4'), ('SN-013', '5')]:
            for mode in ['source', 'prefix']:
                command = base_command + ['--sequence', sequence]
                command += (['--source-check', '8'] if mode == 'source' else
                            ['--specification', str(spec_path), '--limit-frames', '32'])
                log = work / (sequence + '.full_train_' + mode + '.log')
                with log.open('xb', buffering=0) as stream:
                    subprocess.run(command, cwd=ROOT, env=dict(environment, CUDA_VISIBLE_DEVICES=physical),
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, check=True)
            base = work / 'full_train_observations/structural_relative_predictions/dev' / sequence
            meta = json.loads((base / 'source_check_8/complete.json').read_text())
            assert len(meta['outputs']) == 17 and all(row['array_exact'] for row in meta['fresh_RGB_parity'])
            prefix = base / 'prefix_check_32'
            meta = json.loads((prefix / 'complete.json').read_text())
            assert len(meta['outputs']) == 49 and meta['frames'] == 32
            state.setdefault('prefix_proofs', {})[sequence] = proofs(prefix, sequence, plan, specification, 32)
        active = {}
        state['status'] = 'PRODUCING_ALL49_STREAMS_WITHOUT_GT'
        for sequence, physical in [('BT-005', '4'), ('SN-013', '5')]:
            command = base_command + ['--sequence', sequence, '--specification', str(spec_path)]
            log = work / (sequence + '.full_train_full_dev.log')
            with log.open('xb', buffering=0) as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=dict(environment, CUDA_VISIBLE_DEVICES=physical),
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            active[sequence] = process
            state['producers'][sequence] = dict(status='RUNNING', pid=process.pid, physical_gpu=physical,
                                               command=command, log=str(log))
        dump(state_path, state)
        while active:
            for sequence, process in list(active.items()):
                code = process.poll()
                if code is not None:
                    state['producers'][sequence].update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code)
                    del active[sequence]
            dump(state_path, state)
            time.sleep(5)
        assert all(p['status'] == 'COMPLETE' for p in state['producers'].values())
        assert all(sha256(ROOT / name) == digest for name, digest in state['source_hashes'].items())
        for sequence in ['BT-005', 'SN-013']:
            base = work / 'full_train_observations/structural_relative_predictions/dev' / sequence
            meta = json.loads((base / 'complete.json').read_text())
            assert meta['specification_sha256'] == state['specification_sha256']
            assert meta['source_hashes'] == {key: state['source_hashes'][key] for key in meta['source_hashes']}
            audit = proofs(base, sequence, plan, specification)
            audit['all49_prefix_exact'] = {key: compare_packets(base / 'prefix_check_32' / (key + '.npz'),
                base / (key + '.npz'), 32) for key in meta['outputs']}
            assert len(audit['all49_prefix_exact']) == 49
            audit['fresh_RGB'] = fresh_RGB_audit(base, sequence, plan, work)
            state.setdefault('full_proofs', {})[sequence] = audit
        state['status'] = 'RUNNING_ALL50_COMPLETE_DEV_REPLAYS'
        # The canonical RGB replay was run as an engineering canary. Reuse it
        # only with its immutable config and complete coverage, never refit it.
        canary = 'FT_source_RGB_canonical'
        canary_path = work / 'structural_full_train_dev' / canary / 'metrics.json'
        if canary_path.exists():
            canary_data = json.loads(canary_path.read_text())
            assert canary_data['status'] == 'COMPLETE' and canary_data['manifest']['frames'] == 4834
            assert canary_data['manifest']['config_sha256'] == jobs[canary]['sha256']
            jobs[canary].update(status='COMPLETE', reused_engineering_canary=True, metrics_sha256=sha256(canary_path))
        active = {}
        while True:
            for key, process in list(active.items()):
                code = process.poll()
                if code is not None:
                    jobs[key].update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code)
                    del active[key]
            for key, job in jobs.items():
                if len(active) >= 2 or any(v['status'] == 'FAILED' for v in jobs.values()):
                    break
                if job['status'] != 'PENDING':
                    continue
                assert sha256(job['config']) == job['sha256']
                command = [sys.executable, '-u', 'tools/replay_structural_full_train.py', '--config', job['config'],
                           '--output', str(work / 'structural_full_train_dev' / key)]
                log = work / (key + '.full_train_replay.log')
                with log.open('xb', buffering=0) as stream:
                    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                active[key] = process
                job.update(status='RUNNING', pid=process.pid, command=command, log=str(log))
            dump(state_path, state)
            if not active:
                assert all(v['status'] == 'COMPLETE' for v in jobs.values())
                break
            time.sleep(5)
        state['status'] = 'COMPLETE'
        dump(state_path, state)
    except Exception as error:
        state.update(status='FAILED_PRESERVE_FULL_TRAIN_DEV', error=repr(error))
        dump(state_path, state)
        raise


if __name__ == '__main__':
    main()
