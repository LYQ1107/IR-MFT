#!/usr/bin/env python3
"""Six exact zero controls then18 unchanged-input multi-anchor Dev repairs."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.run_pseudo_ir_comparison import dump, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    spec_path = ROOT/'configs/perceptual_trajectory_consensus_20261123.json'
    spec = json.loads(spec_path.read_text())
    assert sha256(ROOT/spec['data_plan']) == spec['data_plan_sha256']
    state_path = work/'trajectory_consensus_dev_state.json'
    lock = state_path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not state_path.exists()
    folder = ROOT/'configs/perceptual_trajectory_consensus_dev_20261123'; folder.mkdir(exist_ok=False)
    sources = {name: sha256(ROOT/name) for name in ['tools/replay_trajectory_consensus.py',
        'yolox/tracker/trajectory_consensus.py', 'yolox/tracker/calibrated_foreground_tracker.py']}
    jobs = {}
    for enabled, seeds in [(False, [0]), (True, [0, 1, 2])]:
        for observation in ['rgb', 'gray', 'ir']:
            for cue in ['rgb', 'ir']:
                for seed in seeds:
                    parent_key = f'CG_{observation}_{cue}_s{seed}_birth07_clear'
                    parent = ROOT/'configs/perceptual_correlation_backend_guarded_dev_20261123'/(parent_key+'.json')
                    config = json.loads(parent.read_text())
                    key = f"TR_{observation}_{cue}_s{seed}_"+('full' if enabled else 'zero')
                    config.update(id=key, backend='trajectory_consensus', trajectory_consensus=enabled,
                        history_consensus=True, mutual_competition=True, consensus_plan_sha256=sha256(spec_path),
                        source_guard_config_sha256=sha256(parent), control_parent_id=parent_key,
                        registration='Fixed multi-anchor and mutual-gate repair, no refit/threshold search. All3observationcues/bothassociationcues/3associationseeds. Zero module must reproduce original full outputs exactly before18 full cases.')
                    path = folder/(key+'.json'); dump(path, config)
                    jobs[key] = dict(status='PENDING', phase='full' if enabled else 'zero',
                                     config=str(path), sha256=sha256(path), parent=parent_key)
    assert len(jobs) == 24
    state = dict(status='WAITING_FOR_ALL36_GUARDED_CONTROLS', jobs=jobs, sources=sources,
                 specification_sha256=sha256(spec_path), Test_read=False)
    dump(state_path, state)
    parent_state = work/'correlation_backend_guarded_dev_state.json'
    while True:
        parent = json.loads(parent_state.read_text())
        if parent['status'] == 'FAILED':
            state['status'] = 'PARENT_FAILED'; dump(state_path, state); return
        if parent['status'] == 'COMPLETE':
            break
        time.sleep(5)
    state['completed_parent_state_sha256'] = sha256(parent_state)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', MFT26IR_ROOT=plan['data_root'])
    for phase in ['zero', 'full']:
        state['status'] = 'RUNNING_'+phase.upper(); active = {}; dump(state_path, state)
        while True:
            for key, proc in list(active.items()):
                code = proc.poll()
                if code is not None:
                    jobs[key].update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code); del active[key]
            for key, job in jobs.items():
                if len(active) >= 2 or any(j['status'] == 'FAILED' for j in jobs.values()):
                    break
                if job['phase'] != phase or job['status'] != 'PENDING':
                    continue
                assert sha256(job['config']) == job['sha256'] and sha256(spec_path) == state['specification_sha256']
                assert all(sha256(ROOT/name) == digest for name, digest in sources.items())
                command = [sys.executable, '-u', 'tools/replay_trajectory_consensus.py', '--config', job['config'],
                           '--output', str(work/'trajectory_consensus_dev'/key)]
                log = work/(key+'.trajectory_consensus_dev.log')
                with log.open('ab', buffering=0) as handle:
                    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                active[key] = proc; job.update(status='RUNNING', pid=proc.pid, command=command, log=str(log))
            dump(state_path, state)
            if not active:
                if any(j['status'] == 'FAILED' for j in jobs.values()):
                    state['status'] = 'FAILED'; dump(state_path, state); return
                assert all(j['status'] == 'COMPLETE' for j in jobs.values() if j['phase'] == phase)
                break
            time.sleep(5)
        if phase == 'zero':
            proof = {}
            for key, job in jobs.items():
                if job['phase'] != 'zero':
                    continue
                current = work/'trajectory_consensus_dev'/key
                reference = work/'correlation_backend_guarded_dev'/job['parent']
                a, b = [json.loads((p/'metrics.json').read_text()) for p in [current, reference]]
                assert a['trackeval'] == b['trackeval'] and len(a['trackeval']) == 39
                for relative in ['track_results/BT-005.txt', 'track_results/SN-013.txt', 'trackeval_output/pedestrian_detailed.csv']:
                    assert sha256(current/relative) == sha256(reference/relative)
                proof[key] = dict(all39_metrics_and_both_track_files_exact=True, complete4834_frames=True,
                                  detailed_CSV_exact=True, parent_metrics_sha256=sha256(reference/'metrics.json'))
            state['all6_full_Dev_zero_module_parities'] = proof; dump(state_path, state)
    state['status'] = 'COMPLETE'; dump(state_path, state)


if __name__ == '__main__':
    main()
