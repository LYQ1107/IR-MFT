#!/usr/bin/env python3
"""Complete the two additional matched adapter seeds on physical GPUs2/3."""
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
    spec_path = ROOT/'configs/perceptual_correlation_replication_20261123.json'
    spec = json.loads(spec_path.read_text())
    assert sha256(ROOT/spec['parent_plan']) == spec['parent_sha256']
    state_path = work/'correlation_replication_fit_state.json'
    lock = state_path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if state_path.exists():
        raise ValueError('Replication already registered')
    sources = {name: sha256(ROOT/name) for name in ['tools/train_correlation_repair.py',
        'yolox/models/local_correlation_repair.py', 'yolox/models/frozen_head_repair.py']}
    jobs = {}
    for physical, entries in spec['schedule'].items():
        for row in entries:
            key = f"{row['cue']}_full_s{row['seed']}"
            assert not (work/'correlation_fit'/key).exists()
            jobs[key] = dict(status='PENDING', physical_gpu=physical, **row)
    assert len(jobs) == 6
    state = dict(status='WAITING_FOR_ALLOWED_GPU_DEPENDENCIES', jobs=jobs,
                 specification_sha256=sha256(spec_path), sources=sources, Test_read=False)
    dump(state_path, state)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               MFT26IR_ROOT=plan['data_root'])
    active = {}
    while True:
        for physical, (key, proc) in list(active.items()):
            code = proc.poll()
            if code is not None:
                jobs[key].update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code)
                if code == 0:
                    fitted = json.loads((work/'correlation_fit'/key/'complete.json').read_text())
                    assert fitted['epochs'] == 24 and sum(r['steps'] for r in fitted['history']) == 6912
                    jobs[key]['complete_sha256'] = sha256(work/'correlation_fit'/key/'complete.json')
                del active[physical]
        temporal = json.loads((work/'temporal_geometry_dev_state.json').read_text())
        staged = json.loads((work/'correlation_recoverability_staged_dev_state.json').read_text())
        photo_path = work/'photometric_compatibility/state.json'
        photo = json.loads(photo_path.read_text()) if photo_path.exists() else {}
        ready2 = all(temporal.get('producers', {}).get(seq, {}).get('status') == 'COMPLETE' for seq in ['BT-005', 'SN-013'])
        ready3 = staged['status'] == 'COMPLETE' and photo.get('status') == 'COMPLETE'
        for physical, ready in [('2', ready2), ('3', ready3)]:
            if not ready or physical in active or any(j['status'] == 'FAILED' for j in jobs.values()):
                continue
            if physical == '2':
                for seq in ['BT-005', 'SN-013']:
                    meta = json.loads((work/'temporal_geometry_predictions/dev'/seq/'complete.json').read_text())
                    assert meta['no_memory_reference_all_exact']
            pending = [(key, j) for key, j in jobs.items() if j['physical_gpu'] == physical and j['status'] == 'PENDING']
            if not pending:
                continue
            assert sha256(spec_path) == state['specification_sha256']
            assert all(sha256(ROOT/name) == digest for name, digest in sources.items())
            key, job = pending[0]
            command = [sys.executable, '-u', 'tools/train_correlation_repair.py', '--cue', job['cue'],
                       '--variant', 'full', '--seed', str(job['seed'])]
            log = work/(key+'.replication_fit.log')
            with log.open('ab', buffering=0) as handle:
                proc = subprocess.Popen(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=physical),
                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            active[physical] = (key, proc)
            job.update(status='RUNNING', pid=proc.pid, command=command, log=str(log))
        state['status'] = 'FITTING_MATCHED_SEEDS' if active else 'WAITING_FOR_ALLOWED_GPU_DEPENDENCIES'
        if not active and any(j['status'] == 'FAILED' for j in jobs.values()):
            state['status'] = 'FAILED'
        if all(j['status'] == 'COMPLETE' for j in jobs.values()):
            for seed in [1, 2]:
                histories = [json.loads((work/'correlation_fit'/f'{cue}_full_s{seed}/complete.json').read_text())['history'] for cue in ['rgb', 'gray', 'ir']]
                assert all([r['sample_digest'] for r in h] == [r['sample_digest'] for r in histories[0]] for h in histories)
            state['status'] = 'COMPLETE'
            state['all24_matched_sample_digests_per_seed'] = True
        dump(state_path, state)
        if state['status'] in ['COMPLETE', 'FAILED']:
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
