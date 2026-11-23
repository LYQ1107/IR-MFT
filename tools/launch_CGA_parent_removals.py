#!/usr/bin/env python3
"""Original-parent mechanism removals after each card's geometry producer."""
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
    main_plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text()); work = Path(main_plan['work_root'])
    spec_path = ROOT/'configs/perceptual_CGA_parent_removals_20261123.json'; spec = json.loads(spec_path.read_text())
    assert sha256(ROOT/spec['parent_plan']) == spec['parent_sha256']
    path = work/'CGA_parent_removal_fit_state.json'; lock = path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); assert not path.exists()
    sources = {name: sha256(ROOT/name) for name in ['tools/train_correlation_repair.py',
        'yolox/models/local_correlation_repair.py', 'yolox/models/frozen_head_repair.py']}
    jobs = {}
    for physical, variants in spec['schedule'].items():
        for variant in variants:
            out = work/'correlation_fit'/f'ir_{variant}_s0'; assert not out.exists()
            jobs[variant] = dict(status='PENDING', physical_gpu=physical)
    state = dict(status='WAITING_FOR_CARD_GEOMETRY_PRODUCER', jobs=jobs, sources=sources,
                 specification_sha256=sha256(spec_path), Test_read=False)
    dump(path, state); active = {}
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MFT26IR_ROOT=main_plan['data_root'])
    while True:
        for physical, (variant, proc) in list(active.items()):
            code = proc.poll()
            if code is not None:
                jobs[variant].update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code); del active[physical]
                if code == 0:
                    complete = work/'correlation_fit'/f'ir_{variant}_s0/complete.json'
                    fit = json.loads(complete.read_text()); assert fit['epochs'] == 24 and sum(r['steps'] for r in fit['history']) == 6912
                    full = json.loads((work/'correlation_fit/ir_full_s0/complete.json').read_text())
                    assert [r['sample_digest'] for r in fit['history']] == [r['sample_digest'] for r in full['history']]
                    jobs[variant]['complete_sha256'] = sha256(complete)
        geometry = json.loads((work/'correlation_geometry_dev_state.json').read_text())
        for physical, sequence in [('2', 'BT-005'), ('3', 'SN-013')]:
            if physical in active or geometry.get('producers', {}).get(sequence, {}).get('status') != 'COMPLETE':
                continue
            if any(j['status'] == 'FAILED' for j in jobs.values()):
                continue
            pending = [v for v, j in jobs.items() if j['physical_gpu'] == physical and j['status'] == 'PENDING']
            if not pending:
                continue
            assert sha256(spec_path) == state['specification_sha256']
            assert all(sha256(ROOT/name) == digest for name, digest in sources.items())
            variant = pending[0]; command = [sys.executable, '-u', 'tools/train_correlation_repair.py',
                '--cue', 'ir', '--variant', variant, '--seed', '0']
            log = work/(variant+'.CGA_parent_removal_fit.log')
            with log.open('ab', buffering=0) as handle:
                proc = subprocess.Popen(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=physical),
                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            active[physical] = (variant, proc); jobs[variant].update(status='RUNNING', pid=proc.pid, command=command, log=str(log))
        state['status'] = 'FITTING' if active else 'WAITING_FOR_CARD_GEOMETRY_PRODUCER'
        if not active and any(j['status'] == 'FAILED' for j in jobs.values()):
            state['status'] = 'FAILED'
        if all(j['status'] == 'COMPLETE' for j in jobs.values()):
            state['status'] = 'COMPLETE'
        dump(path, state)
        if state['status'] in ['COMPLETE', 'FAILED']:
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
