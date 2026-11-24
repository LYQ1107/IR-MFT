#!/usr/bin/env python3
"""Wait for existing GPU work to finish, then run a two-step input parity probe."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.probe_structural_stream_optimizer_gpu import dependency_ready


def main():
    plan = json.loads((ROOT / 'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    path = work / 'structural_stream_optimizer_GPU_schedule.json'
    lock = path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not path.exists(), 'Preserve previous schedule/failure'
    names = ['tools/schedule_structural_stream_optimizer_gpu.py',
             'tools/probe_structural_stream_optimizer_gpu.py', 'tools/structural_stream_loop.py',
             'tools/structural_stream_training.py', 'tools/structural_train_image_reader.py']
    state = dict(status='WAITING_FOR_ALL_FACTORIAL_TEST_NN_PRODUCERS', controller_pid=os.getpid(),
                 source_hashes={name: sha256(ROOT / name) for name in names},
                 physical_gpu='4', diagnostic_updates_per_path=2, no_formal_fit=True,
                 no_additional_seed=True, goal_met=False, supported_innovations=0)
    dump(path, state)
    try:
        while True:
            ready, dependencies = dependency_ready(work)
            assert all(sha256(ROOT / name) == digest for name, digest in state['source_hashes'].items())
            state['dependencies'] = dependencies
            dump(path, state)
            if ready:
                break
            time.sleep(5)
        log = work / 'probe_structural_stream_optimizer_GPU.log'
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='4', OMP_NUM_THREADS='2',
                           MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with log.open('xb', buffering=0) as output:
            proc = subprocess.Popen([sys.executable, '-u', 'tools/probe_structural_stream_optimizer_gpu.py'],
                cwd=ROOT, env=environment, stdin=subprocess.DEVNULL, stdout=output,
                stderr=subprocess.STDOUT, start_new_session=True)
        state.update(status='RUNNING_TWO_STEP_LIVE_GPU_PARITY', probe_pid=proc.pid, log=str(log))
        dump(path, state)
        assert proc.wait() == 0
        proof = json.loads((work / 'structural_stream_optimizer_GPU/state.json').read_text())
        assert proof['status'] == 'COMPLETE_TWO_STEP_LIVE_GPU_OPTIMIZER_PARITY'
        state.update(status='COMPLETE_TWO_STEP_LIVE_GPU_PARITY_NO_FORMAL_FIT',
                     proof_sha256=sha256(work / 'structural_stream_optimizer_GPU/state.json'))
        dump(path, state)
    except Exception as error:
        state.update(status='FAILED_PRESERVE_GPU_PARITY_SCHEDULE', error=repr(error))
        dump(path, state)
        raise


if __name__ == '__main__':
    main()
