#!/usr/bin/env python3
"""Six matched all-Train fits, after the current pipeline and live-input proof."""
import fcntl
import hashlib
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
from tools.run_pseudo_ir_comparison import dump, sha256


def draw_digests(sequences, table, limit=None):
    result = []
    for macro in table:
        digest = hashlib.sha256()
        for row in macro[:limit]:
            draw = [(sequences[int(index)], int(frame)) for index, frame in row]
            digest.update(json.dumps(draw, separators=(',', ':')).encode())
        result.append(digest.hexdigest())
    return result


def main():
    plan = json.loads((ROOT / 'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    study_path = ROOT / 'configs/perceptual_structural_fulltrain_study_20261123.json'
    study = json.loads(study_path.read_text())
    path = work / 'structural_full_train_fit_state.json'
    lock = path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not path.exists()
    schedule = json.loads((work / 'structural_full_train_schedule/complete.json').read_text())
    assert sha256(work / 'structural_full_train_schedule/complete.json') == study['schedule_proof_sha256']
    assert sha256(schedule['address_file']['path']) == schedule['address_file']['sha256']
    with np.load(schedule['address_file']['path']) as z:
        table = z['addresses']
    expected = draw_digests(schedule['sequence_order'], table)
    expected_smoke = draw_digests(schedule['sequence_order'], table[:1], limit=20)[0]
    jobs = {f"{r['cue']}_{r['components']}_s0": dict(r, status='PENDING') for r in study['fits']}
    assert len(jobs) == 6 and all(j['physical_gpu'] in ['4', '5'] and j['seed'] == 0 for j in jobs.values())
    names = ['tools/launch_structural_full_train_fits.py', 'tools/train_structural_full_train.py',
        'tools/structural_full_train_schedule.py', 'tools/structural_full_train_inputs.py',
        'tools/structural_stream_loop.py', 'tools/structural_stream_training.py',
        'tools/structural_train_image_reader.py', 'tools/train_structural_relative.py',
        'yolox/models/structural_factorial_repair.py', 'yolox/models/structural_relative_repair.py',
        'yolox/models/structural_relative_losses.py', 'yolox/models/frozen_head_repair.py',
        'yolox/models/recoverable_curriculum.py']
    state = dict(status='WAITING_FOR_DOWNSTREAM_REGISTRATION_AND_LIVE_GPU_PROOF', controller_pid=os.getpid(),
        jobs=jobs, source_hashes={name: sha256(ROOT / name) for name in names},
        full_train_study_sha256=sha256(study_path), fixed_seed=0, additional_seed_work=False,
        logical_Train_epochs_each=3, updates_each=20736, goal_met=False, supported_innovations=0)
    dump(path, state)
    active = {}
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       MFT26IR_ROOT=plan['data_root'])
    try:
        while True:
            ready_path = work / 'structural_full_train_pipeline_registered.json'
            pipeline = json.loads(ready_path.read_text()) if ready_path.exists() else {}
            diagnostic_path = work / 'structural_stream_optimizer_GPU_schedule.json'
            diagnostic = json.loads(diagnostic_path.read_text())
            assert 'FAIL' not in diagnostic['status'], 'Preserve failed live-input proof'
            if pipeline:
                assert pipeline['study_sha256'] == state['full_train_study_sha256']
                assert pipeline['status'] == 'COMPLETE_REGISTERED_DEV_TEST_PIPELINE'
                assert all(sha256(ROOT / name) == digest for name, digest in pipeline['source_hashes'].items())
            ready = bool(pipeline) and diagnostic['status'] == 'COMPLETE_TWO_STEP_LIVE_GPU_PARITY_NO_FORMAL_FIT'
            assert sha256(study_path) == state['full_train_study_sha256']
            assert all(sha256(ROOT / name) == digest for name, digest in state['source_hashes'].items())
            for physical, (key, phase, proc) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                del active[physical]
                job = jobs[key]
                job[phase + '_exit_code'] = code
                assert code == 0, job['log']
                folder = work / 'structural_full_train_fit' / key
                if phase == 'smoke':
                    folder = folder / 'smoke_20'
                proof_path = folder / 'full_train_complete.json'
                proof = json.loads(proof_path.read_text())
                assert proof['initial_raw_RGB_head_bit_exact'] and proof['post_loop_source_verified']
                assert proof['full_encoder_frozen_no_grad'] and proof['trainable_parameters'] == 145600
                assert proof['full_train_study_sha256'] == state['full_train_study_sha256']
                assert proof['full_train_address_sha256'] == study['semantic_address_sha256']
                if phase == 'smoke':
                    assert proof['status'] == 'FULL_TRAIN_SMOKE20_VERIFIED'
                    assert proof['successful_steps_this_invocation'] == 20
                    assert proof['history'][0]['sample_digest'] == expected_smoke
                    job['status'] = 'SMOKE_PASSED'
                else:
                    assert proof['status'] == 'COMPLETE_THREE_FULL_TRAIN_TRAVERSALS'
                    assert proof['logical_Train_epochs_completed'] == 3
                    assert proof['successful_steps_this_invocation'] == 20736
                    assert [r['sample_digest'] for r in proof['history']] == expected
                    assert [r['actual_frame_draw_sha256'] for r in proof['live_frame_passes']] == expected
                    job['status'] = 'COMPLETE'
                job[phase + '_complete_sha256'] = sha256(proof_path)
            for physical in ['4', '5']:
                if physical in active or not ready:
                    continue
                pending = [(key, job) for key, job in jobs.items()
                           if job['physical_gpu'] == physical and job['status'] in ['PENDING', 'SMOKE_PASSED']]
                if not pending:
                    continue
                key, job = pending[0]
                phase = 'smoke' if job['status'] == 'PENDING' else 'fit'
                assert shutil.disk_usage(work).free >= 40 * 2**30
                used = int(subprocess.check_output(['du', '-s', '-B1', str(work)], text=True).split()[0])
                assert used < 64 * 2**30
                command = [sys.executable, '-u', 'tools/train_structural_full_train.py', '--cue', job['cue'],
                           '--components', job['components'], '--seed', '0']
                if phase == 'smoke':
                    command += ['--smoke-steps', '20']
                log = work / (key + '.structural_full_train_' + phase + '.log')
                with log.open('xb', buffering=0) as output:
                    proc = subprocess.Popen(command, cwd=ROOT, env=dict(environment, CUDA_VISIBLE_DEVICES=physical),
                        stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                active[physical] = (key, phase, proc)
                job.update(status=phase.upper() + '_RUNNING', pid=proc.pid, command=command, log=str(log))
            state['status'] = ('FITTING_SIX_MATCHED_FULL_TRAIN_CONFIGURATIONS' if active else
                              'WAITING_FOR_DOWNSTREAM_REGISTRATION_AND_LIVE_GPU_PROOF')
            if all(j['status'] == 'COMPLETE' for j in jobs.values()):
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
    main()
