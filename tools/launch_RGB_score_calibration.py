#!/usr/bin/env python3
"""Train-only scalar calibration, exact identity control, complete Dev family."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from tools.run_pseudo_ir_comparison import dump, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text()); work = Path(plan['work_root'])
    spec_path = ROOT/'configs/perceptual_RGB_score_calibration_20261123.json'
    state_path = work/'RGB_score_calibration_dev_state.json'
    lock = state_path.with_suffix('.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not state_path.exists()
    jobs = {'SC_'+method: dict(status='PENDING', method=method) for method in ['identity', 'temperature', 'platt']}
    sources = {name: sha256(ROOT/name) for name in ['tools/fit_RGB_score_calibration.py', 'tools/replay_RGB_score_calibration.py']}
    state = dict(status='FITTING_TRAIN_ONLY_SCALAR_RGB_CONTROLS', jobs=jobs, specification_sha256=sha256(spec_path), sources=sources, Test_read=False)
    dump(state_path, state)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MFT26IR_ROOT=plan['data_root'])
    fit_log = work/'RGB_score_calibration_fit.log'
    with fit_log.open('ab', buffering=0) as handle:
        proc = subprocess.Popen([sys.executable, '-u', 'tools/fit_RGB_score_calibration.py'], cwd=ROOT, env=env,
            stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(fit_pid=proc.pid, fit_log=str(fit_log)); dump(state_path, state); code = proc.wait()
    if code:
        state.update(status='FIT_FAILED', exit_code=code); dump(state_path, state); return
    model_path = work/'RGB_score_calibration/models.json'; fitted = json.loads(model_path.read_text())
    assert fitted['status'] == 'COMPLETE_BOTH_RGB_TRAIN_CALIBRATIONS_BEFORE_DEV'
    folder = ROOT/'configs/perceptual_RGB_score_calibration_dev_20261123'; folder.mkdir(exist_ok=False)
    for key, job in jobs.items():
        config = dict(id=key, method=job['method'], fitted_maps_sha256=sha256(model_path),
            specification_sha256=sha256(spec_path), registration='Ordinary RGB-only Train-fitted score map. Fixed temperature and Platt procedures, source identity parity first. Original cached candidates and ByteTrack unchanged, no Test.')
        path = folder/(key+'.json'); dump(path, config); job.update(config=str(path), sha256=sha256(path))
    state['status'] = 'RUNNING'; dump(state_path, state)
    for key, job in jobs.items():
        assert sha256(spec_path) == state['specification_sha256']
        assert all(sha256(ROOT/name) == digest for name, digest in sources.items())
        assert sha256(job['config']) == job['sha256']
        output = work/'RGB_score_calibration_dev'/key
        log = work/(key+'.RGB_score_calibration_dev.log')
        command = [sys.executable, '-u', 'tools/replay_RGB_score_calibration.py', '--config', job['config'], '--output', str(output)]
        with log.open('ab', buffering=0) as handle:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            job.update(status='RUNNING', pid=proc.pid, command=command, log=str(log)); dump(state_path, state); code = proc.wait()
        job.update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code)
        if code:
            state['status'] = 'FAILED'; dump(state_path, state); return
        if job['method'] == 'identity':
            reference = work/'correlation_pilot_policies_dev/CGA_original_RGB_ByteTrack'
            current = json.loads((output/'metrics.json').read_text()); original = json.loads((reference/'metrics.json').read_text())
            assert current['trackeval'] == original['trackeval'] and len(current['trackeval']) == 39
            for relative in ['track_results/BT-005.txt', 'track_results/SN-013.txt', 'trackeval_output/pedestrian_detailed.csv']:
                assert sha256(output/relative) == sha256(reference/relative)
            state['full_identity_RGB_parity'] = dict(all39_metrics_both_track_files_and_detailedCSV_exact=True,
                frames=4834, reference_metrics_sha256=sha256(reference/'metrics.json'))
        dump(state_path, state)
    state['status'] = 'COMPLETE'; dump(state_path, state)


if __name__ == '__main__':
    main()
