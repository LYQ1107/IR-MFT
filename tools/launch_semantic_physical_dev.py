#!/usr/bin/env python3
"""Twelve exact source controls then twelve paired semantic/physical replays."""
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
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text()); work = Path(plan['work_root'])
    spec_path = ROOT/'configs/perceptual_semantic_physical_scores_20261123.json'
    assert json.loads((work/'temporal_geometry_dev_state.json').read_text())['status'] == 'COMPLETE'
    assert json.loads((work/'RGB_score_calibration_dev_state.json').read_text())['status'] == 'COMPLETE'
    path = work/'semantic_physical_dev_state.json'; lock = path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); assert not path.exists()
    cases = []
    for cue in ['rgb', 'gray', 'ir']:
        for role in ['none', 'rgb_boxes', 'rgb_scores']:
            source = f'TC_{cue}_{role}_raw'
            cases.append(dict(key=cue+'_'+role, source_kind='TC', source_id=source, parent='temporal_geometry_dev/'+source))
    models_path = work/'RGB_score_calibration/models.json'
    for method in ['identity', 'temperature', 'platt']:
        cases.append(dict(key='RGB_'+method, source_kind='RGB_map', score_map=method,
                          fitted_maps_sha256=sha256(models_path), parent='RGB_score_calibration_dev/SC_'+method))
    folder = ROOT/'configs/perceptual_semantic_physical_dev_20261123'; folder.mkdir(exist_ok=False)
    jobs = {}
    for enabled in [False, True]:
        for case in cases:
            key = 'SP_'+case['key']+('_on' if enabled else '_off')
            config = dict(case, id=key, separate_association_score=enabled, specification_sha256=sha256(spec_path),
                registration='Fixed currentRGB geometry-reference score rule; all9cue/headrole sources and3ordinaryRGB maps, paired off/on. No parameter search, extra pixels, Test or standalone novelty credit.')
            target = folder/(key+'.json'); dump(target, config)
            jobs[key] = dict(status='PENDING', phase='on' if enabled else 'off', config=str(target),
                             sha256=sha256(target), parent=case['parent'])
    assert len(jobs) == 24
    files = ['tools/replay_semantic_physical_scores.py', 'yolox/tracker/semantic_physical_scores.py',
             'yolox/tracker/evidence_byte_tracker.py', 'yolox/tracker/byte_tracker.py', 'yolox/tracker/matching.py']
    state = dict(status='RUNNING_OFF', jobs=jobs, sources={name: sha256(ROOT/name) for name in files},
                 specification_sha256=sha256(spec_path), Test_read=False)
    dump(path, state)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', MFT26IR_ROOT=plan['data_root'])
    for phase in ['off', 'on']:
        state['status'] = 'RUNNING_'+phase.upper(); active = {}
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
                assert all(sha256(ROOT/name) == digest for name, digest in state['sources'].items())
                command = [sys.executable, '-u', 'tools/replay_semantic_physical_scores.py', '--config', job['config'],
                           '--output', str(work/'semantic_physical_dev'/key)]
                log = work/(key+'.semantic_physical_dev.log')
                with log.open('ab', buffering=0) as handle:
                    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                active[key] = proc; job.update(status='RUNNING', pid=proc.pid, command=command, log=str(log))
            dump(path, state)
            if not active:
                if any(j['status'] == 'FAILED' for j in jobs.values()):
                    state['status'] = 'FAILED'; dump(path, state); return
                assert all(j['status'] == 'COMPLETE' for j in jobs.values() if j['phase'] == phase)
                break
            time.sleep(5)
        if phase == 'off':
            proofs = {}
            for key, job in jobs.items():
                if job['phase'] != phase:
                    continue
                current = work/'semantic_physical_dev'/key; parent = work/job['parent']
                a, b = [json.loads((p/'metrics.json').read_text()) for p in [current, parent]]
                assert a['trackeval'] == b['trackeval'] and len(a['trackeval']) == 39
                for relative in ['track_results/BT-005.txt', 'track_results/SN-013.txt', 'trackeval_output/pedestrian_detailed.csv']:
                    assert sha256(current/relative) == sha256(parent/relative)
                proofs[key] = dict(all39_fields_both_tracks_and_detailedCSV_exact=True, frames=4834,
                                   parent_metrics_sha256=sha256(parent/'metrics.json'))
            state['all12_full_source_parities'] = proofs; dump(path, state)
    state['status'] = 'COMPLETE'; dump(path, state)


if __name__ == '__main__':
    main()
