#!/usr/bin/env python3
"""Three exact off controls, then three fixed same-anchor score repairs."""
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
    spec_path = ROOT/'configs/perceptual_exact_reference_scores_20261123.json'; spec = json.loads(spec_path.read_text())
    assert spec['single_seed_user_policy_sha256'] == sha256(ROOT/'configs/perceptual_single_seed_policy_20261123.json')
    path = work/'exact_reference_dev_state.json'; lock = path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); assert not path.exists()
    folder = ROOT/'configs/perceptual_exact_reference_dev_20261123'; folder.mkdir(exist_ok=False)
    jobs = {}
    for enabled in [False, True]:
        for row in spec['cases']:
            key = 'EX_'+row['cue']+('_on' if enabled else '_off')
            config = dict(row, id=key, separate_association_score=enabled, reliable_threshold=.6,
                single_seed_user_policy_sha256=spec['single_seed_user_policy_sha256'],
                producer_specification_sha256=spec['producer_specification_sha256'],
                specification_sha256=sha256(spec_path),
                registration='Fixed same-anchor strongRGBassociation score,3matchedcues/seed0/RGBboxrole only; no newpixels, fits or standalone novelty claim.')
            target = folder/(key+'.json'); dump(target, config)
            jobs[key] = dict(status='PENDING', phase='on' if enabled else 'off', config=str(target),
                sha256=sha256(target), parent=row['parent'])
    names = ['tools/replay_exact_reference_scores.py', 'yolox/tracker/semantic_physical_scores.py',
        'yolox/tracker/evidence_byte_tracker.py', 'yolox/tracker/byte_tracker.py', 'yolox/tracker/matching.py']
    state = dict(status='WAITING_FOR_BOTH_SINGLE_SEED_FULL_SOURCE_PROOFS', jobs=jobs,
        sources={name: sha256(ROOT/name) for name in names}, specification_sha256=sha256(spec_path), Test_read=False)
    dump(path, state)
    while True:
        previous = json.loads((work/'single_seed_geometry_dev_state.json').read_text())
        if previous['status'] in ['FAILED', 'PRODUCER_FAILED']:
            state['status'] = 'DEPENDENCY_FAILED'; dump(path, state); return
        if len(previous.get('full_both_video_proofs', {})) == 2:
            break
        time.sleep(5)
    state['full_source_proofs_state_sha256'] = sha256(work/'single_seed_geometry_dev_state.json')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', MFT26IR_ROOT=plan['data_root'])
    for phase in ['off', 'on']:
        state['status'] = 'RUNNING_'+phase.upper(); dump(path, state)
        for key, job in jobs.items():
            if job['phase'] != phase:
                continue
            assert sha256(job['config']) == job['sha256'] and sha256(spec_path) == state['specification_sha256']
            assert all(sha256(ROOT/name) == value for name, value in state['sources'].items())
            command = [sys.executable, '-u', 'tools/replay_exact_reference_scores.py', '--config', job['config'],
                       '--output', str(work/'exact_reference_dev'/key)]
            log = work/(key+'.exact_reference_dev.log')
            with log.open('ab', buffering=0) as handle:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                job.update(status='RUNNING', pid=proc.pid, command=command, log=str(log)); dump(path, state)
                code = proc.wait()
            job.update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code); dump(path, state)
            if code:
                state['status'] = 'FAILED'; dump(path, state); return
        if phase == 'off':
            state['all3_full_off_parities'] = {}
            for key, job in jobs.items():
                if job['phase'] != 'off':
                    continue
                current = work/'exact_reference_dev'/key; parent = work/job['parent']
                a, b = [json.loads((p/'metrics.json').read_text()) for p in [current, parent]]
                assert len(a['trackeval']) == 39 and a['trackeval'] == b['trackeval']
                for relative in ['track_results/BT-005.txt', 'track_results/SN-013.txt', 'trackeval_output/pedestrian_detailed.csv']:
                    assert sha256(current/relative) == sha256(parent/relative)
                state['all3_full_off_parities'][key] = dict(all39_fields_both_tracks_and_detailedCSV_exact=True,
                    frames=4834, parent_metrics_sha256=sha256(parent/'metrics.json'))
            dump(path, state)
    state['status'] = 'COMPLETE'; dump(path, state)


if __name__ == '__main__':
    main()
