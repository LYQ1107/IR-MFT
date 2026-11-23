#!/usr/bin/env python3
"""Balanced fixed-epoch geometry replication and frozen feature interventions."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tools.correlation_geometry_checks import trained_anchor_checks, unavailable_checks, fresh_RGB_audit
from tools.run_pseudo_ir_comparison import dump, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text()); work = Path(plan['work_root'])
    spec_path = ROOT/'configs/perceptual_correlation_geometry_eval_20261123.json'; spec = json.loads(spec_path.read_text())
    assert sha256(ROOT/'configs/perceptual_correlation_replication_20261123.json') == spec['replication_fit_plan_sha256']
    old = json.loads((work/'correlation_replication_dev_state.json').read_text())
    assert old['status'].startswith('SUPERSEDED_BEFORE_ANY_REPLICATION') and not old['producers']
    assert all(j['status'] == 'PENDING' for j in old['jobs'].values())
    path = work/'correlation_geometry_dev_state.json'
    lock = path.with_suffix('.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not path.exists()
    source = ROOT/'tools/extract_correlation_policies.py'
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in [source, ROOT/'tools/correlation_geometry_checks.py']}
    folder = ROOT/'configs/perceptual_correlation_geometry_dev_20261123'; folder.mkdir(exist_ok=False)
    jobs = {}
    for row in spec['models']:
        if row.get('anchor'):
            continue
        for role in row['head_roles']:
            for policy in ['priority', 'raw']:
                key = row['id'] + ('' if role == 'full' else '_'+role) + ('_raw' if policy == 'raw' else '')
                config = dict(row, id=key, proposal_key=key, head_role=role, output_policy=policy,
                    producer_directory='correlation_geometry_predictions', fusion_plan_sha256=spec['config_sha256'],
                    specification_sha256=sha256(spec_path),
                    registration='All original48 new-seed full-fusion cases,24 fixed-epoch geometry cases,12 fixed seed0 feature interventions. Same previously frozen epoch across seeds for extra roles. No Test or established contribution.')
                target = folder/(key+'.json'); dump(target, config)
                jobs[key] = dict(status='PENDING', config=str(target), sha256=sha256(target))
    assert len(jobs) == 84
    state = dict(status='WAITING_FOR_ALL6_MATCHED_SEED_FITS', jobs=jobs, producers={}, sources=sources,
                 specification_sha256=sha256(spec_path), Test_read=False)
    dump(path, state)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', MFT26IR_ROOT=plan['data_root'])
    while True:
        fit = json.loads((work/'correlation_replication_fit_state.json').read_text())
        if fit['status'] == 'FAILED':
            state['status'] = 'FIT_DEPENDENCY_FAILED'; dump(path, state); return
        if fit['status'] == 'COMPLETE':
            break
        time.sleep(5)
    assert sha256(spec_path) == state['specification_sha256']
    assert all(sha256(ROOT/name) == digest for name, digest in sources.items())
    used = sum(p.lstat().st_size for p in work.rglob('*') if not p.is_symlink() and p.is_file())
    if used > 64*2**30 or shutil.disk_usage(work).free < 40*2**30:
        raise RuntimeError('User work-cache cap or independent-disk reserve exceeded')
    command_base = [sys.executable, '-u', str(source), '--producer-name', 'correlation_geometry_predictions']
    state['status'] = 'SOURCE_AND_TRAINED_PREFIX_CHECKS'; dump(path, state)
    prefix_proofs = {}
    for sequence, physical in [('BT-005', '2'), ('SN-013', '3')]:
        for check in ['source', 'prefix']:
            command = command_base + ['--sequence', sequence]
            if check == 'source':
                command += ['--source-check', '8']
            else:
                command += ['--specification', str(spec_path), '--limit-frames', '32', '--head-roles', 'full', 'rgb_boxes', 'rgb_scores']
            log = work/(sequence+'.geometry_'+check+'_check.log')
            with log.open('ab', buffering=0) as handle:
                code = subprocess.call(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=physical),
                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            assert code == 0, str(log)
        base = work/'correlation_geometry_predictions/dev'/sequence
        check = json.loads((base/'source_check_8/complete.json').read_text())
        assert check['source_sha256'] == sources[str(source.relative_to(ROOT))] and len(check['outputs']) == 7
        assert all(r['array_exact'] for r in check['fresh_RGB_parity'])
        prefix = base/'prefix_check_32'; meta = json.loads((prefix/'complete.json').read_text())
        assert meta['frames'] == 32 and len(meta['outputs']) == 103 and meta['specification_sha256'] == sha256(spec_path)
        prefix_proofs[sequence] = dict(trained18=trained_anchor_checks(prefix, sequence, spec, work, 32),
                                      unavailable6=unavailable_checks(prefix, sequence, plan, work, 32))
    state['both32_frame_trained_and_unavailable_proofs'] = prefix_proofs
    state['status'] = 'PRODUCING_ALL103_DEV_STREAMS'; active = {}
    for sequence, physical in [('BT-005', '2'), ('SN-013', '3')]:
        command = command_base + ['--sequence', sequence, '--specification', str(spec_path), '--head-roles', 'full', 'rgb_boxes', 'rgb_scores']
        log = work/(sequence+'.correlation_geometry_full_dev.log')
        with log.open('ab', buffering=0) as handle:
            proc = subprocess.Popen(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=physical),
                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        active[sequence] = proc
        state['producers'][sequence] = dict(status='RUNNING', pid=proc.pid, physical_gpu=physical, command=command, log=str(log))
    dump(path, state)
    while active:
        for key, proc in list(active.items()):
            code = proc.poll()
            if code is not None:
                state['producers'][key].update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code); del active[key]
        dump(path, state); time.sleep(5)
    if any(p['status'] != 'COMPLETE' for p in state['producers'].values()):
        state['status'] = 'PRODUCER_FAILED'; dump(path, state); return
    state['full_anchor_and_unavailable_proofs'] = {}
    for sequence in ['BT-005', 'SN-013']:
        base = work/'correlation_geometry_predictions/dev'/sequence
        meta = json.loads((base/'complete.json').read_text())
        assert meta['source_sha256'] == sources[str(source.relative_to(ROOT))] and meta['specification_sha256'] == sha256(spec_path)
        state['full_anchor_and_unavailable_proofs'][sequence] = dict(trained18=trained_anchor_checks(base, sequence, spec, work),
            unavailable6=unavailable_checks(base, sequence, plan, work), fresh_RGB=fresh_RGB_audit(base, sequence, plan, work))
    state['status'] = 'RUNNING'; active = {}; dump(path, state)
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
                       '--output', str(work/'correlation_geometry_dev'/key)]
            log = work/(key+'.correlation_geometry_replay.log')
            with log.open('ab', buffering=0) as handle:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            active[key] = proc; job.update(status='RUNNING', pid=proc.pid, command=command, log=str(log))
        if not active:
            if all(j['status'] == 'COMPLETE' for j in jobs.values()):
                state['status'] = 'COMPLETE'
            elif any(j['status'] == 'FAILED' for j in jobs.values()):
                state['status'] = 'FAILED'
        dump(path, state)
        if state['status'] != 'RUNNING':
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
