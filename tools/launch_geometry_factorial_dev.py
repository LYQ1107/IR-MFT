#!/usr/bin/env python3
"""Cross NMS/emission geometry, then complete every unchanged parent removal."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tools.correlation_geometry_checks import fresh_RGB_audit
from tools.geometry_factorial_checks import previous_stream_checks, factorial_provenance_checks, all_unavailable_checks
from tools.run_pseudo_ir_comparison import dump, sha256

ROOT = Path(__file__).resolve().parents[1]
ROLES = ['full', 'rgb_boxes', 'rgb_scores', 'fused_nms_rgb_boxes', 'rgb_nms_fused_boxes']


def main():
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text()); work = Path(plan['work_root'])
    spec_path = ROOT/'configs/perceptual_geometry_factorial_eval_20261123.json'; spec = json.loads(spec_path.read_text())
    assert sha256(ROOT/'configs/perceptual_CGA_parent_removals_20261123.json') == spec['parent_fit_plan_sha256']
    old = json.loads((work/'CGA_parent_removal_dev_state.json').read_text())
    assert old['status'].startswith('SUPERSEDED_BEFORE_ANY_PARENT_REMOVAL') and not old['producers']
    assert all(j['status'] == 'PENDING' for j in old['jobs'].values())
    path = work/'geometry_factorial_dev_state.json'; lock = path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); assert not path.exists()
    source = ROOT/'tools/extract_geometry_factorial.py'
    files = [source, ROOT/'tools/indexed_postprocess.py', ROOT/'tools/geometry_factorial_checks.py',
             ROOT/'tools/correlation_geometry_checks.py', Path(__file__)]
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    folder = ROOT/'configs/perceptual_geometry_factorial_dev_20261123'; folder.mkdir(exist_ok=False); jobs = {}
    for row in spec['models']:
        for role in row.get('evaluate_roles', []):
            for policy in ['priority', 'raw']:
                key = row['id'] + ('' if role == 'full' else '_'+role) + ('_raw' if policy == 'raw' else '')
                config = dict(row, id=key, proposal_key=key, head_role=role, output_policy=policy,
                    producer_directory='geometry_factorial_predictions', fusion_plan_sha256=spec['config_sha256'],
                    specification_sha256=sha256(spec_path),
                    registration='Fixed five geometry/score roles; all3cues/3seeds at the prior fixed epochs, original3removals/all4snapshots unchanged. Source66anchor streams exact before any accuracy. No Test or novelty presumption.')
                target = folder/(key+'.json'); dump(target, config)
                jobs[key] = dict(status='PENDING', stage=row['stage'], config=str(target), sha256=sha256(target))
    assert len(jobs) == 164
    stages = {name: dict(status='PENDING', registered_cases=count) for name, count in spec['stages'].items()}
    state = dict(status='WAITING_FOR_ALL3_PARENT_REMOVAL_FITS', jobs=jobs, stages=stages, producers={},
                 sources=sources, specification_sha256=sha256(spec_path), Test_read=False); dump(path, state)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MFT26IR_ROOT=plan['data_root'])
    while True:
        fit = json.loads((work/'CGA_parent_removal_fit_state.json').read_text())
        if fit['status'] == 'FAILED':
            state['status'] = 'FIT_DEPENDENCY_FAILED'; dump(path, state); return
        if fit['status'] == 'COMPLETE':
            break
        time.sleep(5)
    prior = json.loads((work/'correlation_geometry_dev_state.json').read_text())
    assert len(prior.get('full_anchor_and_unavailable_proofs', {})) == 2
    assert all(p['status'] == 'COMPLETE' for p in prior['producers'].values())
    assert sha256(spec_path) == state['specification_sha256'] and all(sha256(ROOT/n) == h for n, h in sources.items())
    used = sum(p.lstat().st_size for p in work.rglob('*') if not p.is_symlink() and p.is_file())
    if used > 64*2**30 or shutil.disk_usage(work).free < 40*2**30:
        raise RuntimeError('User work cap/reserve exceeded')
    command_base = [sys.executable, '-u', str(source)]
    state['status'] = 'SOURCE_AND66_STREAM_PREFIX_CHECKS'; dump(path, state); proofs = {}
    for sequence, physical in [('BT-005', '2'), ('SN-013', '3')]:
        for check in ['source', 'prefix']:
            command = command_base+['--sequence', sequence]
            command += ['--source-check', '8'] if check == 'source' else ['--specification', str(spec_path), '--limit-frames', '32', '--head-roles', *ROLES]
            log = work/(sequence+'.factorial_'+check+'_check.log')
            with log.open('ab', buffering=0) as handle:
                code = subprocess.call(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=physical),
                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            assert code == 0, str(log)
        base = work/'geometry_factorial_predictions/dev'/sequence
        meta = json.loads((base/'source_check_8/complete.json').read_text())
        assert len(meta['outputs']) == 7 and all(r['array_exact'] for r in meta['fresh_RGB_parity'])
        prefix = base/'prefix_check_32'; meta = json.loads((prefix/'complete.json').read_text())
        assert len(meta['outputs']) == 231 and meta['source_sha256'] == sources[str(source.relative_to(ROOT))]
        proofs[sequence] = dict(previous66=previous_stream_checks(prefix, sequence, spec, work, 32),
            unavailable10=all_unavailable_checks(prefix, sequence, plan, work, 32), provenance=factorial_provenance_checks(prefix, spec))
    state['both32_prefix_proofs'] = proofs; state['status'] = 'PRODUCING_ALL231_STREAMS'; active = {}
    for sequence, physical in [('BT-005', '2'), ('SN-013', '3')]:
        command = command_base+['--sequence', sequence, '--specification', str(spec_path), '--head-roles', *ROLES]
        log = work/(sequence+'.geometry_factorial_full_dev.log')
        with log.open('ab', buffering=0) as handle:
            proc = subprocess.Popen(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=physical),
                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        active[sequence] = proc; state['producers'][sequence] = dict(status='RUNNING', pid=proc.pid, physical_gpu=physical, command=command, log=str(log))
    dump(path, state)
    while active:
        for sequence, proc in list(active.items()):
            code = proc.poll()
            if code is not None:
                state['producers'][sequence].update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code); del active[sequence]
        dump(path, state); time.sleep(5)
    if any(p['status'] != 'COMPLETE' for p in state['producers'].values()):
        state['status'] = 'PRODUCER_FAILED'; dump(path, state); return
    state['full_both_video_proofs'] = {}
    for sequence in ['BT-005', 'SN-013']:
        base = work/'geometry_factorial_predictions/dev'/sequence; meta = json.loads((base/'complete.json').read_text())
        assert meta['source_sha256'] == sources[str(source.relative_to(ROOT))] and meta['specification_sha256'] == sha256(spec_path)
        state['full_both_video_proofs'][sequence] = dict(previous66=previous_stream_checks(base, sequence, spec, work),
            unavailable10=all_unavailable_checks(base, sequence, plan, work), provenance=factorial_provenance_checks(base, spec),
            freshRGB=fresh_RGB_audit(base, sequence, plan, work))
    for stage in ['main_factorial', 'parent_removals']:
        state['status'] = 'RUNNING_'+stage.upper(); stages[stage]['status'] = 'RUNNING'; active = {}; dump(path, state)
        while True:
            for key, proc in list(active.items()):
                code = proc.poll()
                if code is not None:
                    jobs[key].update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code); del active[key]
            for key, job in jobs.items():
                if len(active) >= 2 or any(j['status'] == 'FAILED' for j in jobs.values()):
                    break
                if job['stage'] != stage or job['status'] != 'PENDING':
                    continue
                assert sha256(job['config']) == job['sha256']
                command = [sys.executable, '-u', 'tools/replay_correlation_observations.py', '--config', job['config'], '--output', str(work/'geometry_factorial_dev'/key)]
                log = work/(key+'.geometry_factorial_replay.log')
                with log.open('ab', buffering=0) as handle:
                    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                active[key] = proc; job.update(status='RUNNING', pid=proc.pid, command=command, log=str(log))
            dump(path, state)
            if not active:
                if any(j['status'] == 'FAILED' for j in jobs.values()):
                    state['status'] = 'FAILED'; stages[stage]['status'] = 'FAILED'; dump(path, state); return
                assert all(j['status'] == 'COMPLETE' for j in jobs.values() if j['stage'] == stage)
                break
            time.sleep(5)
        stages[stage]['status'] = 'COMPLETE'; dump(path, state)
    state['status'] = 'COMPLETE'; dump(path, state)


if __name__ == '__main__':
    main()
