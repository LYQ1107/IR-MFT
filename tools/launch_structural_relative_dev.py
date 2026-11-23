#!/usr/bin/env python3
"""Single-seed full controls, module removals and actual-IR interventions."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tools.correlation_geometry_checks import compare_packets, fresh_RGB_audit
from tools.run_pseudo_ir_comparison import dump, sha256

ROOT = Path(__file__).resolve().parents[1]


def unavailable_proof(base, sequence, plan, frames=None):
    key = 'SR_IR_unavailable_s0_e24'
    canonical = Path(plan['frozen_detection_sources']['rgb_dev'])/'detections'/(sequence+'.npz')
    return dict(raw=compare_packets(base/(key+'_raw.npz'), base/'CG3_source_RGB.npz', frames),
                priority=compare_packets(base/(key+'.npz'), canonical, frames))


def raw_provenance(base, spec):
    checked = 0
    for row in spec['models']:
        if row.get('baseline'):
            continue
        with __import__('numpy').load(base/(row['id']+'_raw.npz')) as packet:
            assert len(packet['boxes']) == len(packet['anchor_indices']) == len(packet['rgb_reference_scores'])
            assert ((packet['anchor_indices'] >= 0) & (packet['anchor_indices'] < 23625)).all()
            assert ((packet['rgb_reference_scores'] >= 0) & (packet['rgb_reference_scores'] <= 1)).all()
        checked += 1
    assert checked == 29
    return dict(raw_streams=checked, dense_RGB_geometry_asserted_at_every_selected_anchor_in_producer=True)


def main():
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text()); work = Path(plan['work_root'])
    spec_path = ROOT/'configs/perceptual_structural_relative_eval_20261123.json'; spec = json.loads(spec_path.read_text())
    config_path = ROOT/'configs/perceptual_structural_relative_20261123.json'
    policy_path = ROOT/'configs/perceptual_single_seed_policy_20261123.json'
    assert spec['config_sha256'] == sha256(config_path) and spec['single_seed_user_policy_sha256'] == sha256(policy_path)
    policy = json.loads(policy_path.read_text()); assert policy['fixed_seed'] == 0 and not policy['additional_seed_inference']
    path = work/'structural_relative_dev_state.json'; lock = path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); assert not path.exists()
    names = ['tools/extract_structural_relative.py', 'tools/structural_relative_inference.py',
        'tools/build_correlation_cache.py', 'yolox/models/yolo_head.py',
        'tools/indexed_postprocess.py', 'tools/structural_grid.py', 'tools/structural_correspondence.py',
        'yolox/models/structural_relative_repair.py', 'yolox/models/frozen_head_repair.py',
        'yolox/tracker/foveated_regions.py', 'yolox/tracker/priority_observations.py',
        'tools/replay_structural_relative.py', 'yolox/tracker/byte_tracker.py', 'tools/correlation_geometry_checks.py',
        'tools/launch_structural_relative_dev.py']
    sources = {name: sha256(ROOT/name) for name in names}
    config_dir = ROOT/'configs/perceptual_structural_relative_dev_20261123'; config_dir.mkdir(exist_ok=False)
    jobs = {}
    for row in spec['models']:
        for policy_name in ['priority', 'raw'] if not row.get('baseline') else ['baseline']:
            key = row['id']+('_raw' if policy_name == 'raw' else '')
            config = dict(row, id=key, model_id=row['id'], proposal_key=key,
                output_policy=policy_name, producer_directory='structural_relative_predictions',
                fusion_plan_sha256=spec['config_sha256'], specification_sha256=sha256(spec_path),
                single_seed_user_policy_sha256=sha256(policy_path),
                registration='Onlyseed0. All4snapshots/bothpolicies for all6fits, registered before fitting. '
                             'RGBgeometry fixed; fullsource8 and32prefix/full59parities beforeaccuracy. NoTest/noveltypresumption.')
            target = config_dir/(key+'.json'); dump(target, config)
            jobs[key] = dict(status='PENDING', stage=row['stage'], config=str(target), sha256=sha256(target))
    assert len(jobs) == 59
    stages = {key: dict(status='PENDING', registered_cases=value) for key, value in spec['stages'].items()}
    state = dict(status='WAITING_FOR_ALL6_SINGLE_SEED_FITS', jobs=jobs, stages=stages, producers={},
        sources=sources, specification_sha256=sha256(spec_path), Test_read=False, goal_met=False, supported_innovations=0)
    dump(path, state)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MFT26IR_ROOT=plan['data_root'])
    while True:
        fit = json.loads((work/'structural_relative_fit_state.json').read_text())
        if fit['status'] == 'FAILED':
            state['status'] = 'FIT_DEPENDENCY_FAILED'; dump(path, state); return
        if fit['status'] == 'COMPLETE':
            break
        time.sleep(5)
    assert sha256(spec_path) == state['specification_sha256'] and all(sha256(ROOT/n) == h for n, h in sources.items())
    if shutil.disk_usage(work).free < 40*2**30:
        raise RuntimeError('Keep independent-disk reserve')
    command_base = [sys.executable, '-u', 'tools/extract_structural_relative.py']
    state['status'] = 'SOURCE8_AND_TRAINED32_PREFIX_CHECKS'; dump(path, state)
    for sequence, physical in [('BT-005', '2'), ('SN-013', '3')]:
        for check in ['source', 'prefix']:
            command = command_base+['--sequence', sequence]
            command += ['--source-check', '8'] if check == 'source' else ['--specification', str(spec_path), '--limit-frames', '32']
            log = work/(sequence+'.structural_relative_'+check+'_check.log')
            with log.open('ab', buffering=0) as handle:
                code = subprocess.call(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=physical),
                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            assert code == 0, str(log)
        base = work/'structural_relative_predictions/dev'/sequence
        meta = json.loads((base/'source_check_8/complete.json').read_text())
        assert len(meta['outputs']) == 7 and all(r['array_exact'] for r in meta['fresh_RGB_parity'])
        prefix = base/'prefix_check_32'; meta = json.loads((prefix/'complete.json').read_text())
        assert len(meta['outputs']) == 59 and meta['frames'] == 32
        state.setdefault('both32_prefix_proofs', {})[sequence] = dict(
            unavailable=unavailable_proof(prefix, sequence, plan, 32), provenance=raw_provenance(prefix, spec))
    active = {}; state['status'] = 'PRODUCING_ALL59_STRUCTURAL_STREAMS'
    for sequence, physical in [('BT-005', '2'), ('SN-013', '3')]:
        command = command_base+['--sequence', sequence, '--specification', str(spec_path)]
        log = work/(sequence+'.structural_relative_full_dev.log')
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
    if any(row['status'] != 'COMPLETE' for row in state['producers'].values()):
        state['status'] = 'PRODUCER_FAILED'; dump(path, state); return
    for sequence in ['BT-005', 'SN-013']:
        base = work/'structural_relative_predictions/dev'/sequence; meta = json.loads((base/'complete.json').read_text())
        assert meta['specification_sha256'] == sha256(spec_path) and meta['source_hashes'] == {key: sources[key] for key in meta['source_hashes']}
        prefix = base/'prefix_check_32'
        proofs = {key: compare_packets(prefix/(key+'.npz'), base/(key+'.npz'), 32) for key in meta['outputs']}
        assert len(proofs) == 59
        state.setdefault('both_full_source_proofs', {})[sequence] = dict(prefix59=proofs,
            unavailable=unavailable_proof(base, sequence, plan), provenance=raw_provenance(base, spec),
            freshRGB=fresh_RGB_audit(base, sequence, plan, work))
    dump(path, state)
    for stage in ['full_controls', 'module_removals', 'IR_interventions']:
        state['status'] = 'RUNNING_'+stage.upper(); stages[stage]['status'] = 'RUNNING'; active = {}
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
                command = [sys.executable, '-u', 'tools/replay_structural_relative.py', '--config', job['config'], '--output', str(work/'structural_relative_dev'/key)]
                log = work/(key+'.structural_relative_replay.log')
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
