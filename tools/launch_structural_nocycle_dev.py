#!/usr/bin/env python3
"""Audit 39 streams before the 36 new fixed-seed Dev replays."""
import fcntl
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
from tools.correlation_geometry_checks import compare_packets, fresh_RGB_audit
from tools.run_pseudo_ir_comparison import dump, sha256


def proofs(base, sequence, plan, spec, work, frames=None):
    canonical = Path(plan['frozen_detection_sources']['rgb_dev'])/'detections'/(sequence+'.npz')
    off = 'NC_IR_unavailable_s0_e06'
    result = dict(unavailable_raw=compare_packets(base/(off+'_raw.npz'), base/'CG3_source_RGB.npz', frames),
                  unavailable_priority=compare_packets(base/(off+'.npz'), canonical, frames))
    old = work/'structural_relative_predictions/dev'/sequence
    for suffix in ['', '_raw']:
        result['trained_IR_anchor'+suffix] = compare_packets(base/('NC_IR_anchor_s0_e06'+suffix+'.npz'),
            old/('SR_ir_no_cycle_s0_e06'+suffix+'.npz'), frames)
    count = 0
    for row in spec['models']:
        if row.get('baseline'):
            continue
        with np.load(base/(row['id']+'_raw.npz')) as z:
            assert len(z['boxes']) == len(z['anchor_indices']) == len(z['rgb_reference_scores'])
            assert ((z['anchor_indices'] >= 0) & (z['anchor_indices'] < 23625)).all()
            assert ((z['rgb_reference_scores'] >= 0) & (z['rgb_reference_scores'] <= 1)).all()
        count += 1
    assert count == 19
    result['raw_provenance'] = dict(streams=count, geometry_checked_at_every_selected_dense_RGB_anchor=True)
    return result


def main():
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    spec_path = ROOT/'configs/perceptual_structural_nocycle_eval_20261123.json'
    spec = json.loads(spec_path.read_text())
    state_path = work/'structural_nocycle_dev_state.json'
    lock = state_path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not state_path.exists()
    source_names = ['tools/extract_structural_nocycle.py', 'yolox/models/structural_nocycle_refinement.py',
        'tools/extract_structural_relative.py', 'tools/structural_relative_inference.py',
        'tools/build_correlation_cache.py', 'yolox/models/yolo_head.py', 'tools/indexed_postprocess.py',
        'tools/structural_grid.py', 'tools/structural_correspondence.py',
        'yolox/models/structural_relative_repair.py', 'yolox/models/frozen_head_repair.py',
        'yolox/tracker/foveated_regions.py', 'yolox/tracker/priority_observations.py',
        'tools/replay_structural_relative.py', 'yolox/tracker/byte_tracker.py',
        'tools/correlation_geometry_checks.py', 'tools/launch_structural_nocycle_dev.py']
    sources = {name: sha256(ROOT/name) for name in source_names}
    config_dir = ROOT/'configs/perceptual_structural_nocycle_dev_20261123'
    jobs = {}
    for cfg in sorted(config_dir.glob('*.json')):
        c = json.loads(cfg.read_text())
        assert c['specification_sha256'] == sha256(spec_path) and c['seed'] == 0
        jobs[cfg.stem] = dict(status='PENDING', stage=c['stage'], config=str(cfg), sha256=sha256(cfg))
    assert len(jobs) == 36
    stages = {key: dict(status='PENDING', registered_cases=n) for key, n in spec['stages'].items()}
    state = dict(status='WAITING_FOR_FOUR_SEED0_CONTROL_FITS', controller_pid=os.getpid(), jobs=jobs,
        stages=stages, producers={}, sources=sources, specification_sha256=sha256(spec_path),
        reused_completed_results=17, fixed_seed=0, additional_seed_work=False,
        new_Test_accuracy_used=False, goal_met=False, supported_innovations=0)
    dump(state_path, state)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', MFT26IR_ROOT=plan['data_root'])
    try:
        while True:
            fits = json.loads((work/'structural_nocycle_fit_state.json').read_text())
            if 'FAILED' in fits['status']:
                raise RuntimeError('Upstream fitting failed; retained evidence, no partial accuracy')
            if fits['status'] == 'COMPLETE':
                break
            time.sleep(5)
        assert sha256(spec_path) == state['specification_sha256']
        assert all(sha256(ROOT/name) == digest for name, digest in sources.items())
        assert shutil.disk_usage(work).free >= 40*2**30
        command_base = [sys.executable, '-u', 'tools/extract_structural_nocycle.py']
        state['status'] = 'SOURCE8_AND_TRAINED32_CHECKS'
        dump(state_path, state)
        for seq, physical in [('BT-005', '2'), ('SN-013', '3')]:
            for check in ['source', 'prefix']:
                cmd = command_base+['--sequence', seq]
                cmd += ['--source-check', '8'] if check == 'source' else ['--specification', str(spec_path), '--limit-frames', '32']
                log = work/(seq+'.nocycle_'+check+'.log')
                with log.open('ab', buffering=0) as handle:
                    subprocess.run(cmd, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=physical),
                        stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, check=True)
            base = work/'nocycle_refinement/structural_relative_predictions/dev'/seq
            meta = json.loads((base/'source_check_8/complete.json').read_text())
            assert len(meta['outputs']) == 11 and all(r['array_exact'] for r in meta['fresh_RGB_parity'])
            prefix = base/'prefix_check_32'
            meta = json.loads((prefix/'complete.json').read_text())
            assert len(meta['outputs']) == 39 and meta['frames'] == 32
            state.setdefault('both32_proofs', {})[seq] = proofs(prefix, seq, plan, spec, work, 32)
        active = {}
        state['status'] = 'PRODUCING_ALL39_STREAMS_WITHOUT_GT'
        for seq, physical in [('BT-005', '2'), ('SN-013', '3')]:
            cmd = command_base+['--sequence', seq, '--specification', str(spec_path)]
            log = work/(seq+'.nocycle_full_dev.log')
            with log.open('ab', buffering=0) as handle:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=physical),
                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            active[seq] = proc
            state['producers'][seq] = dict(status='RUNNING', pid=proc.pid, physical_gpu=physical, command=cmd, log=str(log))
        dump(state_path, state)
        while active:
            for seq, proc in list(active.items()):
                code = proc.poll()
                if code is not None:
                    state['producers'][seq].update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code)
                    del active[seq]
            dump(state_path, state)
            time.sleep(5)
        assert all(p['status'] == 'COMPLETE' for p in state['producers'].values())
        assert all(sha256(ROOT/name) == digest for name, digest in sources.items())
        for seq in ['BT-005', 'SN-013']:
            base = work/'nocycle_refinement/structural_relative_predictions/dev'/seq
            meta = json.loads((base/'complete.json').read_text())
            assert meta['specification_sha256'] == sha256(spec_path)
            assert meta['source_hashes'] == {key: sources[key] for key in meta['source_hashes']}
            audit = proofs(base, seq, plan, spec, work)
            audit['prefix39'] = {key: compare_packets(base/'prefix_check_32'/(key+'.npz'), base/(key+'.npz'), 32)
                                 for key in meta['outputs']}
            assert len(audit['prefix39']) == 39
            audit['fresh_RGB'] = fresh_RGB_audit(base, seq, plan, work)
            state.setdefault('both_full_proofs', {})[seq] = audit
        dump(state_path, state)
        for stage in ['matched_controls', 'module_removals', 'IR_interventions']:
            state['status'] = 'RUNNING_'+stage.upper()
            stages[stage]['status'] = 'RUNNING'
            active = {}
            while True:
                for key, proc in list(active.items()):
                    code = proc.poll()
                    if code is not None:
                        jobs[key].update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code)
                        del active[key]
                for key, job in jobs.items():
                    if len(active) >= 2 or any(j['status'] == 'FAILED' for j in jobs.values()):
                        break
                    if job['stage'] != stage or job['status'] != 'PENDING':
                        continue
                    assert sha256(job['config']) == job['sha256']
                    cmd = [sys.executable, '-u', 'tools/replay_structural_relative.py', '--config', job['config'],
                           '--output', str(work/'structural_nocycle_dev'/key)]
                    log = work/(key+'.nocycle_replay.log')
                    with log.open('ab', buffering=0) as handle:
                        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                    active[key] = proc
                    job.update(status='RUNNING', pid=proc.pid, command=cmd, log=str(log))
                dump(state_path, state)
                if not active:
                    assert all(j['status'] == 'COMPLETE' for j in jobs.values() if j['stage'] == stage)
                    break
                time.sleep(5)
            stages[stage]['status'] = 'COMPLETE'
            dump(state_path, state)
        state['status'] = 'COMPLETE'
        dump(state_path, state)
    except Exception as error:
        state.update(status='FAILED', error=repr(error))
        dump(state_path, state)
        raise


if __name__ == '__main__':
    main()
