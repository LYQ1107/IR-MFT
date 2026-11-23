#!/usr/bin/env python3
"""Only launch after complete Dev selection; source checks precede all Test GT."""
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
from tools.correlation_geometry_checks import compare_packets
from tools.recover_correlation_formal_test import aligned_source_audit
from tools.run_pseudo_ir_comparison import dump, sha256


def read_packet(path):
    with np.load(path) as z:
        return {key: z[key] for key in ['frame_ids', 'offsets', 'boxes', 'scores']}


def main():
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    protocol = json.loads((ROOT/plan['protocol']).read_text())
    spec_path = ROOT/'configs/perceptual_structural_nocycle_test_20261123.json'
    spec = json.loads(spec_path.read_text())
    frozen = json.loads((work/'structural_nocycle_test_selection_frozen.json').read_text())
    assert frozen['specification_sha256'] == sha256(spec_path) and frozen['complete_Dev_before_Test']
    assert spec['fixed_seed'] == 0
    path = work/'structural_nocycle_test_state.json'
    lock = path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not path.exists()
    names = ['tools/extract_structural_nocycle_test.py', 'tools/extract_structural_relative.py',
        'tools/structural_relative_inference.py', 'tools/build_correlation_cache.py',
        'tools/indexed_postprocess.py', 'tools/structural_grid.py', 'tools/structural_correspondence.py',
        'yolox/models/yolo_head.py', 'yolox/models/structural_relative_repair.py',
        'yolox/models/structural_nocycle_refinement.py', 'yolox/models/frozen_head_repair.py',
        'yolox/tracker/priority_observations.py', 'yolox/tracker/foveated_regions.py',
        'tools/replay_structural_nocycle_test.py', 'yolox/tracker/byte_tracker.py',
        'tools/recover_correlation_formal_test.py', 'tools/correlation_geometry_checks.py',
        'tools/structural_gate_observer.py',
        'tools/launch_structural_nocycle_test.py']
    sources = {name: sha256(ROOT/name) for name in names}
    jobs = {}
    config_dir = ROOT/'configs/perceptual_structural_nocycle_test_20261123'
    for case in spec['cases']:
        cfg = config_dir/(case['id']+'.json')
        jobs[case['id']] = dict(status='PENDING', config=str(cfg), sha256=sha256(cfg))
    producers = {seq: dict(status='PENDING') for seq in protocol['historical_test']}
    state = dict(status='PRODUCING_FROZEN_TEST_WITHOUT_GT', controller_pid=os.getpid(),
        jobs=jobs, producers=producers, sources=sources, specification_sha256=sha256(spec_path),
        primary=spec['primary_case'], fixed_seed=0, Test_accuracy_read=False,
        goal_met=False, supported_innovations=0)
    dump(path, state)
    try:
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   MFT26IR_ROOT=plan['data_root'])
        frames = {}
        for seq in producers:
            canonical = Path(plan['frozen_detection_sources']['rgb_test'])/'detections'/(seq+'.npz')
            frames[seq] = len(read_packet(canonical)['frame_ids'])
        assert len(frames) == 7 and sum(frames.values()) == 15985
        order = sorted(producers, key=lambda s: frames[s], reverse=True)
        active = {}
        while not all(row['status'] == 'COMPLETE' for row in producers.values()):
            for physical, (seq, phase, proc) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                del active[physical]
                row = producers[seq]
                row[phase+'_exit_code'] = code
                assert code == 0, row['log']
                base = work/'nocycle_formal_test/structural_relative_predictions/test'/seq
                if phase == 'source':
                    meta = json.loads((base/'source_check_8/complete.json').read_text())
                    assert len(meta['outputs']) == 15 and all(x['array_exact'] for x in meta['fresh_RGB_parity'])
                    row['status'] = 'SOURCE_PASSED'
                else:
                    meta = json.loads((base/'complete.json').read_text())
                    assert meta['status'] == 'COMPLETE_STRUCTURAL_RELATIVE_OBSERVATIONS'
                    assert not meta['GT_read'] and not meta['future_frame_input']
                    assert meta['frames'] == frames[seq] and len(meta['outputs']) == spec['counts']['produced_streams']
                    assert meta['specification_sha256'] == sha256(spec_path)
                    assert meta['source_hashes'] == {key: sources[key] for key in meta['source_hashes']}
                    canonical = Path(plan['frozen_detection_sources']['rgb_test'])/'detections'/(seq+'.npz')
                    proof = aligned_source_audit(read_packet(base/'CG3_source_RGB.npz'), read_packet(canonical))
                    off = next(c for c in spec['cases'] if c['id'] == spec['roles']['IR_unavailable'])
                    proof['unavailable_raw_exact_to_fresh'] = compare_packets(base/(off['proposal_key']+'.npz'), base/'CG3_source_RGB.npz')
                    proof['fresh_source8_exact_to_full_prefix'] = compare_packets(base/'source_check_8/CG3_source_RGB.npz', base/'CG3_source_RGB.npz', 8)
                    for model in spec['models']:
                        if model.get('baseline'):
                            continue
                        with np.load(base/(model['id']+'_raw.npz')) as z:
                            assert len(z['boxes']) == len(z['anchor_indices']) == len(z['rgb_reference_scores'])
                            assert ((z['anchor_indices'] >= 0) & (z['anchor_indices'] < 23625)).all()
                    row.update(status='COMPLETE', source_proof=proof, complete_sha256=sha256(base/'complete.json'))
            for physical in ['2', '3']:
                if physical in active:
                    continue
                pending = [(seq, producers[seq]) for seq in order
                    if producers[seq]['status'] == 'SOURCE_PASSED' and producers[seq]['physical_gpu'] == physical]
                pending += [(seq, producers[seq]) for seq in order if producers[seq]['status'] == 'PENDING']
                if not pending:
                    continue
                seq, row = pending[0]
                phase = 'source' if row['status'] == 'PENDING' else 'full'
                assert all(sha256(ROOT/name) == digest for name, digest in sources.items())
                assert sha256(spec_path) == state['specification_sha256']
                assert shutil.disk_usage(work).free >= 40*2**30
                used = 0
                for parent, dirs, files in os.walk(work, followlinks=False):
                    dirs[:] = [d for d in dirs if not Path(parent,d).is_symlink()]
                    used += sum(Path(parent,n).stat().st_size for n in files if not Path(parent,n).is_symlink())
                assert used < 64*2**30
                cmd = [sys.executable, '-u', 'tools/extract_structural_nocycle_test.py', '--sequence', seq, '--split', 'test']
                cmd += ['--source-check', '8'] if phase == 'source' else ['--specification', str(spec_path)]
                log = work/(seq+'.nocycle_test_'+phase+'.log')
                with log.open('ab', buffering=0) as handle:
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=physical),
                        stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                active[physical] = (seq, phase, proc)
                row.update(status=phase.upper()+'_RUNNING', pid=proc.pid, physical_gpu=physical, command=cmd, log=str(log))
            dump(path, state)
            time.sleep(5)
        state['status'] = 'RUNNING_FROZEN_TEST_REPLAYS'
        dump(path, state)
        active = {}
        while not all(j['status'] == 'COMPLETE' for j in jobs.values()):
            for key, proc in list(active.items()):
                code = proc.poll()
                if code is not None:
                    jobs[key].update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code)
                    del active[key]
                    assert code == 0
            for key, job in jobs.items():
                if len(active) >= 2 or job['status'] != 'PENDING':
                    continue
                assert sha256(job['config']) == job['sha256']
                cmd = [sys.executable, '-u', 'tools/replay_structural_nocycle_test.py', '--split', 'test',
                       '--config', job['config'], '--output', str(work/'structural_nocycle_test'/key)]
                log = work/(key+'.nocycle_test_replay.log')
                with log.open('ab', buffering=0) as handle:
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=''),
                        stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                active[key] = proc
                job.update(status='RUNNING', pid=proc.pid, command=cmd, log=str(log))
            dump(path, state)
            time.sleep(5)
        state['status'] = 'COMPLETE'
        dump(path, state)
    except Exception as error:
        state.update(status='FAILED', error=repr(error))
        dump(path, state)
        raise


if __name__ == '__main__':
    main()
