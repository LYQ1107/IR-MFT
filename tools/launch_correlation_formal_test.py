#!/usr/bin/env python3
"""Frozen seed0 Test after structural Dev producers; quarantine Test accuracy."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from tools.correlation_geometry_checks import compare_packets
from tools.run_pseudo_ir_comparison import dump, sha256

ROOT = Path(__file__).resolve().parents[1]


def storage(work):
    used = 0
    for parent, dirs, files in os.walk(work, followlinks=False):
        dirs[:] = [d for d in dirs if not Path(parent, d).is_symlink()]
        used += sum(Path(parent, n).stat().st_size for n in files if not Path(parent, n).is_symlink())
    assert used < 64*2**30 and shutil.disk_usage(work).free >= 40*2**30
    return used/2**30


def source_audit(base, sequence, plan):
    canonical = Path(plan['frozen_detection_sources']['rgb_test'])/'detections'/(sequence+'.npz')
    fresh = base/'CG3_source_RGB.npz'
    with np.load(canonical) as z:
        ref = {k: z[k] for k in ['frame_ids', 'offsets', 'boxes', 'scores']}
    with np.load(fresh) as z:
        current = {k: z[k] for k in ref}
    for key in ['frame_ids', 'offsets']:
        np.testing.assert_array_equal(current[key], ref[key])
    for threshold in [.6, .7]:
        np.testing.assert_array_equal(current['scores'] >= threshold, ref['scores'] >= threshold)
        np.testing.assert_array_equal(current['scores'] > threshold, ref['scores'] > threshold)
    boxes = float(np.max(np.abs(current['boxes']-ref['boxes']), initial=0))
    scores = float(np.max(np.abs(current['scores']-ref['scores']), initial=0))
    assert boxes <= 2 and scores <= .005, 'Unexpected fresh RGB drift; retain evidence before accuracy'
    unavailable = 'CG3_IR_unavailable_rgb_boxes_raw.npz'
    off = compare_packets(base/unavailable, fresh)
    changed = []
    for frame in current['frame_ids']:
        lo, hi = current['offsets'][frame-1:frame+1]
        if not (np.array_equal(current['boxes'][lo:hi], ref['boxes'][lo:hi]) and
                np.array_equal(current['scores'][lo:hi], ref['scores'][lo:hi])):
            changed.append(int(frame))
    return dict(fresh_and_canonical_offsets_and_both_threshold_memberships_exact=True,
                maximum_box_difference_pixels=boxes, maximum_score_difference=scores,
                frames_with_numeric_difference=changed, unavailable_raw_exact_to_fresh=off,
                fresh_sha256=sha256(fresh), canonical_sha256=sha256(canonical),
                full_RGB_bit_exact=not changed)


def main():
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root']); protocol = json.loads((ROOT/plan['protocol']).read_text())
    spec_path = ROOT/'configs/perceptual_CGA_formal_test_20261123.json'
    spec = json.loads(spec_path.read_text()); assert spec['fixed_seed'] == 0
    frozen = json.loads((work/'correlation_selection_frozen.json').read_text())
    assert frozen['specification_sha256'] == sha256(spec_path)
    old = json.loads((work/'single_seed_geometry_dev_state.json').read_text())
    assert old['status'] == 'COMPLETE' and len(old['full_both_video_proofs']) == 2
    path = work/'CGA_formal_test_state.json'; lock = path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); assert not path.exists()
    files = ['tools/extract_single_seed_geometry.py', 'tools/indexed_postprocess.py',
             'tools/build_correlation_cache.py', 'yolox/models/local_correlation_repair.py',
             'tools/replay_correlation_observations.py', 'yolox/tracker/byte_tracker.py',
             'yolox/tracker/priority_observations.py', 'yolox/tracker/foveated_regions.py',
             'tools/launch_correlation_formal_test.py']
    sources = {name: sha256(ROOT/name) for name in files}
    assert sources['tools/extract_single_seed_geometry.py'] == old['sources']['tools/extract_single_seed_geometry.py']
    configs = ROOT/'configs/perceptual_CGA_formal_test_20261123'
    jobs = {r['id']: dict(status='PENDING', config=str(configs/(r['id']+'.json')),
                         sha256=sha256(configs/(r['id']+'.json'))) for r in spec['cases']}
    sequences = {s: dict(status='PENDING') for s in protocol['historical_test']}
    state = dict(status='WAITING_FOR_STRUCTURAL_RELATIVE_GPU_RELEASE', jobs=jobs,
                 producers=sequences, specification_sha256=sha256(spec_path), sources=sources,
                 primary_case=spec['primary_case'], fixed_seed=0, Test_accuracy_read=False,
                 controller_pid=os.getpid(), goal_met=False, supported_innovations=0)
    expected_frames = {}
    for sequence in sequences:
        source = Path(plan['frozen_detection_sources']['rgb_test'])/'detections'/(sequence+'.npz')
        with np.load(source) as z:
            expected_frames[sequence] = len(z['frame_ids'])
    assert sum(expected_frames.values()) == 15985 and len(expected_frames) == 7
    state['expected_frames_from_unlabeled_source_indices'] = expected_frames
    dump(path, state)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               MFT26IR_ROOT=plan['data_root'])
    active = {}; order = sorted(sequences, key=lambda s: expected_frames[s], reverse=True)
    while not all(r['status'] == 'COMPLETE' for r in sequences.values()):
        for physical, (sequence, phase, proc) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            del active[physical]; row = sequences[sequence]; row[phase+'_exit_code'] = code
            assert code == 0, str(row['log'])
            base = work/'single_seed_geometry_predictions/test'/sequence
            if phase == 'source':
                meta = json.loads((base/'source_check_8/complete.json').read_text())
                assert len(meta['outputs']) == 7 and all(x['array_exact'] for x in meta['fresh_RGB_parity'])
                row['status'] = 'SOURCE_PASSED'
            else:
                meta = json.loads((base/'complete.json').read_text())
                assert meta['status'] == 'COMPLETE_CORRELATION_OBSERVATIONS' and not meta['GT_read']
                assert meta['frames'] == expected_frames[sequence]
                assert meta['specification_sha256'] == state['specification_sha256']
                assert meta['source_sha256'] == sources['tools/extract_single_seed_geometry.py']
                row.update(status='COMPLETE', source_audit=source_audit(base, sequence, plan),
                           complete_manifest_sha256=sha256(base/'complete.json'))
        dependency = json.loads((work/'structural_relative_dev_state.json').read_text())
        fits = json.loads((work/'structural_relative_fit_state.json').read_text())
        assert 'FAIL' not in dependency['status'] and 'FAIL' not in fits['status']
        for physical, source_sequence in [('2', 'BT-005'), ('3', 'SN-013')]:
            if physical in active or fits['status'] != 'COMPLETE':
                continue
            if dependency.get('producers', {}).get(source_sequence, {}).get('status') != 'COMPLETE':
                continue
            pending = [(s, sequences[s]) for s in order if sequences[s]['status'] == 'SOURCE_PASSED'
                       and sequences[s]['physical_gpu'] == physical]
            pending += [(s, sequences[s]) for s in order if sequences[s]['status'] == 'PENDING']
            if not pending:
                continue
            sequence, row = pending[0]; phase = 'full' if row['status'] == 'SOURCE_PASSED' else 'source'
            assert sha256(spec_path) == state['specification_sha256']
            assert all(sha256(ROOT/n) == h for n, h in sources.items())
            state['work_physical_GiB_before_latest_launch'] = storage(work)
            command = [sys.executable, '-u', 'tools/extract_single_seed_geometry.py',
                       '--sequence', sequence, '--split', 'test']
            command += ['--source-check', '8'] if phase == 'source' else ['--specification', str(spec_path)]
            log = work/(sequence+'.CGA_formal_test_'+phase+'.log')
            with log.open('ab', buffering=0) as handle:
                proc = subprocess.Popen(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=physical),
                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            active[physical] = (sequence, phase, proc)
            row.update(status=phase.upper()+'_RUNNING', physical_gpu=physical, pid=proc.pid,
                       command=command, log=str(log))
        state['status'] = 'PRODUCING_FROZEN_TEST_INPUTS_WITHOUT_GT' if active else 'WAITING_FOR_STRUCTURAL_RELATIVE_GPU_RELEASE'
        dump(path, state); time.sleep(5)
    state['status'] = 'WAITING_FOR_ALL59_STRUCTURAL_DEV_BEFORE_TEST_ACCURACY'; dump(path, state)
    while True:
        dependency = json.loads((work/'structural_relative_dev_state.json').read_text())
        assert 'FAIL' not in dependency['status']
        if dependency['status'] == 'COMPLETE':
            break
        time.sleep(5)
    state['structural_59Dev_completed_before_Test_accuracy_sha256'] = sha256(work/'structural_relative_dev_state.json')
    active = {}; state['status'] = 'RUNNING_FROZEN_ALL7_TEST_REPLAYS'; dump(path, state)
    while not all(j['status'] == 'COMPLETE' for j in jobs.values()):
        for key, proc in list(active.items()):
            code = proc.poll()
            if code is not None:
                jobs[key].update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code)
                del active[key]; assert code == 0
        for key, row in jobs.items():
            if len(active) >= 2 or row['status'] != 'PENDING':
                continue
            assert sha256(row['config']) == row['sha256']
            command = [sys.executable, '-u', 'tools/replay_correlation_observations.py', '--split', 'test',
                       '--config', row['config'], '--output', str(work/'CGA_formal_test'/key)]
            log = work/(key+'.CGA_formal_test_replay.log')
            with log.open('ab', buffering=0) as handle:
                proc = subprocess.Popen(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=''),
                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            active[key] = proc; row.update(status='RUNNING', pid=proc.pid, command=command, log=str(log))
        dump(path, state); time.sleep(5)
    state['status'] = 'COMPLETE'; dump(path, state)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
        path = Path(plan['work_root'])/'CGA_formal_test_state.json'
        if path.exists():
            state = json.loads(path.read_text())
            if state.get('controller_pid') == os.getpid():
                state.update(status='FAILED', error=repr(error)); dump(path, state)
        raise
