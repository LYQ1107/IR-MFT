#!/usr/bin/env python3
"""Retain a failed source audit and resume unchanged frozen Test cases, before GT."""
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
from scipy.optimize import linear_sum_assignment
from tools.correlation_geometry_checks import compare_packets
from tools.run_pseudo_ir_comparison import dump, sha256


def aligned_source_audit(current, reference, tail_frames=4):
    """Bound matched numerical drift and explicitly expose weak-NMS topology changes."""
    np.testing.assert_array_equal(current['frame_ids'], reference['frame_ids'])
    differences = []
    last = int(current['frame_ids'][-1])
    for frame in current['frame_ids']:
        lo, hi = current['offsets'][frame-1:frame+1]
        a = np.c_[current['boxes'][lo:hi], current['scores'][lo:hi]]
        lo, hi = reference['offsets'][frame-1:frame+1]
        b = np.c_[reference['boxes'][lo:hi], reference['scores'][lo:hi]]
        if np.array_equal(a, b):
            continue
        assert frame > last-tail_frames, 'Unexpected change outside final batch'
        i, j = linear_sum_assignment(abs(a[:, None, :4]-b[None, :, :4]).sum(-1))
        box = float(np.max(abs(a[i, :4]-b[j, :4]), initial=0))
        score = float(np.max(abs(a[i, 4]-b[j, 4]), initial=0))
        assert box <= 2 and score <= .005, 'Exceeds original predeclared numerical bounds'
        for threshold in [.6, .7]:
            np.testing.assert_array_equal(a[i, 4] > threshold, b[j, 4] > threshold)
            np.testing.assert_array_equal(a[i, 4] >= threshold, b[j, 4] >= threshold)
        unmatched_a, unmatched_b = np.delete(a, i, 0), np.delete(b, j, 0)
        assert len(unmatched_a)+len(unmatched_b) <= 1
        assert (unmatched_a[:, 4] < .6).all() and (unmatched_b[:, 4] < .6).all()
        differences.append(dict(frame=int(frame), current_detections=len(a), canonical_detections=len(b),
            matched_maximum_box_difference_pixels=box, matched_maximum_score_difference=score,
            unmatched_current=unmatched_a.tolist(), unmatched_canonical=unmatched_b.tolist(),
            matched_strict_and_nonstrict_06_07_memberships_exact=True))
    return dict(full_RGB_bit_exact=not differences, changed_tail_frames=differences,
        counts_offsets_exact=bool(np.array_equal(current['offsets'], reference['offsets'])),
        interpretation='Weak NMS differences can affect ByteTrack; fresh and canonical RGB both remain mandatory controls. '
                       'This audit does not assert byte-exact detector reproduction or tracking equivalence.')


def audit(base, sequence, plan):
    canonical = Path(plan['frozen_detection_sources']['rgb_test'])/'detections'/(sequence+'.npz')
    fresh = base/'CG3_source_RGB.npz'
    def read(path):
        with np.load(path) as z:
            return {key: z[key] for key in ['frame_ids', 'offsets', 'boxes', 'scores']}
    result = aligned_source_audit(read(fresh), read(canonical))
    result.update(fresh_sha256=sha256(fresh), canonical_sha256=sha256(canonical),
        unavailable_raw_exact_to_fresh=compare_packets(base/'CG3_IR_unavailable_rgb_boxes_raw.npz', fresh))
    return result


def main():
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    path = work/'CGA_formal_test_state.json'
    state = json.loads(path.read_text())
    assert state['status'] == 'FAILED' and all(j['status'] == 'PENDING' for j in state['jobs'].values())
    archive = work/'CGA_source_audit_attempt01_failure'
    archive.mkdir(exist_ok=False)
    shutil.copy2(path, archive/path.name)
    for name in ['tools/launch_correlation_formal_test.py', 'tools/extract_single_seed_geometry.py']:
        shutil.copy2(ROOT/name, archive/Path(name).name)
    recovery_path = ROOT/'configs/perceptual_CGA_source_recovery_20261123.json'
    assert not recovery_path.exists()
    recovery = dict(status='REGISTERED_AFTER_UNLABELED_SOURCE_FAILURE_BEFORE_ANY14_TEST_ACCURACY',
        created_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        original_failure_state_sha256=sha256(archive/path.name),
        frozen_case_specification_sha256=state['specification_sha256'],
        unchanged_producer_sha256=state['sources']['tools/extract_single_seed_geometry.py'],
        constraints=dict(final_frames_only=4, matched_box_difference_at_most_pixels=2,
            matched_score_difference_at_most=.005, unmatched_detections_at_most_per_frame=1,
            unmatched_score_strictly_below=.6, matched_06_07_memberships_exact=True),
        observed_failure='MSK-003 frame629:61 fresh vs62 canonical detections; one canonical0.3751254 NMS candidate unmatched. '
                         'Matched box difference<=.675049px, score<=.002258; all matched0.6/0.7 memberships exact.',
        method_changed=False, checkpoint_changed=False, GT_read=False, Test_accuracy_read=False,
        reporting='Retain canonical RGB29.529 and fresh RGB as separate controls, require unavailable IR tracks/all39metrics '
                  'exact to fresh. Report any canonical/fresh tracking difference; no source parity claim for weak NMS.')
    dump(recovery_path, recovery)
    state.update(status='RECOVERING_UNLABELED_SOURCE_AUDIT', original_source_failure_archive=str(archive),
        original_source_failure=state.pop('error'), recovery_plan_sha256=sha256(recovery_path),
        recovery_source_sha256=sha256(__file__), controller_pid=os.getpid())
    dump(path, state)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               MFT26IR_ROOT=plan['data_root'])
    spec_path = ROOT/'configs/perceptual_CGA_formal_test_20261123.json'
    assert sha256(spec_path) == state['specification_sha256']
    assert all(sha256(ROOT/n) == h for n, h in state['sources'].items())
    for seq, row in state['producers'].items():
        base = work/'single_seed_geometry_predictions/test'/seq
        if row['status'] == 'PENDING':
            assert seq == 'BT-002'
            for phase in ['source', 'full']:
                cmd = [sys.executable, '-u', 'tools/extract_single_seed_geometry.py', '--sequence', seq, '--split', 'test']
                cmd += ['--source-check', '8'] if phase == 'source' else ['--specification', str(spec_path)]
                log = work/(seq+'.CGA_formal_test_'+phase+'.log')
                with log.open('ab', buffering=0) as handle:
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES='2'),
                        stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                row.update(status=phase.upper()+'_RUNNING', pid=proc.pid, physical_gpu='2', command=cmd, log=str(log))
                dump(path, state)
                assert proc.wait() == 0
                if phase == 'source':
                    m = json.loads((base/'source_check_8/complete.json').read_text())
                    assert len(m['outputs']) == 7 and all(r['array_exact'] for r in m['fresh_RGB_parity'])
        while not (base/'complete.json').exists():
            # The original SN-014 worker remains the only GPU3 source worker.
            assert row['status'] == 'FULL_RUNNING'
            os.kill(row['pid'], 0)
            time.sleep(5)
        meta = json.loads((base/'complete.json').read_text())
        assert meta['status'] == 'COMPLETE_CORRELATION_OBSERVATIONS' and not meta['GT_read']
        assert meta['frames'] == state['expected_frames_from_unlabeled_source_indices'][seq]
        assert meta['specification_sha256'] == state['specification_sha256']
        assert meta['source_sha256'] == state['sources']['tools/extract_single_seed_geometry.py']
        row.update(status='COMPLETE', recovered_source_audit=audit(base, seq, plan),
                   complete_manifest_sha256=sha256(base/'complete.json'))
        dump(path, state)
    assert all(r['status'] == 'COMPLETE' for r in state['producers'].values())
    assert json.loads((work/'structural_relative_dev_state.json').read_text())['status'] == 'COMPLETE'
    state['structural_59Dev_completed_before_Test_accuracy_sha256'] = sha256(work/'structural_relative_dev_state.json')
    state['status'] = 'RUNNING_FROZEN_ALL7_TEST_REPLAYS'
    active = {}
    dump(path, state)
    while not all(j['status'] == 'COMPLETE' for j in state['jobs'].values()):
        for key, proc in list(active.items()):
            code = proc.poll()
            if code is not None:
                state['jobs'][key].update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code)
                del active[key]
                assert code == 0
        for key, row in state['jobs'].items():
            if len(active) >= 2 or row['status'] != 'PENDING':
                continue
            assert sha256(row['config']) == row['sha256']
            cmd = [sys.executable, '-u', 'tools/replay_correlation_observations.py', '--split', 'test',
                   '--config', row['config'], '--output', str(work/'CGA_formal_test'/key)]
            log = work/(key+'.CGA_formal_test_replay.log')
            with log.open('ab', buffering=0) as handle:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=''),
                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            active[key] = proc
            row.update(status='RUNNING', pid=proc.pid, command=cmd, log=str(log))
        dump(path, state)
        time.sleep(5)
    state['status'] = 'COMPLETE'
    dump(path, state)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
        path = Path(plan['work_root'])/'CGA_formal_test_state.json'
        state = json.loads(path.read_text())
        if state.get('controller_pid') == os.getpid():
            state.update(status='FAILED', error=repr(error))
            dump(path, state)
        raise
