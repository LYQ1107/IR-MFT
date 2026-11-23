"""No-GT proposal parity for balanced semantic/geometry replication."""
import json
from pathlib import Path

import numpy as np
from tools.run_pseudo_ir_comparison import sha256


def compare_packets(current, reference, frames=None):
    with np.load(current) as a, np.load(reference) as b:
        n = len(a['frame_ids']) if frames is None else frames
        assert len(a['frame_ids']) == n
        stop = int(b['offsets'][n])
        for key, target in [('frame_ids', b['frame_ids'][:n]), ('offsets', b['offsets'][:n+1]),
                            ('boxes', b['boxes'][:stop]), ('scores', b['scores'][:stop])]:
            np.testing.assert_array_equal(a[key], target)
    return dict(all4_arrays_exact=True, frames=n, current_sha256=sha256(current), reference_sha256=sha256(reference))


def trained_anchor_checks(base, sequence, spec, work, frames=None):
    proofs = {}
    for row in spec['models']:
        if not row.get('anchor') or row.get('baseline'):
            continue
        for role in row['head_roles']:
            key = row['id'] + ('' if role == 'full' else '_'+role)
            old_key = row['reference_model_id'] + ('_none' if role == 'full' else '_'+role)
            for suffix in ['', '_raw']:
                old = work/'temporal_geometry_predictions/dev'/sequence/(old_key+suffix+'.npz')
                proofs[key+suffix] = compare_packets(base/(key+suffix+'.npz'), old, frames)
    assert len(proofs) == 18
    return proofs


def unavailable_checks(base, sequence, plan, work, frames=None):
    proofs = {}; fresh = base/'CG3_source_RGB.npz'
    canonical = Path(plan['frozen_detection_sources']['rgb_dev'])/'detections'/(sequence+'.npz')
    for role in ['full', 'rgb_boxes', 'rgb_scores']:
        key = 'CG3_IR_unavailable' + ('' if role == 'full' else '_'+role)
        proofs[key+'_raw'] = compare_packets(base/(key+'_raw.npz'), fresh, frames)
        proofs[key] = compare_packets(base/(key+'.npz'), canonical, frames)
    return proofs


def fresh_RGB_audit(base, sequence, plan, work):
    current = base/'CG3_source_RGB.npz'
    reference = Path(plan['frozen_detection_sources']['rgb_dev'])/'detections'/(sequence+'.npz')
    previous_path = work/'correlation_RGB_parity'/(sequence+'.json')
    previous = json.loads(previous_path.read_text())
    allowed = {r['frame']: r for r in previous['mismatches']}; mismatches = []
    with np.load(current) as a, np.load(reference) as b:
        for key in ['frame_ids', 'offsets']:
            np.testing.assert_array_equal(a[key], b[key])
        for frame in a['frame_ids']:
            lo, hi = map(int, a['offsets'][frame-1:frame+1])
            x = np.c_[a['boxes'][lo:hi], a['scores'][lo:hi]]
            y = np.c_[b['boxes'][lo:hi], b['scores'][lo:hi]]
            if np.array_equal(x, y):
                continue
            assert int(frame) in allowed
            old = allowed[int(frame)]
            assert float(abs(x[:, :4]-y[:, :4]).max()) <= old['max_box_difference']+1e-7
            assert float(abs(x[:, 4]-y[:, 4]).max()) <= old['max_score_difference']+1e-7
            for threshold in [.6, .7]:
                np.testing.assert_array_equal(x[:, 4] > threshold, y[:, 4] > threshold)
            mismatches.append(int(frame))
    return dict(counts_offsets_exact=True, known_tail_mismatch_frames=mismatches,
                reference_sha256=sha256(reference), current_sha256=sha256(current),
                preexisting_full_audit_sha256=sha256(previous_path),
                scope='Only previously quantified per-sequence tail-batch differences; no0.6/0.7 membership changes. Not full freshRGB bit-exactness.')
