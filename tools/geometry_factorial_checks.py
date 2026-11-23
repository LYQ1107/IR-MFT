"""No-label geometry-factorial reference, candidate-set and anchor checks."""
import numpy as np

from tools.correlation_geometry_checks import compare_packets, unavailable_checks


def previous_stream_checks(base, sequence, spec, work, frames=None):
    proofs = {}
    for row in spec['models']:
        if 'reference' not in row:
            continue
        ref = row['reference']
        for role in ['full', 'rgb_boxes', 'rgb_scores']:
            key = row['id'] + ('' if role == 'full' else '_'+role)
            suffix = ref['full_role_name'] if role == 'full' else role
            reference = ref['prefix'] + ('_'+suffix if suffix else '')
            for policy in ['', '_raw']:
                proofs[key+policy] = compare_packets(base/(key+policy+'.npz'),
                    work/ref['directory']/'dev'/sequence/(reference+policy+'.npz'), frames)
    assert len(proofs) == 66
    return proofs


def factorial_provenance_checks(base, spec):
    counts = dict(raw_streams=0, crossed_candidate_sets=0, RGB_score_streams=0)
    for row in spec['models']:
        if row.get('baseline'):
            continue
        for role in row['head_roles']:
            key = row['id'] + ('' if role == 'full' else '_'+role) + '_raw'
            with np.load(base/(key+'.npz')) as packet:
                assert len(packet['scores']) == len(packet['rgb_reference_scores']) == len(packet['anchor_indices'])
                assert np.isfinite(packet['rgb_reference_scores']).all()
                assert ((packet['rgb_reference_scores'] >= 0) & (packet['rgb_reference_scores'] <= 1)).all()
                assert ((packet['anchor_indices'] >= 0) & (packet['anchor_indices'] < 23625)).all()
                if role == 'rgb_scores' or row.get('feature_intervention') == 'unavailable':
                    np.testing.assert_array_equal(packet['scores'], packet['rgb_reference_scores'])
                    counts['RGB_score_streams'] += 1
            counts['raw_streams'] += 1
        for first, second in [('full', 'fused_nms_rgb_boxes'), ('rgb_boxes', 'rgb_nms_fused_boxes')]:
            prefix = row['id']
            a = base/(prefix+('' if first == 'full' else '_'+first)+'_raw.npz')
            b = base/(prefix+'_'+second+'_raw.npz')
            with np.load(a) as left, np.load(b) as right:
                for field in ['frame_ids', 'offsets', 'scores', 'anchor_indices', 'rgb_reference_scores']:
                    np.testing.assert_array_equal(left[field], right[field])
            counts['crossed_candidate_sets'] += 1
    assert counts['raw_streams'] == 115 and counts['crossed_candidate_sets'] == 46
    return counts


def all_unavailable_checks(base, sequence, plan, work, frames=None):
    proofs = unavailable_checks(base, sequence, plan, work, frames)
    from pathlib import Path
    canonical = Path(plan['frozen_detection_sources']['rgb_dev'])/'detections'/(sequence+'.npz')
    for role in ['fused_nms_rgb_boxes', 'rgb_nms_fused_boxes']:
        key = 'CG3_IR_unavailable_'+role
        proofs[key+'_raw'] = compare_packets(base/(key+'_raw.npz'), base/'CG3_source_RGB.npz', frames)
        proofs[key] = compare_packets(base/(key+'.npz'), canonical, frames)
    assert len(proofs) == 10
    return proofs
