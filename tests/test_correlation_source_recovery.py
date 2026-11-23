import numpy as np
import pytest

from tools.recover_correlation_formal_test import aligned_source_audit


def packet(last_score=.375, first_extra=False):
    boxes = np.array([[0, 0, 20, 20], [30, 0, 50, 20], [60, 0, 80, 20]], np.float32)
    return dict(frame_ids=np.array([1, 2, 3, 4, 5]),
        offsets=np.array([0, 2, 2, 2, 2, 3] if first_extra else [0, 1, 1, 1, 1, 3]),
        boxes=boxes, scores=np.array([.9, .8, last_score], np.float32))


def remove_last(p):
    return dict(frame_ids=p['frame_ids'].copy(), offsets=np.array([0, 1, 1, 1, 1, 2]),
                boxes=p['boxes'][:2].copy(), scores=p['scores'][:2].copy())


def test_tail_weak_nms_difference_is_reported_and_order_independent():
    reference = packet()
    current = remove_last(reference)
    audit = aligned_source_audit(current, reference)
    assert not audit['counts_offsets_exact'] and not audit['full_RGB_bit_exact']
    assert len(audit['changed_tail_frames'][0]['unmatched_canonical']) == 1


def test_recovery_rejects_unmatched_high_or_birth_confidence():
    reference = packet(last_score=.7)
    with pytest.raises(AssertionError):
        aligned_source_audit(remove_last(reference), reference)


def test_recovery_rejects_matched_threshold_crossing_and_nontail_difference():
    reference = packet()
    current = {k: v.copy() for k, v in reference.items()}
    reference['scores'][1] = .6001
    current['scores'][1] = .5999
    with pytest.raises(AssertionError):
        aligned_source_audit(current, reference)
    reference = packet()
    current = {k: v.copy() for k, v in reference.items()}
    current['boxes'][0, 0] = .01
    with pytest.raises(AssertionError):
        aligned_source_audit(current, reference)
