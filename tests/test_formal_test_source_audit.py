"""Test provenance must reject mismatches before opening evaluation labels."""
from pathlib import Path

import numpy as np
import pytest

from tools.launch_correlation_formal_test import source_audit


def packets(tmp_path, *, drift=False):
    base = tmp_path/'produced'; base.mkdir()
    canonical = tmp_path/'canonical'/'detections'; canonical.mkdir(parents=True)
    fields = dict(frame_ids=np.array([1, 2]), offsets=np.array([0, 1, 2]),
                  boxes=np.array([[1, 2, 20, 30], [10, 12, 40, 50]], np.float32),
                  scores=np.array([.8, .3], np.float32))
    np.savez(canonical/'test-sequence.npz', **fields)
    fresh = {k: v.copy() for k, v in fields.items()}
    if drift:
        fresh['boxes'][1, 0] += .5
        fresh['scores'][1] += .0005
    np.savez(base/'CG3_source_RGB.npz', **fresh)
    np.savez(base/'CG3_IR_unavailable_rgb_boxes_raw.npz', **fresh)
    plan = dict(frozen_detection_sources=dict(rgb_test=str(canonical.parent)))
    return base, plan, fresh


def test_small_batch_rounding_is_reported_without_claiming_exact_identity(tmp_path):
    base, plan, _ = packets(tmp_path, drift=True)
    report = source_audit(base, 'test-sequence', plan)
    assert report['frames_with_numeric_difference'] == [2]
    assert not report['full_RGB_bit_exact']
    assert report['unavailable_raw_exact_to_fresh']['all4_arrays_exact']


def test_threshold_membership_drift_blocks_accuracy_even_with_tiny_score_change(tmp_path):
    base, plan, fields = packets(tmp_path)
    canonical = Path(plan['frozen_detection_sources']['rgb_test'])/'detections/test-sequence.npz'
    fields['scores'][1] = .5999; np.savez(canonical, **fields)
    fields['scores'][1] = .6001
    np.savez(base/'CG3_source_RGB.npz', **fields)
    np.savez(base/'CG3_IR_unavailable_rgb_boxes_raw.npz', **fields)
    with pytest.raises(AssertionError):
        source_audit(base, 'test-sequence', plan)


def test_auxiliary_off_must_preserve_the_entire_fresh_packet(tmp_path):
    base, plan, fields = packets(tmp_path)
    fields['boxes'][1, 0] += .01
    np.savez(base/'CG3_IR_unavailable_rgb_boxes_raw.npz', **fields)
    with pytest.raises(AssertionError):
        source_audit(base, 'test-sequence', plan)
