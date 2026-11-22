import numpy as np
import torch

from yolox.tracker.association_calibrator import AssociationCalibrator
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.calibrated_foreground_tracker import CalibratedForegroundTracker
from yolox.tracker.native_query_tracker import NativeQueryTracker
from test_calibrated_tracker import data_with_descriptors
from test_foreground_tracker import assert_same


class IdentityContrast(torch.nn.Module):
    def increment(self, base, cue, reference, available):
        return torch.where(available, 4*cue[..., 0], torch.zeros_like(cue[..., 0]))


def owner(**config):
    return NativeQueryTracker(dict(native_cue='ir', **config), model=AssociationCalibrator(),
                              native_model=IdentityContrast(), prior=dict(t=0, a=0, b=0, c=0))


def example():
    data = data_with_descriptors()
    data['native_reference'] = np.ones((6, 2), np.float32)/np.sqrt(2)
    data['native_auxiliary'] = np.tile(np.eye(2, dtype=np.float32), (3, 1))
    return data


def test_disabled_routes_exactly_preserve_the_full_frozen_RGB_tracker():
    data = example()
    BaseTrack._count = 0
    parent = CalibratedForegroundTracker({}, model=AssociationCalibrator(), cue='rgb')
    expected = [parent.update(f, data) for f in range(1, 4)]
    BaseTrack._count = 0
    tracker = owner(query_ranking=False, prebirth_recovery=False)
    assert_same(expected, [tracker.update(f, data) for f in range(1, 4)])


def test_zero_native_increment_keeps_the_RGB_null_posterior_without_IR_inputs():
    data = example()
    del data['native_auxiliary']
    data['quality'] = data['quality'][:, :1]
    tracker = owner(disable_native_increment=True)
    tracker.data = data
    packet = tracker.query_posterior(2, dict(frame=1, index=0))
    np.testing.assert_array_equal(packet['delta'], np.zeros((2, 2)))
    np.testing.assert_array_equal(packet['probability'], np.full((2, 2), np.float32(1/3)))
    np.testing.assert_array_equal(packet['null'], np.full(2, np.float32(1/3)))


def test_native_identity_can_open_the_correct_crossing_without_point_support():
    data = example()
    for key, value in data['foreground_transport'].items():
        if key.startswith('vote_') and 'offsets' not in key:
            value[:] = 0
    BaseTrack._count = 0
    tracker = owner(association_weight=0, preserve_distinct_objects=True)
    outputs = [tracker.update(f, data) for f in range(1, 4)]
    centers = {row[2]: row[1][0]+row[1][2]/2 for row in outputs[1]}
    assert centers[1] > centers[2]
    assert tracker.statistics['native_opened_edges'] >= 2


def test_future_native_crops_cannot_change_a_returned_prefix():
    first = example()
    second = {key: (value.copy() if isinstance(value, np.ndarray)
                    else {k: v.copy() for k, v in value.items()}) for key, value in first.items()}
    second['native_auxiliary'][4:] = -1000
    second['native_reference'][4:] = 1000
    outputs = []
    for data in [first, second]:
        BaseTrack._count = 0
        tracker = owner(prebirth_recovery=True, identity_retention=128)
        outputs.append([tracker.update(f, data) for f in range(1, 3)])
    assert_same(*outputs)


def test_null_prevents_retained_identity_recovery_and_future_source_is_rejected():
    data = example()
    tracker = owner()
    tracker.prior['b'] = 20
    tracker.data = data
    scores, _ = tracker.recovery_scores(2, dict(frame=1, index=0), data)
    np.testing.assert_array_equal(scores, np.zeros(2))
    try:
        tracker.query_inputs(2, dict(frame=2, index=2))
    except ValueError:
        pass
    else:
        raise AssertionError('Future/current source evidence was admitted')
