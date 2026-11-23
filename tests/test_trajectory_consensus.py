from types import SimpleNamespace

import numpy as np
import pytest

from yolox.tracker.association_calibrator import AssociationCalibrator
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.calibrated_foreground_tracker import CalibratedForegroundTracker
from yolox.tracker.trajectory_consensus import anchor_consensus, TrajectoryConsensusTracker
from test_calibrated_tracker import data_with_descriptors
from test_foreground_tracker import assert_same, with_transport


def test_consensus_rejects_single_and_conflicting_anchors():
    assert anchor_consensus([[.99, .01]], [[1., 0.]]) is None
    assert anchor_consensus([[.99, .01], [.01, .99]], [[1., 0.], [0., 1.]]) is None
    assert anchor_consensus([[.99, .01], [.99, .01]], [[0., 0.], [0., 0.]]) is None


def test_two_agreeing_anchors_tolerate_one_corrupted_anchor():
    p = [[.95, .05], [.96, .04], [.02, .98]]
    votes = [[.8, .1], [.9, .0], [.0, .9]]
    combined, vote, best, fraction = anchor_consensus(p, votes)
    assert best == 0 and fraction == 2/3
    np.testing.assert_allclose(combined, [.95, .05])
    assert vote[0] >= .15


def test_disabled_module_preserves_fitted_parent_tracks_exactly():
    data = data_with_descriptors(); model = AssociationCalibrator()
    config = dict(trajectory_consensus=False, preserve_distinct_objects=True, global_birth_threshold=.7)
    output = []
    for factory in [CalibratedForegroundTracker, TrajectoryConsensusTracker]:
        BaseTrack._count = 0
        tracker = factory(config, model=model, cue='ir')
        output.append([tracker.update(frame, data) for frame in range(1, 4)])
    assert_same(*output)


def test_future_object_anchor_is_rejected_before_a_pair_read():
    tracker = TrajectoryConsensusTracker({}, model=AssociationCalibrator(), cue='rgb')
    tracker.current_start = 0
    track = SimpleNamespace(object_observations=[dict(frame=5, index=0)])
    observation = SimpleNamespace(evidence=dict(index=0))
    with pytest.raises(ValueError, match='precede'):
        tracker.rank_cost([track], [observation], np.array([[.1]]), 4, 'primary')


def test_future_descriptors_and_points_preserve_active_consensus_prefix():
    boxes = [[x+2*f, 20, x+2*f+25, 40] for f in range(6) for x in [20, 100]]
    data = with_transport(boxes, [.95]*12, np.arange(0, 13, 2))
    descriptors = np.tile(np.eye(2, dtype=np.float32), (6, 1))
    for key in ['embedding', 'rgb_mask_embedding', 'ir_mask_embedding', 'rgb_hog', 'ir_hog', 'color']:
        data[key] = descriptors.copy()
    data['quality'] = np.ones((12, 2), np.float32)
    other = {k: v.copy() if isinstance(v, np.ndarray) else {name: arr.copy() for name, arr in v.items()} for k, v in data.items()}
    for key in ['boxes', 'embedding', 'rgb_mask_embedding', 'ir_mask_embedding', 'rgb_hog', 'ir_hog', 'color']:
        other[key][8:] = -1000
    for lag in [1, 2, 4, 8]:
        for key, arr in other['foreground_transport'].items():
            if key.startswith(f'vote_{lag}_'):
                arr[other['foreground_transport'][f'vote_offsets_{lag}'][4]:] = 1000
    results = []; counts = []
    for packet in [data, other]:
        BaseTrack._count = 0
        tracker = TrajectoryConsensusTracker(dict(preserve_clear_motion=False), model=AssociationCalibrator(), cue='ir')
        results.append([tracker.update(frame, packet) for frame in range(1, 5)])
        counts.append(tracker.statistics['multi_history_rows'])
    assert min(counts) > 0
    assert_same(*results)
