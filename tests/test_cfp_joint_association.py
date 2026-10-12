from types import SimpleNamespace

import numpy as np
import torch

from cfp.association import IdentityTrack
from cfp.joint_association import JointAssociationTracker
from cfp.predictive_association import PredictiveAssociationTracker
from yolox.tracker.kalman_filter import KalmanFilter


class RejectGate(torch.nn.Module):
    def forward(self, evidence):
        return torch.full((len(evidence),), -1.)


def inputs(ir=True):
    track = IdentityTrack([0, 0, 30, 20], .9, np.array([1., 0.], np.float32), np.array([1., 0.], np.float32) if ir else None)
    track.activate(KalmanFilter(), 1)
    track.write_identity(1, .9)
    detection = IdentityTrack([0, 0, 30, 20], .9, np.array([1., 0.], np.float32), np.array([0., 1.], np.float32) if ir else None)
    return [track], [detection]


def test_rejected_ir_and_missing_ir_preserve_predictive_scores_bit_exact():
    args = SimpleNamespace(track_thresh=.58, track_buffer=45, match_thresh=.97, mot20=False)
    for ir in (True, False):
        tracks, detections = inputs(ir)
        baseline = PredictiveAssociationTracker(args, memory='last', appearance_weight=.3)
        joint = JointAssociationTracker(args, memory='last', appearance_weight=.3, utility_model=RejectGate())
        baseline.frame_id = joint.frame_id = 2
        expected = baseline.association_cost(tracks, detections)
        assert np.array_equal(joint.association_cost(tracks, detections), expected)
        assert tracks[0].last_identity_frame == 1 and len(tracks[0].identity_history) == 1


def test_joint_fixed_ir_mixes_only_available_current_candidates_and_clears_state():
    args = SimpleNamespace(track_thresh=.58, track_buffer=45, match_thresh=.97, mot20=False)
    tracks, detections = inputs()
    baseline = PredictiveAssociationTracker(args, memory='last', appearance_weight=.3)
    joint = JointAssociationTracker(args, memory='last', appearance_weight=.3, utility_kind='fixed_matched')
    baseline.frame_id = joint.frame_id = 2
    expected = baseline.association_cost(tracks, detections)
    actual = joint.association_cost(tracks, detections)
    assert np.allclose(actual - expected, .3 * .5 * .5)
    assert joint.joint_counts['accepted_ir_edges'] == 1
    joint.association_cost([], [])
    assert joint.utility_scorer.last_ir_weights.size == 0
    # No current-frame scores or gates survive the next empty association call.
    assert joint.association_cost(tracks, []).size == 0
