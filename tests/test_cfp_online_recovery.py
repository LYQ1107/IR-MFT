from types import SimpleNamespace

import numpy as np
import torch

from cfp.online_recovery import install_risk_association


class Last(torch.nn.Module):
    def forward(self, past, times, valid, frame):
        assert not torch.any(valid & (times >= frame[:, None]))
        return past[:, 0], valid.any(1)


class FixedRisk(torch.nn.Module):
    def forward(self, evidence, legal):
        return evidence.new_zeros(legal.shape)


class Tracker:
    frame_id = 3
    predictor = Last()

    def association_cost(self, tracks, detections):
        return np.full((len(tracks), len(detections)), .3)

    def update(self, tracks, detections):
        return [self.association_cost(tracks, detections), self.association_cost(tracks, detections)]


def objects(missing=False):
    feature = np.zeros(512, np.float32); feature[0] = 1
    track = SimpleNamespace(initial_identity_rgb=None,
        identity_history=[(1, feature.copy(), None), (2, feature.copy(), None)],
        tlbr=np.array([0, 0, 10, 10]), start_frame=1, end_frame=2)
    detection = SimpleNamespace(current_rgb=None if missing else feature, score=.9, tlbr=np.array([0, 0, 10, 10]))
    return [track], [detection]


def test_risk_changes_high_assignment_but_preserves_later_stage_and_missing_fallback():
    tracker = install_risk_association(Tracker(), FixedRisk(), dict(temperature=1., bias=0.), 'risk_gate', .2)
    first, later = tracker.update(*objects())
    assert first[0, 0] == 1.01 and later[0, 0] == .3
    assert tracker.phase_vi_risk_counts['modified_edges'] == 1
    first, later = tracker.update(*objects(missing=True))
    assert first[0, 0] == later[0, 0] == .3


def test_disabled_mode_uses_no_new_history_or_features():
    tracker = install_risk_association(Tracker())
    first, later = tracker.update([None], [None])
    assert first[0, 0] == later[0, 0] == .3
    assert tracker.phase_vi_risk_counts['modified_edges'] == 0

