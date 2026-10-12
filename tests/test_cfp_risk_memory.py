from types import SimpleNamespace

import numpy as np

from cfp.online_recovery_write import install_risk_memory


class Tracker:
    frame_id = 3
    def association_cost(self, tracks, detections):
        return np.array([[.2]])
    def update(self):
        return None


def test_provisional_write_preserves_original_score_and_memory_and_clears_pending():
    calls = []
    track = SimpleNamespace(track_id=1, current_proposal_index=5, pending=True,
        identity_history=[(1, None, None)], initial_identity_rgb=None,
        write_identity=lambda frame, ema: calls.append(frame) or True)
    detection = SimpleNamespace(current_rgb=np.array([1., 0]), current_proposal_index=5)
    tracker = install_risk_memory(Tracker(), None, None, constant_risk=1.)
    cost = tracker.association_cost([track], [detection])
    assert cost[0, 0] == .2
    assert not track.write_identity(3, .9) and not track.pending and not calls
    assert len(track.identity_history) == 1
    assert tracker.phase_vi_memory_counts['provisional_rejected'] == 1


def test_unavailable_scoring_falls_back_without_unknown_as_wrong():
    calls = []
    track = SimpleNamespace(track_id=1, current_proposal_index=5, pending=True,
        identity_history=[], write_identity=lambda frame, ema: calls.append(frame) or True)
    detection = SimpleNamespace(current_rgb=None, current_proposal_index=5)
    tracker = install_risk_memory(Tracker(), None, None, constant_risk=1.)
    tracker.association_cost([track], [detection])
    assert track.write_identity(3, .9) and calls == [3]
    assert tracker.phase_vi_memory_counts['unavailable_fallback'] == 1

