from types import SimpleNamespace

import numpy as np
import torch

from cfp.lifecycle_crossover import make_cfp_lifecycle
from cfp.online_candidate_relation import install_candidate_relation, RawMeanCandidateControl
from cfp.candidate_relation_residual import ResidualRelationStudent
from yolox.tracker.basetrack import BaseTrack


def test_relation_runtime_current_past_only_and_fallback_lifecycle():
    BaseTrack._count = 0
    tracker = make_cfp_lifecycle(SimpleNamespace(), 'gru', memory_mode='last', read_view='early3_recent5')
    tracker = install_candidate_relation(tracker, RawMeanCandidateControl().eval())
    features = np.eye(2, 512, dtype=np.float32)
    detections = np.array([[0, 0, 30, 20, .9], [80, 0, 110, 20, .9]])
    for frame in range(1, 12):
        output = tracker.update(detections, features, np.arange(2) + 2 * frame)
        assert len(output) == 2
        assert len({t.track_id for t in output}) == 2
        assert all(t.frame_id == frame for t in output)
    assert tracker.relation_counts['scored_rows'] == 20
    assert tracker.relation_counts['null_rejected_rows'] == 0
    assert all(len(t.early_identity_history) == 3 and len(t.identity_history) == 8 for t in output)


def test_raw_control_explicit_original_appearance_region():
    h = torch.zeros(1, 8, 512); h[0, 0, 0] = 1
    times = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]])
    candidates = h[:, :1].clone(); evidence = torch.zeros(1, 1, 21); evidence[..., 14] = .9
    a = RawMeanCandidateControl(False)(h, times, times > 0, torch.tensor([2]), candidates, torch.ones(1, 1, dtype=torch.bool), evidence)
    b = RawMeanCandidateControl(True)(h, times, times > 0, torch.tensor([2]), candidates, torch.ones(1, 1, dtype=torch.bool), evidence)
    assert torch.allclose(a['scores'][:, 0], torch.tensor([-7.]))
    assert torch.allclose(b['scores'][:, 0], torch.tensor([-10.]))


def test_disabled_and_zero_residual_preserve_original_stream():
    first = make_cfp_lifecycle(SimpleNamespace(), 'gru', memory_mode='last')
    assert install_candidate_relation(first, None) is first
    streams = []
    for enabled in [False, True]:
        BaseTrack._count = 0
        tracker = make_cfp_lifecycle(SimpleNamespace(), 'gru', memory_mode='last')
        if enabled:
            tracker = install_candidate_relation(tracker, ResidualRelationStudent().eval(), residual=True)
        features = np.eye(2, 512, dtype=np.float32)
        detections = np.array([[0, 0, 30, 20, .9], [80, 0, 110, 20, .9]])
        rows = []
        for frame in range(1, 12):
            output = tracker.update(detections, features, np.arange(2) + 2 * frame)
            rows.append([(t.track_id, t.current_proposal_index, tuple(t.tlbr), tuple(t.identity_proposals)) for t in output])
        streams.append(rows)
    assert streams[0] == streams[1]
