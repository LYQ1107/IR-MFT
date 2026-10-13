from types import SimpleNamespace

import numpy as np
import torch

from cfp.ambiguity_association_stable import StableAmbiguityAssociation
from cfp.candidate_relation_residual import ResidualRelationStudent
from cfp.lifecycle_crossover import make_cfp_lifecycle
from cfp.online_candidate_relation import install_candidate_relation
from cfp.online_recovery_write_quota import ObservableWriteRisk, install_quota_risk_memory
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker import matching


class Last(torch.nn.Module):
    def forward(self, history, times, valid, frame):
        index = (valid.sum(1) - 1).clamp_min(0)
        return history[torch.arange(len(history)), index], valid.any(1)


def stream(with_residual, maximum):
    BaseTrack._count = 0
    tracker = make_cfp_lifecycle(SimpleNamespace(), 'gru', Last())
    if with_residual:
        tracker = install_candidate_relation(tracker, ResidualRelationStudent().eval(), residual=True)
    tracker = install_quota_risk_memory(tracker, ObservableWriteRisk('margin'), dict(temperature=1., bias=0.))
    bank = StableAmbiguityAssociation(tracker, maximum, ambiguity_delta=1., switch_margin=.02)
    features = np.zeros((10, 512), np.float32); features[:, 0] = 1
    boxes = np.array([[80 * i, 0, 80 * i + 30, 20, .9] for i in range(10)], np.float32)
    boxes[1, :4] = [2, 0, 32, 20]
    rows = []
    original = matching.linear_assignment
    for frame in range(1, 5):
        output = bank.update(boxes, features, np.arange(10) + 10 * frame)
        assert matching.linear_assignment is original
        assert len({t.track_id for t in output}) == len(output)
        rows.append([(t.track_id, t.current_proposal_index) for t in output])
    return bank, rows


def test_zero_prediction_preserves_quota_and_alternate_matching():
    # Nested assignment transactions must rank the actual alternate assignment.
    for maximum in (1, 2):
        plain, a = stream(False, maximum)
        residual, b = stream(True, maximum)
        assert a == b
        assert plain.tracker.phase_vi_memory_counts == residual.tracker.phase_vi_memory_counts
        assert plain.tracker.quota_counts == residual.tracker.quota_counts
        assert residual.tracker.quota_counts['rejected_budget'] > 0


def test_combined_branch_histories_evidence_and_counters_are_independent():
    bank, _ = stream(True, 2)
    assert bank.statistics['branch_expansions'] > 0
    assert len(bank.bank.hypotheses) == 2
    left, right = [h.state for h in bank.bank.hypotheses]
    track_a, track_b = left.tracked_stracks[0], right.tracked_stracks[0]
    assert track_a.identity_history is not track_b.identity_history
    assert track_a.early_identity_history is not track_b.early_identity_history
    old_count = dict(right.phase_vi_memory_counts)
    left.phase_vi_memory_counts['accepted'] += 99
    assert right.phase_vi_memory_counts == old_count
    old_evidence = dict(track_b.risk_memory_evidence)
    track_a.risk_memory_evidence.clear()
    assert track_b.risk_memory_evidence == old_evidence
