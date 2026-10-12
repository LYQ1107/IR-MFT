from types import SimpleNamespace

import numpy as np

from cfp.ambiguity_association import AmbiguityAssociation
from cfp.ambiguity_association_stable import StableAmbiguityAssociation
from yolox.tracker import matching


class FakeTracker:
    predictor = None
    def __init__(self):
        self.frame_id = 0
        self.initial_alternate = False
        self.history = []
    def update(self, detections, features, indices):
        self.frame_id += 1
        if self.frame_id == 1:
            cost = np.array([[.1, .11], [.11, .1]])
        else:
            diagonal = .18 if self.initial_alternate else .2
            cost = np.array([[diagonal, 1.], [1., diagonal]])
        pairs, _, _ = matching.linear_assignment(cost, .97)
        if self.frame_id == 1:
            self.initial_alternate = int(pairs[0, 1]) == 1
        self.history.append(pairs.copy())
        return [SimpleNamespace(track_id=int(i + 1), current_proposal_index=int(j)) for i, j in pairs]


def test_confirmation_and_margin_preserve_current_incumbent_until_past_evidence_accumulates():
    plain = AmbiguityAssociation(FakeTracker())
    confirmation = StableAmbiguityAssociation(FakeTracker(), confirmation=2)
    margin = StableAmbiguityAssociation(FakeTracker(), switch_margin=.02)
    for model in (plain, confirmation, margin):
        model.update(None, None, None)
        model.update(None, None, None)
    assert plain.statistics['branch_switches'] == 1
    assert confirmation.statistics['branch_switches'] == margin.statistics['branch_switches'] == 0
    for model in (confirmation, margin):
        assert model.statistics['blocked_switches'] == 1
        model.update(None, None, None)
        assert model.statistics['branch_switches'] == 1 and len(model.bank.hypotheses) == 2
        assert model.bank.hypotheses[0].state.history is not model.bank.hypotheses[1].state.history

