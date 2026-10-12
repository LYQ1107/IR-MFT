"""Bounded, branch-isolated causal state bank. Historical output is immutable."""
import copy
from dataclasses import dataclass


@dataclass
class IdentityHypothesis:
    state: object
    score: float
    next_identity: int
    branch_id: int


def isolated_state(state):
    # Frozen inference models may be shared; mutable track/memory/motion state may not.
    memo = {}
    predictor = getattr(state, 'predictor', None)
    if predictor is not None:
        memo[id(predictor)] = predictor
    return copy.deepcopy(state, memo)


class HypothesisBank:
    def __init__(self, initial, maximum=2):
        if maximum not in (1, 2):
            raise ValueError('Initial experiment bounds K to 1 or 2')
        self.maximum = maximum
        self.hypotheses = [IdentityHypothesis(initial, 0., 0, 0)]
        self.frame = 0
        self._branch_counter = 0

    def branch_id(self):
        self._branch_counter += 1
        return self._branch_counter

    def advance(self, frame, hypotheses):
        if frame != self.frame + 1 or not hypotheses:
            raise ValueError('Exactly one causal frame step with nonempty hypotheses required')
        self.frame = frame
        self.hypotheses = sorted(hypotheses, key=lambda h: (h.score, h.branch_id))[:self.maximum]
        if len({id(h.state) for h in self.hypotheses}) != len(self.hypotheses):
            raise ValueError('Branches must own separate mutable state')
        return self.hypotheses[0]

