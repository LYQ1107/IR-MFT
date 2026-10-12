"""Strictly online K=1/K=2 assignment explanations over frozen GRU lifecycle.

K=2 branches only on a near-tied legal two-row assignment swap. Each branch owns
its accepted history, motion and lifecycle. Future observations choose future
outputs; no old output can be edited. Cost accumulation is an explicit heuristic,
not a probabilistic posterior. This module never reads GT or annotation labels.
"""
from contextlib import contextmanager

import numpy as np

from .hypothesis_bank import HypothesisBank, IdentityHypothesis, isolated_state
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker import matching


def alternative_pair(cost, matches, threshold, ambiguity_delta):
    """Best feasible local pair swap; no duplicate candidate assignments."""
    pairs = np.asarray(matches, dtype=int).reshape(-1, 2)
    best = None
    for a in range(len(pairs)):
        for b in range(a + 1, len(pairs)):
            i, j = pairs[a]; k, l = pairs[b]
            if cost[i, l] > threshold or cost[k, j] > threshold:
                continue
            difference = float(cost[i, l] + cost[k, j] - cost[i, j] - cost[k, l])
            if -1e-6 <= difference <= ambiguity_delta and (best is None or difference < best[0]):
                alternate = pairs.copy()
                alternate[a, 1], alternate[b, 1] = l, j
                best = (difference, alternate)
    return best


@contextmanager
def observed_assignment(alternative=None, ambiguity_delta=.05):
    """Process-local transaction; restore the original Hungarian function always."""
    original = matching.linear_assignment
    evidence = {}
    calls = 0

    def assign(cost, thresh):
        nonlocal calls
        result = original(cost, thresh)
        calls += 1
        if calls == 1:
            primary = np.asarray(result[0]).reshape(-1, 2)
            evidence.update(cost=np.asarray(cost).copy(), threshold=float(thresh),
                primary=primary.copy(), alternate=alternative_pair(cost, primary, thresh, ambiguity_delta),
                unmatched_rows=len(result[1]), track_rows=cost.shape[0])
            if alternative is not None:
                alt = np.asarray(alternative, int).reshape(-1, 2)
                if (len(alt) != len(primary) or set(alt[:, 0]) != set(primary[:, 0]) or
                        set(alt[:, 1]) != set(primary[:, 1]) or
                        len(set(alt[:, 1])) != len(alt) or np.any(cost[alt[:, 0], alt[:, 1]] > thresh)):
                    raise ValueError('Alternative must be a feasible bijective swap of accepted rows/columns')
                result = (alt, result[1], result[2])
            accepted = np.asarray(result[0]).reshape(-1, 2)
            evidence['increment'] = float((cost[accepted[:, 0], accepted[:, 1]].sum() + len(result[1]) * thresh) / max(cost.shape[0], 1))
        return result

    matching.linear_assignment = assign
    try:
        yield evidence
    finally:
        matching.linear_assignment = original


class AmbiguityAssociation:
    def __init__(self, tracker, maximum=2, ambiguity_delta=.05, discount=.9):
        if ambiguity_delta < 0 or not 0 <= discount <= 1:
            raise ValueError('Finite nonnegative ambiguity and bounded discount required')
        self.bank = HypothesisBank(tracker, maximum)
        self.ambiguity_delta, self.discount = ambiguity_delta, discount
        self.statistics = dict(branch_expansions=0, branch_switches=0, frames=0, hypotheses_peak=1)
        self.last_winner = 0
        self.tracker = tracker

    def update(self, detections, features, indices):
        if self.bank.maximum == 1:
            # Exact original path; do not change matching, counters or memory.
            result = self.tracker.update(detections, features, indices)
            self.statistics['frames'] += 1
            return result
        candidates = []
        for hypothesis in self.bank.hypotheses:
            state = isolated_state(hypothesis.state)
            BaseTrack._count = hypothesis.next_identity
            with observed_assignment(ambiguity_delta=self.ambiguity_delta) as observed:
                state._phase_vi_current_output = state.update(detections, features, indices)
            next_identity = BaseTrack._count
            score = self.discount * hypothesis.score + observed.get('increment', 0.)
            candidates.append(IdentityHypothesis(state, score, next_identity, hypothesis.branch_id))
            alternate = observed.get('alternate')
            if alternate is not None:
                alt_state = isolated_state(hypothesis.state)
                BaseTrack._count = hypothesis.next_identity
                with observed_assignment(alternative=alternate[1], ambiguity_delta=self.ambiguity_delta) as alt_observed:
                    alt_state._phase_vi_current_output = alt_state.update(detections, features, indices)
                candidates.append(IdentityHypothesis(alt_state,
                    self.discount * hypothesis.score + alt_observed.get('increment', 0.),
                    BaseTrack._count, self.bank.branch_id()))
                self.statistics['branch_expansions'] += 1
        winner = self.bank.advance(self.bank.frame + 1, candidates)
        self.statistics['branch_switches'] += int(winner.branch_id != self.last_winner)
        self.statistics['hypotheses_peak'] = max(self.statistics['hypotheses_peak'], len(self.bank.hypotheses))
        self.statistics['frames'] += 1
        self.last_winner, self.tracker = winner.branch_id, winner.state
        BaseTrack._count = winner.next_identity
        output = self.tracker._phase_vi_current_output
        if len({int(t.track_id) for t in output}) != len(output):
            raise ValueError('Duplicate output identity in winning branch')
        return output

