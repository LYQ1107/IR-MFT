"""Discriminating branch-choice stability experiments on unchanged v1 scoring.

Retention/confirmation use current and past branch scores only. Stored alternate
memories remain isolated; incumbent output is never revised after the frame.
"""
from .ambiguity_association import AmbiguityAssociation, observed_assignment
from .hypothesis_bank import IdentityHypothesis, isolated_state
from yolox.tracker.basetrack import BaseTrack


class StableAmbiguityAssociation(AmbiguityAssociation):
    def __init__(self, tracker, maximum=2, ambiguity_delta=.05, discount=.9,
                 switch_margin=0., confirmation=1):
        super().__init__(tracker, maximum, ambiguity_delta, discount)
        if switch_margin < 0 or confirmation < 1:
            raise ValueError('Nonnegative margin and positive causal confirmation count required')
        self.switch_margin, self.confirmation = switch_margin, confirmation
        self.challenger, self.challenger_streak = None, 0
        self.statistics.update(blocked_switches=0, max_challenger_streak=0)

    def update(self, detections, features, indices):
        if self.bank.maximum == 1:
            return super().update(detections, features, indices)
        candidates = []
        for hypothesis in self.bank.hypotheses:
            state = isolated_state(hypothesis.state)
            BaseTrack._count = hypothesis.next_identity
            with observed_assignment(ambiguity_delta=self.ambiguity_delta) as observed:
                state._phase_vi_current_output = state.update(detections, features, indices)
            candidates.append(IdentityHypothesis(state, self.discount * hypothesis.score + observed.get('increment', 0.),
                BaseTrack._count, hypothesis.branch_id))
            alternate = observed.get('alternate')
            if alternate is not None:
                state = isolated_state(hypothesis.state)
                BaseTrack._count = hypothesis.next_identity
                with observed_assignment(alternative=alternate[1], ambiguity_delta=self.ambiguity_delta) as observed_alt:
                    state._phase_vi_current_output = state.update(detections, features, indices)
                candidates.append(IdentityHypothesis(state, self.discount * hypothesis.score + observed_alt.get('increment', 0.),
                    BaseTrack._count, self.bank.branch_id()))
                self.statistics['branch_expansions'] += 1
        best = min(candidates, key=lambda h: (h.score, h.branch_id))
        incumbent = next(h for h in candidates if h.branch_id == self.last_winner)
        selected = best
        if best.branch_id != incumbent.branch_id:
            self.challenger_streak = self.challenger_streak + 1 if self.challenger == best.branch_id else 1
            self.challenger = best.branch_id
            self.statistics['max_challenger_streak'] = max(self.statistics['max_challenger_streak'], self.challenger_streak)
            if incumbent.score - best.score < self.switch_margin or self.challenger_streak < self.confirmation:
                selected = incumbent
                self.statistics['blocked_switches'] += 1
        else:
            self.challenger, self.challenger_streak = None, 0
        # Keep the chosen explanation and the strongest distinct challenger.
        # Incumbent preservation prevents a marginal third-ranked old explanation
        # from being silently evicted before its confirmation rule is evaluated.
        other = min((h for h in candidates if h.branch_id != selected.branch_id),
                    key=lambda h: (h.score, h.branch_id), default=None)
        retained = [selected] + ([other] if other is not None else [])
        self.bank.advance(self.bank.frame + 1, retained)
        self.bank.hypotheses.sort(key=lambda h: (h.branch_id != selected.branch_id, h.score, h.branch_id))
        self.statistics['branch_switches'] += int(selected.branch_id != self.last_winner)
        self.statistics['frames'] += 1
        self.statistics['hypotheses_peak'] = max(self.statistics['hypotheses_peak'], len(self.bank.hypotheses))
        self.tracker, self.last_winner = selected.state, selected.branch_id
        BaseTrack._count = selected.next_identity
        output = self.tracker._phase_vi_current_output
        if len({int(t.track_id) for t in output}) != len(output):
            raise ValueError('Duplicate current identity')
        return output

