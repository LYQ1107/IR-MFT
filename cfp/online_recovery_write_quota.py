"""Per-stage bounded write rejection on actual current Hungarian assignments."""
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from .online_recovery_write_isolated import install_isolated_risk_memory
from yolox.tracker import matching


class ObservableWriteRisk(nn.Module):
    """Fixed ordinary cue controls, using exactly the same real candidate graph."""
    def __init__(self, policy):
        super().__init__()
        if policy not in ('confidence', 'margin', 'entropy'):
            raise ValueError('Declared ordinary cue required')
        self.policy = policy

    def forward(self, features, legal):
        if self.policy == 'confidence':
            return 1 - features[..., 14]
        if self.policy == 'entropy':
            return features[..., 8]
        cost = features[..., 19]
        # For each candidate edge, compare the best OTHER real column.
        width = cost.shape[-1]
        if width < 2:
            return torch.zeros_like(cost)
        sorted_cost, indices = cost.masked_fill(~legal, torch.inf).sort(-1)
        competitor = sorted_cost[..., :1].expand_as(cost).clone()
        competitor.scatter_(-1, indices[..., :1], sorted_cost[..., 1:2])
        competitor = torch.where(torch.isfinite(competitor), competitor, cost)
        return cost - competitor


def install_quota_risk_memory(tracker, estimator, calibration, fraction=.1, seed=0, random_order=False):
    if not 0 <= fraction < 1:
        raise ValueError('Rejection budget must leave a nonempty scored matched set')
    tracker = install_isolated_risk_memory(tracker, estimator, calibration, threshold=.2,
                                          constant_risk=1. if random_order else None)
    base = type(tracker)
    plain_base = base.__bases__[0]

    class QuotaTracker(base):
        def association_cost(self, tracks, detections):
            # No possible scored matching can consume a rejection budget here.
            # Keep the original cost and writes; avoid redundant risk encoding.
            if len(detections) * self.quota_fraction < 1:
                cost = plain_base.association_cost(self, tracks, detections)
                self.quota_counts['skipped_small_pool_stages'] += 1
            else:
                cost = super().association_cost(tracks, detections)
            self.quota_matrix, self.quota_pair = cost, (tracks, detections)
            return cost

        @contextmanager
        def quota_assignment(self):
            original = matching.linear_assignment

            def assignment(cost, *args, **kwargs):
                result = original(cost, *args, **kwargs)
                if cost is self.quota_matrix:
                    tracks, detections = self.quota_pair
                    choices = []
                    for row, column in result[0]:
                        track, detection = tracks[row], detections[column]
                        proposal = detection.current_proposal_index
                        risk = getattr(track, 'risk_memory_evidence', {}).get(proposal) if getattr(track, 'risk_memory_frame', -1) == self.frame_id else None
                        if risk is not None:
                            choices.append((float(risk), int(track.track_id), int(proposal), track))
                    reject = int(np.floor(len(choices) * self.quota_fraction))
                    if self.quota_random:
                        order = self.quota_rng.permutation(len(choices))
                    else:
                        order = sorted(range(len(choices)), key=lambda k: choices[k][:3], reverse=True)
                    rejected = set(order[:reject])
                    for index, (_, _, proposal, track) in enumerate(choices):
                        track.risk_memory_evidence[proposal] = float(index in rejected)
                    self.quota_counts['scored_matched'] += len(choices)
                    self.quota_counts['rejected_budget'] += reject
                    self.quota_counts['assignment_stages'] += 1
                return result

            matching.linear_assignment = assignment
            try:
                yield
            finally:
                matching.linear_assignment = original

        def update(self, *args, **kwargs):
            self.quota_matrix, self.quota_pair = None, None
            with self.quota_assignment():
                return super().update(*args, **kwargs)

    tracker.__class__ = QuotaTracker
    tracker.quota_fraction = fraction
    tracker.quota_random = random_order
    tracker.quota_rng = np.random.default_rng(seed)
    tracker.quota_counts = dict(scored_matched=0, rejected_budget=0, assignment_stages=0, skipped_small_pool_stages=0)
    tracker.quota_matrix, tracker.quota_pair = None, None
    return tracker
