"""Causal candidate-relative student integration, with original later stages."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .candidate_graph import CandidateGraph
from .history import early_recent_history
from .recoverability import observable_candidate_features
from .predictive import checked_history
from yolox.tracker import matching


class RawMeanCandidateControl(nn.Module):
    def __init__(self, original_eligibility=False, weight=.3, native_geometry=False):
        super().__init__()
        self.original_eligibility, self.weight = original_eligibility, weight
        self.native_geometry = native_geometry

    def forward(self, history, times, valid, frame, candidates, legal, evidence):
        source, _, _, available, _ = checked_history(history, times, valid, frame, 'log1p_15000')
        mean = F.normalize(source.sum(1), dim=-1)
        similarity = torch.einsum('bd,bmd->bm', mean, F.normalize(candidates.float(), dim=-1))
        geometry = 1 - evidence[..., 9] * evidence[..., 14]
        cost = (1 - self.weight) * geometry + self.weight * (1 - similarity).clamp(0, 2) / 2
        if self.original_eligibility:
            cost = torch.where(evidence[..., 9] >= .1, cost, geometry)
        cost = torch.where(available[:, None], cost, geometry)
        scores = (-cost / .1).masked_fill(~legal, -torch.inf)
        return dict(scores=torch.cat([scores, scores.new_full((len(history), 1), -.97 / .1)], dim=1),
                    history_available=available)


def install_candidate_relation(tracker, student, use_null=True, residual=False):
    if student is None:
        return tracker
    base = type(tracker)

    class RelationTracker(base):
        def association_cost(self, tracks, detections):
            baseline = super().association_cost(tracks, detections)
            # Supervision and initial intervention cover the high stage only.
            if not baseline.size or any(not track.is_activated for track in tracks) or any(d.current_rgb is None for d in detections):
                return baseline
            n, m = baseline.shape
            past = np.zeros((n, 8, 512), np.float32)
            times = np.zeros((n, 8), np.int64)
            valid = np.zeros((n, 8), bool)
            for i, track in enumerate(tracks):
                observed_early = getattr(track, 'early_identity_history', [])
                history = early_recent_history(list(track.identity_history), observed_early)
                for j, (frame, rgb, _) in enumerate(history):
                    past[i, j], times[i, j], valid[i, j] = rgb, frame, True
            inputs = [torch.from_numpy(v) for v in (past, times, valid)]
            current = torch.tensor(np.array([d.current_rgb for d in detections]), dtype=torch.float32)
            graph = CandidateGraph(*inputs, current,
                torch.tensor(np.array([t.tlbr for t in tracks]), dtype=torch.float32),
                torch.tensor(np.array([d.tlbr for d in detections]), dtype=torch.float32),
                torch.tensor([d.score for d in detections], dtype=torch.float32),
                torch.from_numpy(baseline.astype(np.float32)), torch.ones(n, m, dtype=torch.bool),
                torch.tensor([self.frame_id - t.start_frame for t in tracks]),
                torch.tensor([self.frame_id - t.end_frame for t in tracks]), self.frame_id)
            with torch.inference_mode():
                evidence = observable_candidate_features(graph)
                if getattr(self.relation_student, 'native_geometry', False):
                    evidence[..., 9] = torch.from_numpy((1 - matching.iou_distance(tracks, detections)).astype(np.float32))
                output = self.relation_student(*inputs, torch.full((n,), self.frame_id, dtype=torch.int64),
                    current[None].expand(n, m, 512), graph.legal, evidence)
            # Scores use the declared -cost/.1 scale. Apply the explicit row NULL
            # decision only when its learned score exceeds every real candidate.
            scores = output['scores'].numpy()
            if self.relation_residual:
                delta = output['candidate_delta'].numpy()
                eligible = matching.iou_distance(tracks, detections) <= .9
                result = baseline - delta * eligible
            else:
                result = (-scores[:, :m] * .1).astype(np.float64)
            available = output['history_available'].numpy()
            if self.relation_use_null and not self.relation_residual:
                rejected = available & (scores[:, -1] > scores[:, :m].max(1))
                result[rejected] = 1.01
                self.relation_counts['null_rejected_rows'] += int(rejected.sum())
            result[~available] = baseline[~available]
            self.relation_counts['scored_rows'] += int(available.sum())
            return result

        def update(self, *args, **kwargs):
            output = super().update(*args, **kwargs)
            if self.relation_residual:
                # The ordinary anchor/recent prior is untouched. Supplementary
                # early observations are collected only after accepted writes.
                for track in self.tracked_stracks:
                    early = getattr(track, 'early_identity_history', [])
                    times = {row[0] for row in early}
                    for row in track.identity_history:
                        if row[0] not in times and len(early) < 3:
                            early.append(row); times.add(row[0])
                    track.early_identity_history = early
                live = {id(track) for track in self.tracked_stracks + self.lost_stracks}
                for track in self.removed_stracks:
                    if id(track) not in live:
                        track.early_identity_history = []
            return output

    tracker.__class__ = RelationTracker
    tracker.relation_student = student
    tracker.relation_use_null = use_null
    tracker.relation_residual = residual
    tracker.relation_counts = dict(scored_rows=0, null_rejected_rows=0)
    return tracker
