"""Explicit composition of frozen RGB prediction and observable IR utility.

Utility retains its trained EMA evidence coordinates. Accepted weights mix the
predicted RGB distance with the historical IR EMA distance, only on the existing
utility candidate subset. This is a frozen composition trial, not a gate retrained
on predictive counterfactual targets or a paired RGB/IR latent predictor.
"""
import numpy as np

from .history import anchor_recent_history
from .predictive_association import PredictiveAssociationTracker
from .utility_association import UtilityAssociationTracker
from yolox.tracker import matching


class JointAssociationTracker(PredictiveAssociationTracker):
    def __init__(self, args, utility_model=None, utility_kind='cmue', calibration=None,
                 utility_threshold=.02, utility_context_version=2, teacher_association_weight=.3,
                 no_calibration=False, no_fallback=False, **kwargs):
        super().__init__(args, **kwargs)
        self.utility_scorer = UtilityAssociationTracker(
            args, model=utility_model, kind=utility_kind, calibration=calibration,
            threshold=utility_threshold, no_calibration=no_calibration, no_fallback=no_fallback,
            appearance_weight=self.appearance_weight, ema=self.ema, proximity_iou=self.proximity_iou,
            appearance_max_distance=self.appearance_max_distance)
        self.utility_scorer.candidate_context_version = utility_context_version
        self.utility_scorer.teacher_association_weight = teacher_association_weight
        self.joint_counts = dict(accepted_ir_edges=0, rejected_or_unavailable_edges=0)

    def association_cost(self, tracks, detections):
        predicted = super().association_cost(tracks, detections)
        if predicted.size == 0 or self.appearance_weight == 0:
            self.utility_scorer.last_ir_weights = np.zeros_like(predicted)
            self.utility_scorer.last_ir_distances = np.zeros_like(predicted)
            return predicted
        scorer = self.utility_scorer
        scorer.frame_id = self.frame_id
        scorer.association_cost(tracks, detections)
        weights, ir_costs = scorer.last_ir_weights, scorer.last_ir_distances
        if not np.any(weights > 0):
            self.joint_counts['rejected_or_unavailable_edges'] += int(weights.size)
            return predicted
        raw = matching.iou_distance(tracks, detections)
        geometry = matching.fuse_score(raw, detections) if not self.args.mot20 else raw
        result = predicted.copy()
        for i, track in enumerate(tracks):
            history = list(track.identity_history)
            if self.history_view == 'anchor_recent':
                anchor = None if track.initial_identity_rgb is None else (track.initial_identity_frame, track.initial_identity_rgb, track.initial_identity_ir)
                history = anchor_recent_history(history, anchor, 8)
            if not history:
                continue
            eligible = (weights[i] > 0) & (raw[i] <= 1 - self.proximity_iou)
            if not eligible.any():
                continue
            # Recover the bounded predicted RGB distance from its unchanged
            # geometry/appearance mixture. Avoid recomputing the predictor.
            rgb_costs = (predicted[i, eligible] - (1 - self.appearance_weight) * geometry[i, eligible]) / self.appearance_weight
            if not np.isfinite(rgb_costs).all() or np.any(rgb_costs < -1e-6) or np.any(rgb_costs > 1 + 1e-6):
                raise ValueError('Joint IR edits require valid predicted RGB costs')
            appearance = (1 - weights[i, eligible]) * rgb_costs + weights[i, eligible] * ir_costs[i, eligible]
            result[i, eligible] = (1 - self.appearance_weight) * geometry[i, eligible] + self.appearance_weight * appearance
            self.joint_counts['accepted_ir_edges'] += int(eligible.sum())
        return result
