"""Causal candidate-quality attenuation for native appearance association."""

import numpy as np

from .association import AppearanceTracker
from yolox.tracker import matching


class VisibilityQualityAppearanceTracker(AppearanceTracker):
    """Attenuate only the appearance blend by current crop content fraction."""

    def __init__(self, args, *, frame_rate=25, appearance_weight=0.3,
                 proximity_iou=0.1, appearance_max_distance=0.35):
        super().__init__(args, frame_rate=frame_rate, mode="rgb",
                         appearance_weight=appearance_weight,
                         proximity_iou=proximity_iou,
                         appearance_max_distance=appearance_max_distance)
        self._visibility_by_proposal = {}
        self.visibility_quality_counts = dict(scored_rows=0, eligible_edges=0,
                                              quality_sum=0.0, quality_count=0)

    def update(self, output_results, img_info, img_size, rgb_features=None,
               ir_features=None, proposal_indices=None, visibility4=None):
        if visibility4 is None or np.asarray(visibility4).shape != (len(output_results), 4):
            raise ValueError("Current native visibility4 is required and aligned")
        values = np.asarray(visibility4, dtype=np.float32)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("Native visibility4 must be finite in [0, 1]")
        indices = np.arange(len(output_results), dtype=np.int64) if proposal_indices is None else np.asarray(proposal_indices)
        if indices.shape != (len(output_results),) or len(np.unique(indices)) != len(indices):
            raise ValueError("Proposal indices must align with native visibility4")
        self._visibility_by_proposal = {
            int(index): float(values[row].mean()) for row, index in enumerate(indices)
        }
        try:
            return super().update(output_results, img_info, img_size, rgb_features,
                                  ir_features, proposal_indices)
        finally:
            self._visibility_by_proposal = {}

    def association_cost(self, tracks, detections):
        raw_geometry = matching.iou_distance(tracks, detections)
        geometry = matching.fuse_score(raw_geometry, detections) if not self.args.mot20 else raw_geometry
        if geometry.size == 0 or self.appearance_weight == 0:
            return geometry
        result = geometry.copy()
        for row, track in enumerate(tracks):
            if track.memory_rgb is None:
                continue
            if track.last_identity_frame >= self.frame_id:
                raise ValueError("Current observation leaked into association memory")
            valid_rgb = np.array([det.current_rgb is not None for det in detections])
            appearance = np.array([
                float(np.clip((1 - track.memory_rgb @ det.current_rgb) / 2, 0, 1))
                if det.current_rgb is not None else 1.0 for det in detections
            ])
            legal = valid_rgb & (raw_geometry[row] <= 1 - self.proximity_iou) \
                & (appearance <= self.appearance_max_distance)
            for col in np.flatnonzero(legal):
                proposal = detections[col].current_proposal_index
                quality = self._visibility_by_proposal.get(int(proposal))
                if quality is None:
                    raise ValueError("Current proposal visibility is missing")
                effective_weight = self.appearance_weight * quality
                result[row, col] = ((1 - effective_weight) * geometry[row, col]
                                    + effective_weight * appearance[col])
                self.visibility_quality_counts["quality_sum"] += quality
                self.visibility_quality_counts["quality_count"] += 1
            self.visibility_quality_counts["scored_rows"] += 1
            self.visibility_quality_counts["eligible_edges"] += int(legal.sum())
        return result
