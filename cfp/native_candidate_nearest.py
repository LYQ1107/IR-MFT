"""Candidate-conditioned nearest evidence over a strictly past native history."""

import numpy as np

from .association import AppearanceTracker
from yolox.tracker import matching


class CandidateNearestAppearanceTracker(AppearanceTracker):
    """Keep the best observed past identity cue separately for each candidate."""

    def __init__(self, args, *, frame_rate=25, appearance_weight=0.3,
                 proximity_iou=0.1, appearance_max_distance=0.35,
                 history_length=4, maximum_history_age=1500):
        super().__init__(args, frame_rate=frame_rate, mode="rgb",
                         appearance_weight=appearance_weight,
                         proximity_iou=proximity_iou,
                         appearance_max_distance=appearance_max_distance)
        if not 1 <= history_length <= 8 or maximum_history_age <= 0:
            raise ValueError("Registered bounded native history is invalid")
        self.history_length = int(history_length)
        self.maximum_history_age = int(maximum_history_age)
        self.candidate_nearest_counts = dict(scored_rows=0, eligible_edges=0,
                                             past_observations=0)

    def association_cost(self, tracks, detections):
        raw_geometry = matching.iou_distance(tracks, detections)
        geometry = matching.fuse_score(raw_geometry, detections) if not self.args.mot20 else raw_geometry
        if geometry.size == 0 or self.appearance_weight == 0:
            return geometry
        candidate_valid = np.array([d.current_rgb is not None for d in detections], bool)
        result = geometry.copy()
        for row, track in enumerate(tracks):
            history = [item for item in list(track.identity_history)
                       if self.frame_id - self.maximum_history_age <= item[0] < self.frame_id]
            history = history[-self.history_length:]
            if not history:
                continue
            if any(frame >= self.frame_id for frame, _, _ in history):
                raise ValueError("Current identity observation leaked into nearest history")
            past = np.asarray([item[1] for item in history], dtype=np.float32)
            if past.ndim != 2 or not np.isfinite(past).all():
                raise ValueError("Finite normalized native history required")
            valid_candidates = np.flatnonzero(candidate_valid)
            if not len(valid_candidates):
                continue
            current = np.asarray([detections[index].current_rgb for index in valid_candidates], dtype=np.float32)
            distances = np.clip((1.0 - past @ current.T) / 2.0, 0.0, 1.0)
            appearance = np.ones(len(detections), dtype=np.float64)
            appearance[valid_candidates] = distances.min(axis=0)
            legal = candidate_valid & (raw_geometry[row] <= 1 - self.proximity_iou) \
                & (appearance <= self.appearance_max_distance)
            result[row, legal] = ((1 - self.appearance_weight) * geometry[row, legal]
                                  + self.appearance_weight * appearance[legal])
            self.candidate_nearest_counts["scored_rows"] += 1
            self.candidate_nearest_counts["eligible_edges"] += int(legal.sum())
            self.candidate_nearest_counts["past_observations"] += len(history)
        return result
