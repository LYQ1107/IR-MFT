"""Current geometry-matched RGB scores for physical association only."""
import numpy as np

from yolox.tracker.evidence_byte_tracker import EvidenceByteTracker
from yolox.tracker import matching


def geometry_reference_scores(boxes, scores, reference_boxes, reference_scores):
    result = np.asarray(scores, np.float32).copy()
    protected = np.zeros(len(result), dtype=bool)
    selected = np.flatnonzero(np.asarray(reference_scores) >= .6)
    if not len(boxes) or not len(selected):
        return result, protected
    overlaps = matching.ious(np.asarray(boxes), np.asarray(reference_boxes)[selected])
    pairs, _, _ = matching.linear_assignment(1-overlaps, .3)
    for current, reference in pairs:
        result[current] = reference_scores[selected[reference]]
        protected[current] = True
    return result, protected


class SemanticPhysicalScoreTracker(EvidenceByteTracker):
    def __init__(self, config):
        if any(config.get(k, 0) for k in ['appearance_penalty', 'shape_penalty', 'color_penalty']):
            raise ValueError('This isolated score mechanism does not mix appearance penalties')
        super().__init__(config)
        self.statistics.update(physical_score_columns=0, score_opened_edges=0, score_closed_edges=0)

    def observations(self, frame, data, indices):
        one = np.ones(1, np.float32)
        return [dict(owner=self, index=int(index), frame=int(frame), appearance=one.copy(), shape=one.copy(),
                     color=one.copy(), association_score=float(data['association_scores'][index])) for index in indices]

    def rank_cost(self, tracks, observations, base, frame, stage):
        if not self.config.get('separate_association_score', True) or not base.size or stage == 'secondary':
            return base
        if stage not in {'primary', 'pending'}:
            raise ValueError('Unknown original ByteTrack association stage')
        semantic = np.array([o.score for o in observations])
        physical = np.array([o.evidence['association_score'] for o in observations])
        changed = physical != semantic
        if not changed.any():
            return base
        overlap = 1-matching.iou_distance(tracks, observations)
        corrected = 1-overlap*physical[None]
        result = base.copy(); result[:, changed] = corrected[:, changed]
        limit = .9 if stage == 'primary' else .7
        self.statistics['physical_score_columns'] += int(changed.sum())
        self.statistics['score_opened_edges'] += int(((base > limit) & (result <= limit)).sum())
        self.statistics['score_closed_edges'] += int(((base <= limit) & (result > limit)).sum())
        return result
