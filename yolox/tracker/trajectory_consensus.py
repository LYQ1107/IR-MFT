"""Conservative multi-anchor integration of unchanged fitted point scores."""
import numpy as np

from yolox.tracker.calibrated_foreground_tracker import CalibratedForegroundTracker


def anchor_consensus(probabilities, votes, minimum_anchors=2, fraction=2/3):
    probabilities = np.asarray(probabilities, dtype=float)
    votes = np.asarray(votes, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape != votes.shape:
        raise ValueError('One candidate-probability/vote row per causal anchor required')
    if len(probabilities) < minimum_anchors or not probabilities.shape[1]:
        return None
    if not np.isfinite(probabilities).all() or not np.isfinite(votes).all():
        raise ValueError('Finite current-candidate evidence required')
    combined = np.median(probabilities, axis=0)
    best = int(combined.argmax())
    agreeing = (probabilities.argmax(1) == best) & (votes[:, best] >= .15)
    if agreeing.sum() < minimum_anchors or agreeing.mean() < fraction:
        return None
    return combined, np.median(votes, axis=0), best, float(agreeing.mean())


class TrajectoryConsensusTracker(CalibratedForegroundTracker):
    def __init__(self, config, model=None, cue=None):
        super().__init__(config, model=model, cue=cue)
        self.statistics.update(multi_history_rows=0, consensus_rejections=0, cross_track_rejections=0)

    def rank_cost(self, tracks, observations, base, frame, stage):
        if not self.config.get('trajectory_consensus', True):
            return super().rank_cost(tracks, observations, base, frame, stage)
        if not base.size or stage not in self.config.get('point_stages', ['primary', 'secondary']):
            return base
        amount = float(self.config.get('association_weight', .8))
        if amount <= 0:
            return base
        limit = {'primary': .9, 'secondary': .5, 'pending': .7}[stage]
        columns = np.array([o.evidence['index']-self.current_start for o in observations], int)
        feasible = base <= limit; masked = np.where(feasible, base, np.inf)
        near_rows = feasible & (base-np.min(masked, axis=1, keepdims=True) < .12)
        near_cols = feasible & (base-np.min(masked, axis=0, keepdims=True) < .12)
        ambiguity = (near_rows & (near_rows.sum(1, keepdims=True) > 1)) | (near_cols & (near_cols.sum(0, keepdims=True) > 1))
        candidates = {}
        for row, track in enumerate(tracks):
            history = track.object_observations if self.config.get('object_files', True) else [track.evidence]
            probabilities = []; votes = []; increments = []
            for context in reversed(history):
                lag = frame-context['frame']
                if lag <= 0:
                    raise ValueError('Object-file anchors must precede the current frame')
                if lag not in self.config.get('lags', [1, 2, 4, 8]):
                    continue
                start, p, delta, vote, good, total = self.predict_pairs(frame, lag)
                index = context['index']-start
                if index < 0 or index >= len(p):
                    raise ValueError('Causal historical source index mismatch')
                if good[index] < 4:
                    continue
                probabilities.append(p[index, columns]); votes.append(vote[index, columns]); increments.append(delta[index, columns])
                if not self.config.get('history_consensus', True):
                    break
            minimum = 2 if self.config.get('history_consensus', True) else 1
            if not probabilities:
                continue
            result = anchor_consensus(probabilities, votes, minimum_anchors=minimum)
            if result is None:
                self.statistics['consensus_rejections'] += 1
                continue
            p, vote, best, fraction = result
            maximum = float(p[best]); second = float(np.partition(p, -2)[-2]) if len(p) > 1 else 0.
            confidence = maximum * min(1., max(0., maximum-second)/self.config.get('probability_margin', .15)) * fraction
            if maximum < self.config.get('minimum_probability', .8) or confidence < .3:
                continue
            candidates[row] = dict(p=p, vote=vote, best=best, maximum=maximum, confidence=confidence,
                anchors=len(probabilities), ir_changed=self.cue == 'ir' and abs(float(np.median(increments, axis=0)[best])) > 1e-4)
        result = base.copy()
        for row, evidence in candidates.items():
            p, best = evidence['p'], evidence['best']
            if self.config.get('mutual_competition', True):
                competing = [float(other['p'][best]) for other_row, other in candidates.items() if other_row != row]
                if competing and evidence['maximum']-max(competing) < self.config.get('probability_margin', .15):
                    self.statistics['cross_track_rejections'] += 1
                    continue
            if self.config.get('preserve_clear_motion', True) and not ambiguity[row].any():
                if feasible[row].any() or evidence['maximum'] < .95:
                    self.statistics['motion_preserved_rows'] += 1
                    continue
            scores = np.array([o.score for o in observations]) if stage != 'secondary' else np.ones(len(observations))
            weight = amount*evidence['confidence']
            adjusted = (1-weight)*base[row] + weight*(1-p*scores)
            rejected = base[row] > limit
            opening = (p >= self.config.get('opening_probability', .95)) & (evidence['vote'] >= .15)
            adjusted[rejected & ~opening] = base[row, rejected & ~opening]
            result[row] = adjusted
            self.trusted[(tracks[row].track_id, observations[best].evidence['index'])] = evidence['confidence']
            self.statistics['history_anchors'] += evidence['anchors']
            self.statistics['multi_history_rows'] += 1
            self.statistics['calibrated_rows'] += 1; self.statistics['point_rows'] += 1
            self.statistics['point_opened_edges'] += int((rejected & (adjusted <= limit)).sum())
            if evidence['ir_changed']:
                self.statistics['IR_increment_rows'] += 1; self.statistics['IR_applied_rows'] += 1; self.statistics['IR_supported_rows'] += 1
        return result
