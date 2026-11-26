"""Causal re-identification of unmatched lost tracks without stealing matches.

Unlike bounded geometric cost penalties, this explicitly admits a new edge
when last-observation reachability and unique identity evidence agree. Every
original primary assignment is retained. The original lost lifetime stays25.
"""
import numpy as np
from tools.mft_ambiguity_reasoning import AmbiguityReasoning, policies as previous_policies, unique_preferences
from yolox.tracker import matching
from yolox.tracker.basetrack import TrackState


def policies():
    return [dict(id='OR_native_zero', rescue=False, late=False, cue='none'),
            dict(id='OR_late_only', rescue=False, late=True, cue='none'),
            dict(id='OR_RGB_rescue', rescue=True, late=False, cue='none'),
            dict(id='OR_RGB_IR_rescue', rescue=True, late=False, cue='ir'),
            dict(id='OR_RGB_rescue_late', rescue=True, late=True, cue='none'),
            dict(id='OR_RGB_IR_rescue_late', rescue=True, late=True, cue='ir'),
            dict(id='OR_IR_descriptor_roll', rescue=True, late=True, cue='ir_roll'),
            dict(id='OR_IR_descriptor_unavailable', rescue=True, late=True, cue='missing')]


class OcclusionRescue(AmbiguityReasoning):
    def __init__(self, case, thresholds):
        assert case in policies()
        self.rescue_case = dict(case)
        index = 0 if not case['rescue'] and not case['late'] else 1
        if case['rescue']:
            index = {'none':4, 'ir':5, 'ir_roll':7, 'missing':8}[case['cue']]
        super().__init__(previous_policies()[index], thresholds)

    def reset(self, sequence):
        super().reset(sequence)
        self.core.association_hook = self.rescue_cost if self.rescue_case['rescue'] else None
        self.statistics.update(rescue_calls=0, geometrically_reachable_pairs=0,
            RGB_unique_lost_pairs=0, IR_confirmed_lost_pairs=0, new_primary_rescue_pairs=0,
            rescue_outside_original_geometric_graph=0, rescued_frames=0,
            original_primary_matches_preserved=0)

    def link(self, new, frame, active_roots):
        return super().link(new, frame, active_roots) if self.rescue_case['late'] else {}

    def rescue_cost(self, tracks, detections, base, frame, stage):
        if stage != 'primary' or not base.size:
            return base
        self.statistics['rescue_calls'] += 1
        original = matching.linear_assignment(base, thresh=self.core.args.match_thresh)[0]
        used_rows = {int(i) for i, j in original}
        used_cols = {int(j) for i, j in original}
        scores = np.full(base.shape, -np.inf, np.float64)
        ir = np.full(base.shape, -np.inf, np.float64)
        for i, track in enumerate(tracks):
            if i in used_rows or track.state != TrackState.Lost:
                continue
            gap = frame - track.frame_id
            if not 2 <= gap <= self.core.max_time_lost or track.accepted < 3 or track.rgb is None:
                continue
            assert track.memory_frame is not None and track.memory_frame < frame
            assert track.last_box is not None
            a = np.maximum(track.last_box[2:] - track.last_box[:2], 1.)
            old_center = (track.last_box[:2] + track.last_box[2:]) / 2
            for j, detection in enumerate(detections):
                obs = detection.observation
                if j in used_cols or obs is None or not obs['reliable']:
                    continue
                box = obs['box']; b = np.maximum(box[2:] - box[:2], 1.)
                scale = b / a
                if (scale < .5).any() or (scale > 2.).any():
                    continue
                distance = np.linalg.norm((box[:2] + box[2:]) / 2 - old_center) / np.linalg.norm(a)
                if distance > .5 + self.thresholds['Train_unknown_speed_upper'] * gap:
                    continue
                self.statistics['geometrically_reachable_pairs'] += 1
                scores[i,j] = float(track.rgb @ obs['rgb'])
                if obs['auxiliary_available'] and track.ir is not None:
                    ir[i,j] = float(track.ir @ obs['ir'])
        preferred = unique_preferences(scores, self.thresholds['primary_RGB_cosine_threshold'])
        self.statistics['RGB_unique_lost_pairs'] += len(preferred)
        if self.rescue_case['cue'] in ['ir', 'ir_roll']:
            # Confirmation never erases lower-scoring RGB competitors from the margin.
            preferred = {i:j for i,j in preferred.items()
                         if ir[i,j] >= self.thresholds['primary_IR_cosine_threshold']}
            self.statistics['IR_confirmed_lost_pairs'] += len(preferred)
        if not preferred:
            return base
        result = base.copy()
        required = [tuple(map(int, pair)) for pair in original] + list(preferred.items())
        for i,j in required:
            result[i,:] = np.inf
            result[:,j] = np.inf
        for i,j in original:
            result[i,j] = base[i,j]
        for i,j in preferred.items():
            result[i,j] = 0.
        after = matching.linear_assignment(result, thresh=self.core.args.match_thresh)[0]
        assert set(required).issubset(set(map(tuple, after)))
        self.statistics['original_primary_matches_preserved'] += len(original)
        self.statistics['new_primary_rescue_pairs'] += len(preferred)
        self.statistics['rescue_outside_original_geometric_graph'] += sum(int(base[i,j] > self.core.args.match_thresh) for i,j in preferred.items())
        self.statistics['rescued_frames'] += 1
        self.statistics['primary_assignment_changed_frames'] += 1
        self.statistics['primary_association_costs_changed'] += int(np.count_nonzero(result != base))
        return result
