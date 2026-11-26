"""Causal Fisher appearance association over the existing ByteTrack core.

No network, GT, image loader, future observations or output relabeling.
"""
from types import SimpleNamespace
import numpy as np
from .byte_tracker import BYTETracker, STrack
from . import matching


class FisherTrack(STrack):
    def __init__(self, tlwh, score, observation, counter, config):
        super().__init__(tlwh, score)
        self.observation, self.counter, self.config = observation, counter, config
        self.memory, self.memory_frame = None, None

    def next_id(self):
        self.counter[0] += 1
        return self.counter[0]

    def accept(self, observation, frame):
        if observation is None or not observation['reliable']:
            return
        value = observation['vector']
        if np.linalg.norm(value) < 1e-8:
            return
        updated = value.copy() if self.memory is None else (
            (1-self.config['update_alpha'])*self.memory+self.config['update_alpha']*value)
        norm = np.linalg.norm(updated)
        if norm > 1e-8:
            self.memory = (updated/norm).astype(np.float32)
            self.memory_frame = frame

    def activate(self, kalman_filter, frame_id):
        super().activate(kalman_filter, frame_id)
        self.accept(self.observation, frame_id)

    def update(self, new_track, frame_id):
        super().update(new_track, frame_id)
        self.accept(new_track.observation, frame_id)

    def re_activate(self, new_track, frame_id, new_id=False):
        super().re_activate(new_track, frame_id, new_id)
        self.accept(new_track.observation, frame_id)

    def mark_removed(self):
        super().mark_removed()
        self.memory, self.observation = None, None


class FisherIdentityTracker:
    def __init__(self, weight, lost_buffer, config):
        if weight not in [0., .25, .5] or lost_buffer not in [25, 250]:
            raise ValueError('Only the preregistered fixed weight/buffer cases')
        self.weight, self.lost_buffer, self.config = weight, lost_buffer, dict(config)
        self.sequence, self.core = None, None
        self.statistics = {}

    def reset(self, sequence):
        self.sequence = sequence
        counter = [0]
        # 25 fps and original track_buffer30 =>25-frame original maximum age.
        args = SimpleNamespace(track_thresh=.6, track_buffer=self.lost_buffer*30//25,
                               match_thresh=.9, mot20=False)
        self.core = BYTETracker(args, frame_rate=25,
            track_factory=lambda box, score, obs: FisherTrack(box, score, obs, counter, self.config),
            association_hook=self.cost if self.weight else None)
        assert self.core.max_time_lost == self.lost_buffer
        self.statistics = dict(primary_calls=0, actual_float_cost_entries_changed=0,
                               frames_with_changed_primary_assignment=0, remembered_candidate_pairs=0)

    def cost(self, tracks, detections, base, frame, stage):
        if stage != 'primary' or not base.size:
            return base
        result = base.copy()
        self.statistics['primary_calls'] += 1
        ious = 1-matching.iou_distance(tracks, detections)
        for i, track in enumerate(tracks):
            if track.memory is None:
                continue
            assert track.memory_frame < frame, 'Current/future features cannot be past memory'
            old_box = track.tlbr
            old_center = (old_box[:2]+old_box[2:])/2
            scale = np.maximum(old_box[2:]-old_box[:2], 1.)
            for j, detection in enumerate(detections):
                obs = detection.observation
                if obs is None or np.linalg.norm(obs['vector']) < 1e-8:
                    continue
                box = detection.tlbr
                center = (box[:2]+box[2:])/2
                distance = float(np.linalg.norm((center-old_center)/scale))
                if (ious[i, j] < self.config['minimum_geometry_IoU'] and
                        distance > self.config['maximum_normalized_center_distance']):
                    continue
                appearance = float(np.clip((1-float(track.memory@obs['vector']))/2, 0, 1))
                result[i, j] = (1-self.weight)*base[i, j]+self.weight*appearance
                self.statistics['remembered_candidate_pairs'] += 1
        self.statistics['actual_float_cost_entries_changed'] += int(np.count_nonzero(result != base))
        original = matching.linear_assignment(base, thresh=self.core.args.match_thresh)[0]
        changed = matching.linear_assignment(result, thresh=self.core.args.match_thresh)[0]
        self.statistics['frames_with_changed_primary_assignment'] += int(
            set(map(tuple, original)) != set(map(tuple, changed)))
        return result

    def update(self, sequence, frame, detections, descriptors=None):
        if not sequence or not isinstance(frame, int) or frame < 1:
            raise ValueError('Explicit positive current sequence/frame required')
        if self.sequence != sequence:
            if frame != 1:
                raise ValueError('A new sequence must start at1')
            self.reset(sequence)
        elif frame != self.core.frame_id+1:
            raise ValueError('Deliver every consecutive frame exactly once')
        values = np.asarray(detections, dtype=np.float32)
        if (values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all()
                or ((values[:, 4] < 0)|(values[:, 4] > 1)).any()
                or (values[:, 2:4] <= values[:, :2]).any()):
            raise ValueError('Finite source-coordinate boxes and probabilities required')
        evidence = None
        if self.weight:
            vectors = np.asarray(descriptors, dtype=np.float32)
            if vectors.ndim != 2 or len(vectors) != len(values) or not np.isfinite(vectors).all():
                raise ValueError('Current learned descriptors must align with all current proposals')
            overlap = matching.ious(values[:, :4], values[:, :4]) if len(values) else np.empty((0, 0))
            np.fill_diagonal(overlap, 0)
            maximum = overlap.max(axis=1) if len(values) else np.empty(0)
            evidence = [dict(vector=vector.copy(), reliable=bool(
                values[i, 4] > self.config['minimum_update_score'] and
                maximum[i] <= self.config['maximum_update_overlap'])) for i, vector in enumerate(vectors)]
        tracks = self.core.update(values.copy(), (1, 1), (1, 1), evidence=evidence)
        # Evidence is consumed on update; keep only a bounded vector per live ID.
        for group in [self.core.tracked_stracks, self.core.lost_stracks, self.core.removed_stracks]:
            for track in group:
                track.observation = None
        return tracks
