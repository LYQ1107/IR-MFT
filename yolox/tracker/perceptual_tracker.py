"""Online evidence association with optional spatial, transport and memory cues.

This is an experimental implementation. Public tracker ideas and pretrained
RGB appearance are controls; contribution credit requires the registered
input and removal ablations. No future frame or annotation enters this API.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from yolox.tracker.kalman_filter import KalmanFilter
from yolox.tracker.matching import ious, linear_assignment
from yolox.tracker.perceptual_features import unit


def xyah(box):
    box = np.asarray(box, np.float64)
    wh = np.maximum(box[2:] - box[:2], 1.)
    return np.r_[(box[:2] + box[2:]) / 2, wh[0] / wh[1], wh[1]]


def box_from_mean(mean):
    height = max(float(mean[3]), 1.)
    width = max(float(mean[2]) * height, 1.)
    return np.r_[mean[:2] - [width / 2, height / 2], mean[:2] + [width / 2, height / 2]]


@dataclass
class ObservationTrack:
    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    last_frame: int
    born: int
    last_index: int
    confirmed: bool
    score: float
    embedding: np.ndarray
    shape: np.ndarray
    color: np.ndarray
    last_box: np.ndarray
    bank: list = field(default_factory=list)
    last_bank_frame: int = 0
    hits: int = 1


class PerceptualTracker:
    def __init__(self, config):
        self.config = dict(config)
        self.kf = KalmanFilter()
        self.tracks = []
        self.frame = 0
        self.next_id = 1
        self.statistics = dict(matches=0, protected_updates=0, flow_predictions=0,
                               ir_flow_accepted=0, reactivations=0, births=0,
                               supported_gap_outputs=0)

    def representations(self, data, indices):
        raw = data['embedding'][indices].astype(np.float32)
        view = self.config.get('appearance_view', 'raw')
        if view == 'raw':
            appearance = raw
        else:
            name = {'ir_mask': 'ir_mask_embedding', 'rgb_mask': 'rgb_mask_embedding',
                    'double_rgb': 'embedding'}[view]
            second = data[name][indices].astype(np.float32)
            alpha = float(self.config.get('mask_weight', .5))
            appearance = np.concatenate([raw * np.sqrt(1-alpha), second * np.sqrt(alpha)], axis=1)
        norm = np.maximum(np.linalg.norm(appearance, axis=1, keepdims=True), 1e-8)
        appearance = appearance / norm
        shape_name = 'ir_hog' if self.config.get('shape', 'rgb') == 'ir' else 'rgb_hog'
        shape = data[shape_name][indices].astype(np.float32)
        shape /= np.maximum(np.linalg.norm(shape, axis=1, keepdims=True), 1e-8)
        color = data['color'][indices].astype(np.float32)
        color /= np.maximum(np.linalg.norm(color, axis=1, keepdims=True), 1e-8)
        return appearance, shape, color

    def prediction(self, track, data):
        mean = track.mean.copy()
        if track.last_frame < self.frame - 1:
            mean[7] = 0
        track.mean, track.covariance = self.kf.predict(mean, track.covariance)
        predicted = box_from_mean(track.mean)
        mode = self.config.get('flow', 'none')
        if mode == 'none' or track.last_frame != self.frame - 1:
            return predicted
        rgb = data['flow_rgb'][track.last_index].astype(np.float32)
        aux = (data['flow_ir'][track.last_index].astype(np.float32)
               if mode in {'ir','consensus','robust_consensus'} else np.zeros(4,np.float32))
        minimum = self.config.get('flow_min_quality', .3)
        used_aux = False
        if mode == 'ir':
            if aux[2] < minimum:
                return predicted
            displacement, quality = aux[:2], float(aux[2])
            used_aux = True
        elif mode == 'robust_consensus':
            rgb_valid, aux_valid = rgb[2] >= minimum, aux[2] >= minimum
            size = np.maximum(track.last_box[2:] - track.last_box[:2], 1)
            disagreement = np.linalg.norm((aux[:2] - rgb[:2]) / size)
            if rgb_valid and aux_valid:
                if disagreement >= .15:
                    return predicted
                displacement = (rgb[:2]*rgb[2] + aux[:2]*aux[2])/(rgb[2]+aux[2])
                quality = min(1., (rgb[2]+aux[2])/1.5)
                used_aux = True
            elif aux_valid and aux[3] < 3:
                displacement, quality = aux[:2], float(aux[2])*.7
                used_aux = True
            elif rgb_valid:
                displacement, quality = rgb[:2], float(rgb[2])*.7
            else:
                return predicted
        else:
            if rgb[2] < minimum:
                return predicted
            displacement, quality = rgb[:2], float(rgb[2])
        if mode == 'consensus':
            size = np.maximum(track.last_box[2:] - track.last_box[:2], 1)
            disagreement = np.linalg.norm((aux[:2] - rgb[:2]) / size)
            if aux[2] >= .3 and disagreement < .15:
                displacement = (rgb[:2] * rgb[2] + aux[:2] * aux[2]) / (rgb[2] + aux[2])
                quality = min(1., (rgb[2] + aux[2]) / 1.5)
                used_aux = True
        if used_aux:
            self.statistics['ir_flow_accepted'] += 1
        transported = track.last_box + np.r_[displacement, displacement]
        weight = min(float(self.config.get('flow_weight', .9)), quality)
        self.statistics['flow_predictions'] += 1
        result = predicted * (1-weight) + transported * weight
        self.transport_support[track.track_id] = (result, quality, used_aux)
        return result

    def costs(self, tracks, predictions, data, indices, low=False):
        if not len(tracks) or not len(indices):
            return np.empty((len(tracks), len(indices)), np.float64)
        boxes = data['boxes'][indices]
        overlap = ious(np.asarray(predictions), boxes)
        scores = data['scores'][indices]
        motion = 1 - overlap * (1 if low else scores[None, :])
        appearance, shape, color = self.representations(data, indices)
        templates = np.stack([t.embedding for t in tracks])
        app_cost = np.clip((1 - templates @ appearance.T) / self.config.get('appearance_scale', .2), 0, 2)
        if self.config.get('protected_memory', False):
            for row, track in enumerate(tracks):
                if track.bank:
                    gallery = np.stack(track.bank)
                    gallery_cost = np.clip((1-gallery @ appearance.T) / self.config.get('appearance_scale', .2), 0, 2)
                    app_cost[row] = np.minimum(app_cost[row], gallery_cost.min(axis=0))
        shape_cost = np.clip((1-np.stack([t.shape for t in tracks]) @ shape.T) / .35, 0, 2)
        color_cost = np.clip((1-np.stack([t.color for t in tracks]) @ color.T) / .35, 0, 2)
        app_weight = float(self.config.get('appearance_weight', .35)) * (.5 if low else 1)
        shape_weight = float(self.config.get('shape_weight', 0))
        color_weight = float(self.config.get('color_weight', .05))
        motion_weight = max(0., 1-app_weight-shape_weight-color_weight)
        cost = motion_weight * motion + app_weight * app_cost + shape_weight * shape_cost + color_weight * color_cost
        ca = (np.asarray(predictions)[:, :2] + np.asarray(predictions)[:, 2:]) / 2
        cb = (boxes[:, :2] + boxes[:, 2:]) / 2
        size = (np.asarray(predictions)[:, None, 2:] - np.asarray(predictions)[:, None, :2] + boxes[None, :, 2:] - boxes[None, :, :2]) / 2
        center = np.linalg.norm((ca[:, None] - cb[None]) / np.maximum(size, 1), axis=-1)
        gaps = np.array([self.frame - t.last_frame for t in tracks])[:, None]
        maximum = self.config.get('maximum_center_distance', 1.5) + np.minimum(gaps * .025, .75)
        cost[(center > maximum) | ((overlap < .01) & (app_cost > 1.2))] = 1e4
        return cost

    def update_observation(self, track, index, data, cost, alternative):
        box = data['boxes'][index]
        appearance, shape, color = [x[0] for x in self.representations(data, np.array([index]))]
        gap = self.frame - track.last_frame
        if gap > 1:
            self.statistics['reactivations'] += 1
        track.mean, track.covariance = self.kf.update(track.mean, track.covariance, xyah(box))
        protected = self.config.get('protected_memory', False)
        other = data['boxes'][self.current_indices]
        overlaps = ious(box[None], other)[0]
        overlaps[self.current_indices == index] = 0
        ambiguous = (len(overlaps) and overlaps.max() > .3) or alternative-cost < .04 or data['scores'][index] < .7
        if protected and ambiguous:
            self.statistics['protected_updates'] += 1
        else:
            alpha = float(self.config.get('template_ema', .9))
            track.embedding = unit(alpha*track.embedding+(1-alpha)*appearance)
            track.shape = unit(alpha*track.shape+(1-alpha)*shape)
            track.color = unit(alpha*track.color+(1-alpha)*color)
            if protected and self.frame-track.last_bank_frame >= 10:
                if not track.bank or max(float(p@appearance) for p in track.bank) < .98:
                    track.bank.append(appearance.copy())
                    track.bank = track.bank[-3:]
                    track.last_bank_frame = self.frame
        track.last_frame, track.last_index = self.frame, int(index)
        track.last_box = box.copy()
        track.confirmed = True
        track.score = float(data['scores'][index])
        track.hits += 1
        self.statistics['matches'] += 1

    def update(self, frame, data):
        if frame != self.frame + 1:
            raise ValueError('Frames must be processed once, sequentially')
        self.frame = int(frame)
        self.transport_support = {}
        lo, hi = data['offsets'][frame-1:frame+1]
        indices = np.arange(lo, hi, dtype=int)
        self.current_indices = indices
        high = indices[data['scores'][indices] >= self.config.get('track_threshold', .6)]
        low = indices[(data['scores'][indices] >= .1) & (data['scores'][indices] < self.config.get('track_threshold', .6))]
        maximum_age = int(self.config.get('maximum_age', 30))
        self.tracks = [t for t in self.tracks if frame-t.last_frame <= maximum_age
                       and (t.confirmed or frame-t.last_frame <= 1)]
        predicted = {t.track_id:self.prediction(t,data) for t in self.tracks}
        active = [t for t in self.tracks if t.confirmed]
        pending = [t for t in self.tracks if not t.confirmed]
        used_tracks, used_detections = set(), set()

        def match(pool, observations, threshold, second=False):
            pool = [t for t in pool if t.track_id not in used_tracks]
            observations = np.array([int(i) for i in observations if int(i) not in used_detections], dtype=int)
            cost = self.costs(pool, [predicted[t.track_id] for t in pool], data, observations, second)
            pairs, _, _ = linear_assignment(cost, threshold)
            for a,b in pairs:
                alternatives = np.delete(cost[a], b)
                alternative = float(alternatives.min()) if len(alternatives) else float('inf')
                track, index = pool[a], int(observations[b])
                self.update_observation(track,index,data,float(cost[a,b]),alternative)
                used_tracks.add(track.track_id)
                used_detections.add(index)

        match(active, high, float(self.config.get('match_limit', .8)))
        match([t for t in active if t.last_frame==frame-1], low,
              float(self.config.get('low_match_limit', .55)), True)
        match(pending, high, float(self.config.get('pending_match_limit', .7)))
        self.tracks = [t for t in self.tracks if t.confirmed or t.track_id in used_tracks]
        for index in high:
            if int(index) in used_detections or data['scores'][index] < self.config.get('birth_threshold', .7):
                continue
            mean, covariance = self.kf.initiate(xyah(data['boxes'][index]))
            appearance, shape, color = [x[0] for x in self.representations(data, np.array([index]))]
            self.tracks.append(ObservationTrack(self.next_id, mean, covariance, frame, frame,
                                                int(index), frame==1, float(data['scores'][index]),
                                                appearance,shape,color,data['boxes'][index].copy(),
                                                [appearance.copy()],frame))
            self.next_id += 1
            self.statistics['births'] += 1
        output = []
        for track in self.tracks:
            if track.confirmed and track.last_frame == frame:
                box = box_from_mean(track.mean)
                tlwh = np.r_[box[:2],box[2:]-box[:2]]
                if tlwh[2]*tlwh[3] > 100:
                    output.append((frame,tlwh,track.track_id,track.score))
        for track in self.tracks:
            if (self.config.get('supported_gap_output', False) and track.confirmed
                  and frame-track.last_frame == 1
                  and track.track_id in self.transport_support):
                # A current-image transport measurement supports a single gap.
                # Do not update appearance, pretend a detector observation was
                # present, or revise the previous frame after a later recovery.
                box, quality, used_aux = self.transport_support[track.track_id]
                if quality < self.config.get('gap_minimum_quality', .65):
                    continue
                if self.config.get('gap_requires_aux', False) and not used_aux:
                    continue
                other = np.array([np.r_[r[1][:2],r[1][:2]+r[1][2:]] for r in output])
                high_boxes = data['boxes'][high]
                if (len(other) and ious(box[None],other).max()>.4) or (len(high_boxes) and ious(box[None],high_boxes).max()>.4):
                    continue
                tlwh=np.r_[box[:2],box[2:]-box[:2]]
                if tlwh[2]*tlwh[3]>100:
                    output.append((frame,tlwh,track.track_id,track.score*quality))
                    self.statistics['supported_gap_outputs'] += 1
        return output
