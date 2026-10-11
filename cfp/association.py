"""Appearance baselines with explicit score-before-write identity state.

The enabled lifecycle adapts the repository's MIT-licensed BYTETracker update;
the disabled path directly calls the unchanged original implementation.
Original copyright and permission notice are in the repository LICENSE:
Copyright (c) 2021 Yifu Zhang. SPDX-License-Identifier: MIT.
"""
from collections import deque

import numpy as np

from yolox.tracker.byte_tracker import BYTETracker, STrack, joint_stracks, sub_stracks, remove_duplicate_stracks
from yolox.tracker.basetrack import TrackState
from yolox.tracker import matching


def normalize(value):
    value = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(value)
    if not np.isfinite(value).all() or not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError('Invalid identity embedding')
    return value / norm


def optional_feature(value):
    if value is None:
        return None
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite appearance feature')
    # Zero marks an explicitly recorded unextractable crop. Keep detections and
    # fall back to geometry instead of dropping them from the fixed proposal set.
    return None if np.linalg.norm(value) <= 1e-8 else normalize(value)


class IdentityTrack(STrack):
    def __init__(self, tlwh, score, rgb, ir, proposal_index=None):
        super().__init__(tlwh, score)
        self.current_rgb = optional_feature(rgb)
        self.current_ir = optional_feature(ir)
        self.memory_rgb, self.memory_ir = None, None
        self.pending = True
        self.last_identity_frame = 0
        self.identity_history = deque(maxlen=8)
        self.current_proposal_index = proposal_index
        self.identity_proposals = deque(maxlen=8)

    def update(self, new_track, frame_id):
        super().update(new_track, frame_id)
        self.current_rgb, self.current_ir = new_track.current_rgb, new_track.current_ir
        self.current_proposal_index = new_track.current_proposal_index
        self.pending = True

    def re_activate(self, new_track, frame_id, new_id=False):
        super().re_activate(new_track, frame_id, new_id)
        self.current_rgb, self.current_ir = new_track.current_rgb, new_track.current_ir
        self.current_proposal_index = new_track.current_proposal_index
        self.pending = True

    def write_identity(self, frame_id, ema):
        if not self.pending:
            return False
        if frame_id <= self.last_identity_frame:
            raise ValueError('Identity writes must follow scoring once per frame')
        if self.current_rgb is None and self.current_ir is None:
            self.pending = False
            return False
        if self.current_rgb is not None:
            self.memory_rgb = self.current_rgb.copy() if self.memory_rgb is None else normalize(ema * self.memory_rgb + (1 - ema) * self.current_rgb)
        if self.current_ir is not None:
            self.memory_ir = self.current_ir.copy() if self.memory_ir is None else normalize(ema * self.memory_ir + (1 - ema) * self.current_ir)
        if self.current_rgb is not None:
            self.identity_history.append((int(frame_id), self.current_rgb.copy(), None if self.current_ir is None else self.current_ir.copy()))
            self.identity_proposals.append((int(frame_id), self.current_proposal_index))
        self.last_identity_frame, self.pending = int(frame_id), False
        return True


class AppearanceTracker(BYTETracker):
    def __init__(self, args, frame_rate=25, mode='rgb', appearance_weight=.3,
                 ir_weight=.5, ema=.9, proximity_iou=.1, appearance_max_distance=.35):
        super().__init__(args, frame_rate)
        if mode not in ('off', 'rgb', 'ir', 'fixed'):
            raise ValueError('Undeclared appearance baseline')
        if not all(0 <= value <= 1 for value in [appearance_weight, ir_weight, ema, proximity_iou, appearance_max_distance]):
            raise ValueError('Invalid appearance baseline parameters')
        self.mode = mode
        self.appearance_weight, self.ir_weight, self.ema = appearance_weight, ir_weight, ema
        self.proximity_iou, self.appearance_max_distance = proximity_iou, appearance_max_distance
        self.decision_counts = dict(scored_rows=0, eligible_edges=0, identity_writes=0)

    def association_cost(self, tracks, detections):
        raw_geometry = matching.iou_distance(tracks, detections)
        geometry = matching.fuse_score(raw_geometry, detections) if not self.args.mot20 else raw_geometry
        if geometry.size == 0 or self.appearance_weight == 0:
            return geometry
        result = geometry.copy()
        for i, track in enumerate(tracks):
            if track.memory_rgb is None:
                continue
            if track.last_identity_frame >= self.frame_id:
                raise ValueError('Current observation leaked into association memory')
            valid_rgb = np.array([det.current_rgb is not None for det in detections])
            rgb_cost = np.array([float(np.clip((1 - track.memory_rgb @ det.current_rgb) / 2, 0, 1)) if det.current_rgb is not None else 1. for det in detections])
            appearance = rgb_cost.copy()
            if self.mode in ('fixed', 'ir') and track.memory_ir is not None:
                for j, detection in enumerate(detections):
                    if detection.current_ir is None:
                        continue
                    ir_cost = float(np.clip((1 - track.memory_ir @ detection.current_ir) / 2, 0, 1))
                    appearance[j] = ir_cost if self.mode == 'ir' else (1 - self.ir_weight) * rgb_cost[j] + self.ir_weight * ir_cost
            eligible = valid_rgb & (raw_geometry[i] <= 1 - self.proximity_iou) & (appearance <= self.appearance_max_distance)
            result[i, eligible] = (1 - self.appearance_weight) * geometry[i, eligible] + self.appearance_weight * appearance[eligible]
            self.decision_counts['scored_rows'] += 1
            self.decision_counts['eligible_edges'] += int(eligible.sum())
        return result

    def update(self, output_results, img_info, img_size, rgb_features=None, ir_features=None, proposal_indices=None):
        if self.mode == 'off':
            return super().update(output_results, img_info, img_size)
        if output_results.ndim != 2 or output_results.shape[1] != 5:
            raise ValueError('Enabled Protocol A requires original-coordinate Nx5 detections')
        if rgb_features is None or rgb_features.ndim != 2 or len(rgb_features) != len(output_results):
            raise ValueError('RGB feature/detection alignment required')
        if ir_features is not None and (ir_features.shape != rgb_features.shape):
            raise ValueError('Paired appearance shapes disagree')
        indices = np.arange(len(output_results), dtype=np.int64) if proposal_indices is None else np.asarray(proposal_indices)
        if indices.shape != (len(output_results),) or not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0) or len(np.unique(indices)) != len(indices):
            raise ValueError('Proposal indices must be unique nonnegative integers for the current frame')
        self._current_proposal_indices = indices
        return self._appearance_update(output_results, img_info, img_size, rgb_features, ir_features)

    # The lifecycle method below is adapted from the unchanged repository
    # BYTETracker.update, preserving thresholds, confidence stages and geometry.
    def _appearance_update(self, output_results, img_info, img_size, rgb_features, ir_features):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [IdentityTrack(STrack.tlbr_to_tlwh(tlbr), s, rgb, ir, int(index)) for
                          (tlbr, s, rgb, ir, index) in zip(dets, scores_keep, rgb_features[remain_inds],
                          [None] * len(dets) if ir_features is None else ir_features[remain_inds], self._current_proposal_indices[remain_inds])]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = self.association_cost(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [IdentityTrack(STrack.tlbr_to_tlwh(tlbr), s, rgb, ir, int(index)) for
                          (tlbr, s, rgb, ir, index) in zip(dets_second, scores_second, rgb_features[inds_second],
                          [None] * len(dets_second) if ir_features is None else ir_features[inds_second], self._current_proposal_indices[inds_second])]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = self.association_cost(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        for track in self.tracked_stracks:
            if track.pending:
                if track.write_identity(self.frame_id, self.ema):
                    self.decision_counts['identity_writes'] += 1
        for track in removed_stracks:
            # Original BYTETracker retains removed object IDs for its lifecycle.
            # Retire the larger identity state once reactivation is impossible.
            track.memory_rgb = track.memory_ir = None
            track.current_rgb = track.current_ir = None
            track.identity_history.clear()
            track.identity_proposals.clear()
        return output_stracks
