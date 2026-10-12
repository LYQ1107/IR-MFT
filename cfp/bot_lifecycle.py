"""Explicit BoT lifecycle adaptation with independent association policy.

Pinned original: BoT-SORT 251985436d6712aaf682aaaf5f71edb4987224bd.
Changes: track type/evidence adapter, two first/unconfirmed cost hooks, and
post-stage identity bookkeeping. KF, GMC identity warp, stage sets, lifecycle,
low-confidence IoU, duplicate suppression and output policy remain upstream.

MIT License

Copyright (c) 2022 Nir Aharon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import numpy as np


def bot_lifecycle_update(self, output_results, img):
    self.frame_id += 1
    activated_starcks = []
    refind_stracks = []
    lost_stracks = []
    removed_stracks = []

    if len(output_results):
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            classes = output_results[:, -1]
        else:
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
            classes = output_results[:, -1]

        # Remove bad detections
        lowest_inds = scores > self.track_low_thresh
        bboxes = bboxes[lowest_inds]
        scores = scores[lowest_inds]
        classes = classes[lowest_inds]

        # Find high threshold detections
        remain_inds = scores > self.args.track_high_thresh
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        classes_keep = classes[remain_inds]

    else:
        bboxes = []
        scores = []
        classes = []
        dets = []
        scores_keep = []
        classes_keep = []

    '''Extract embeddings '''
    if self.args.with_reid:
        features_keep = self.encoder.inference(img, dets)

    if len(dets) > 0:
        '''Detections'''
        if self.args.with_reid:
            detections = [self.track_type(self.track_type.tlbr_to_tlwh(tlbr), s, f) for
                          (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
        else:
            detections = [self.track_type(self.track_type.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
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
    strack_pool = self.native_helpers.joint_stracks(tracked_stracks, self.lost_stracks)

    # Predict the current location with KF
    self.track_type.multi_predict(strack_pool)

    # Fix camera motion
    warp = self.gmc.apply(img, dets)
    self.track_type.multi_gmc(strack_pool, warp)
    self.track_type.multi_gmc(unconfirmed, warp)

    dists = self.association_cost(strack_pool, detections)

    matches, u_track, u_detection = self.native_matching.linear_assignment(dists, thresh=self.args.match_thresh)

    for itracked, idet in matches:
        track = strack_pool[itracked]
        det = detections[idet]
        if track.state == self.track_states.Tracked:
            track.update(detections[idet], self.frame_id)
            activated_starcks.append(track)
        else:
            track.re_activate(det, self.frame_id, new_id=False)
            refind_stracks.append(track)

    ''' Step 3: Second association, with low score detection boxes'''
    if len(scores):
        inds_high = scores < self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        scores_second = scores[inds_second]
        classes_second = classes[inds_second]
    else:
        dets_second = []
        scores_second = []
        classes_second = []

    # association the untrack to the low score detections
    if len(dets_second) > 0:
        '''Detections'''
        detections_second = [self.track_type(self.track_type.tlbr_to_tlwh(tlbr), s) for
                             (tlbr, s) in zip(dets_second, scores_second)]
    else:
        detections_second = []

    r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == self.track_states.Tracked]
    dists = self.native_matching.iou_distance(r_tracked_stracks, detections_second)
    matches, u_track, u_detection_second = self.native_matching.linear_assignment(dists, thresh=0.5)
    for itracked, idet in matches:
        track = r_tracked_stracks[itracked]
        det = detections_second[idet]
        if track.state == self.track_states.Tracked:
            track.update(det, self.frame_id)
            activated_starcks.append(track)
        else:
            track.re_activate(det, self.frame_id, new_id=False)
            refind_stracks.append(track)

    for it in u_track:
        track = r_tracked_stracks[it]
        if not track.state == self.track_states.Lost:
            track.mark_lost()
            lost_stracks.append(track)

    '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
    detections = [detections[i] for i in u_detection]
    dists = self.association_cost(unconfirmed, detections)

    matches, u_unconfirmed, u_detection = self.native_matching.linear_assignment(dists, thresh=0.7)
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
        if track.score < self.new_track_thresh:
            continue

        track.activate(self.kalman_filter, self.frame_id)
        activated_starcks.append(track)

    """ Step 5: Update state"""
    for track in self.lost_stracks:
        if self.frame_id - track.end_frame > self.max_time_lost:
            track.mark_removed()
            removed_stracks.append(track)

    """ Merge """
    self.tracked_stracks = [t for t in self.tracked_stracks if t.state == self.track_states.Tracked]
    self.tracked_stracks = self.native_helpers.joint_stracks(self.tracked_stracks, activated_starcks)
    self.tracked_stracks = self.native_helpers.joint_stracks(self.tracked_stracks, refind_stracks)
    self.lost_stracks = self.native_helpers.sub_stracks(self.lost_stracks, self.tracked_stracks)
    self.lost_stracks.extend(lost_stracks)
    self.lost_stracks = self.native_helpers.sub_stracks(self.lost_stracks, self.removed_stracks)
    self.removed_stracks.extend(removed_stracks)
    self.tracked_stracks, self.lost_stracks = self.native_helpers.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

    # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
    output_stracks = [track for track in self.tracked_stracks]


    self.finalize_identity()
    return output_stracks
