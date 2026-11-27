"""Birth-only qualification with original scores and original association stages.

The readable BYTETracker.update port below differs from the frozen original only
at the Step4 birth check. The original file is never modified. Descriptors,
matching scores and template reliability always receive original detections.
"""
from types import SimpleNamespace
import numpy as np
from yolox.tracker.byte_tracker import (BYTETracker,STrack,TrackState,
    joint_stracks,sub_stracks,remove_duplicate_stracks)
from yolox.tracker import matching
from tools.mft_candidate_risk_teacher import CandidateTeacher,CandidateTrack
from tools.mft_proposal_quality import current_features
from tools.mft_proposal_quality_evaluation import SharedConfirmation,ShapeTeacher
from tools.mft_archive_evaluation import ArchiveTracker,policies as archive_policies

ORIGINAL_BYTE_TRACKER_SHA256='1a19bde221fc65bbca11d3c45cb2acbe571705510e3f94bfbaca95571d409f1a'


class BirthOnlyByteTracker(BYTETracker):
    def begin_frame(self,allow_birth):
        self.allow_birth=np.asarray(allow_birth,bool).copy()
        assert self.allow_birth.ndim==1
        self.last_birth_events=[]

    def birth_allowed(self,track):
        index=int(track.observation['proposal_index'])
        assert 0<=index<len(self.allow_birth)
        original=bool(track.score>=self.det_thresh)
        extra=bool(not original and self.allow_birth[index])
        self.last_birth_events.append(dict(proposal_index=index,score=float(track.score),
            original_allowed=original,extra_allowed=extra))
        return original or extra

    def update(self, output_results, img_info, img_size, evidence=None):
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
            detections = self._observations(dets, scores_keep, np.flatnonzero(remain_inds), evidence)
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
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        if self.association_hook is not None:
            dists = self.association_hook(strack_pool, detections, dists, self.frame_id, "primary")
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
            detections_second = self._observations(dets_second, scores_second, np.flatnonzero(inds_second), evidence)
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        if self.association_hook is not None:
            dists = self.association_hook(r_tracked_stracks, detections_second, dists, self.frame_id, "secondary")
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
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        if self.association_hook is not None:
            dists = self.association_hook(unconfirmed, detections, dists, self.frame_id, "pending")
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
            if not self.birth_allowed(track):
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

        return output_stracks


class BirthOnlyTeacher(CandidateTeacher):
    """Original measurements and templates; an optional mask affects births."""
    def __init__(self,speed_upper):
        super().__init__(speed_upper);self.qualification=None

    def reset(self,sequence):
        super().reset(sequence);counter=[0]
        self.core=BirthOnlyByteTracker(self.core.args,frame_rate=25,
            track_factory=lambda b,s,o:CandidateTrack(b,s,o,counter),association_hook=self.observe)
        self.statistics.update(birth_opportunities=0,extra_qualified_births=0)

    def update(self,sequence,frame,detections,rgb,ir=None):
        qualification=self.qualification
        if sequence!=self.sequence:
            assert frame==1;self.reset(sequence)
        d=np.asarray(detections,np.float32)
        allowed=np.zeros(len(d),bool)if qualification is None else np.asarray(qualification,bool)
        assert allowed.shape==(len(d),)
        assert not (allowed&((d[:,4]<=.6)|(d[:,4]>=.7))).any()
        self.core.begin_frame(allowed)
        try:result=super().update(sequence,frame,d,rgb,ir)
        finally:self.qualification=None
        events=self.core.last_birth_events
        self.statistics['birth_opportunities']+=sum(not e['original_allowed']for e in events)
        self.statistics['extra_qualified_births']+=sum(e['extra_allowed']for e in events)
        return result


def birth_mask(case,d,rgb,ir,views,heads,thresholds,streak):
    mode=case.get('intervention')
    if mode=='missing':ir=None
    elif mode=='roll':ir=np.roll(ir,1,axis=0).copy()if ir is not None else None
    else:assert mode is None
    kind=case['kind'];required=case['confirmation']
    assert required in [0,3]
    if kind=='unconditional':p=np.ones(len(d));threshold=0.
    else:
        assert kind in ['RGB','paired','duplicatedRGB']
        effective='RGB'if kind=='paired'and ir is None else kind
        p=heads[effective].probabilities(current_features(d,rgb,ir,views))
        condition=case.get('calibration_condition','confirmed'if required else'all')
        threshold=thresholds[effective][case['quantile']][condition]
    return (d[:,4]>.6)&(d[:,4]<.7)&(p>threshold)&(streak>=required)


def policies():
    cases=[dict(id='BO_native_zero',kind='reference_native'),
        dict(id='BO_archive_zero',kind='reference_archive')]
    for confirmation in [0,3]:
        cases.append(dict(id='BO_unconditional_c'+str(confirmation),kind='unconditional',
            confirmation=confirmation,quantile=None))
    for q in ['q99','q999']:
        for confirmation in [0,3]:
            for kind in ['RGB','paired','duplicatedRGB']:
                cases.append(dict(id='BO_'+kind+'_'+q+'_c'+str(confirmation),kind=kind,
                    quantile=q,confirmation=confirmation))
    return cases


class BirthOnlyQuality:
    def __init__(self,case,views,heads,thresholds,speed,archive_models,archive_thresholds,shared=None):
        self.case=dict(case);self.views,self.heads,self.thresholds=views,heads,thresholds
        self.confirmation=SharedConfirmation()if shared is None else shared
        self.teacher=BirthOnlyTeacher(speed)
        if case['kind']=='reference_native':self.parent=self.teacher
        elif case.get('parent')=='native_shape':
            self.parent=ShapeTeacher(speed);self.parent.teacher=self.teacher
        else:
            name='AR_RGB_q999'if case.get('parent')=='archive_raw'else'AR_RGB_q999_shape'
            c=next(c for c in archive_policies()if c['id']==name)
            self.parent=ArchiveTracker(c,archive_models,archive_thresholds,speed)
            self.parent.linker.inner=self.teacher
        self.sequence=None;self.qualified=0

    @property
    def statistics(self):
        return dict(self.parent.statistics,qualified_current_proposals=self.qualified,
            original_scores_preserved=True)

    def update(self,sequence,frame,detections,rgb,ir=None):
        if self.sequence!=sequence:self.sequence=sequence;self.qualified=0
        d=np.asarray(detections,np.float32)
        if self.case['kind'].startswith('reference_'):
            allowed=np.zeros(len(d),bool)
        else:
            streak=self.confirmation.update(sequence,frame,d)
            allowed=birth_mask(self.case,d,rgb,ir,self.views,self.heads,self.thresholds,streak)
        self.qualified+=int(allowed.sum());self.teacher.qualification=allowed
        return self.parent.update(sequence,frame,d,rgb,ir)
