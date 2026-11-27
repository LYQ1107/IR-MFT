"""Coverage-triggered recall with original scores and ordered associations.

Readable extension of the frozen birth-only port. Original active-track primary,
secondary and high-score pending stages precede weak pending and weak birth.
Extra tracks can change future candidate graphs; zero recall must match the
frozen birth-only parent exactly. No GT or sequence-specific rule enters here.
"""
import numpy as np
from yolox.tracker.byte_tracker import STrack,TrackState,joint_stracks,sub_stracks,remove_duplicate_stracks
from yolox.tracker import matching
from tools.mft_birth_only import BirthOnlyByteTracker,BirthOnlyQuality,BirthOnlyTeacher
from tools.mft_candidate_risk_teacher import CandidateTeacher,CandidateTrack
from tools.mft_proposal_quality import ProposalConfirmation
from tools.mft_scene_coverage import coverage_features
from tools.mft_proposal_quality_evaluation import ShapeTeacher


class CoverageRecallCore(BirthOnlyByteTracker):
    def begin_frame(self,allow_birth,recall_mask):
        super().begin_frame(allow_birth)
        self.recall_mask=np.asarray(recall_mask,bool).copy()
        assert self.recall_mask.shape==self.allow_birth.shape

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
        # Confirm recall-born pending tracks using only low detections left
        # after all original active-track secondary assignments. Strong pending
        # tracks retain the original confirmation rule.
        unused_low = [detections_second[i] for i in u_detection_second]
        weak_pending = [unconfirmed[i] for i in u_unconfirmed
                        if getattr(unconfirmed[i], "coverage_weak_birth", False)]
        weak_cost = matching.iou_distance(weak_pending, unused_low)
        weak_matches, _, weak_unused = matching.linear_assignment(weak_cost, thresh=0.5)
        weak_confirmed = set()
        for it, jd in weak_matches:
            track = weak_pending[it]
            track.update(unused_low[jd], self.frame_id)
            activated_starcks.append(track)
            weak_confirmed.add(id(track))
            self.weak_pending_confirmations += 1
        for it in u_unconfirmed:
            track = unconfirmed[it]
            if id(track) not in weak_confirmed:
                track.mark_removed()
                removed_stracks.append(track)
        unused_low = [unused_low[i] for i in weak_unused]

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if not self.birth_allowed(track):
                continue
            index = int(track.observation['proposal_index'])
            track.coverage_weak_birth = bool(self.recall_mask[index] and track.score < self.det_thresh)
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        # Qualified weak births cannot consume a proposal already used by any
        # original association or the recall-only pending confirmation stage.
        for track in unused_low:
            index = int(track.observation['proposal_index'])
            if not self.recall_mask[index]:
                continue
            assert 0.3 < track.score < 0.6
            assert self.birth_allowed(track)
            track.coverage_weak_birth = True
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
            self.low_score_births += 1
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


class CoverageRecallTeacher(CandidateTeacher):
    def __init__(self,speed):
        super().__init__(speed);self.qualification=None;self.recall_qualification=None

    def reset(self,sequence):
        super().reset(sequence);counter=[0]
        self.core=CoverageRecallCore(self.core.args,frame_rate=25,
            track_factory=lambda b,s,o:CandidateTrack(b,s,o,counter),association_hook=self.observe)
        self.core.low_score_births=0;self.core.weak_pending_confirmations=0
        self.statistics.update(birth_opportunities=0,extra_qualified_births=0)

    def update(self,sequence,frame,detections,rgb,ir=None):
        base=self.qualification;recall=self.recall_qualification
        if sequence!=self.sequence:
            assert frame==1;self.reset(sequence)
        d=np.asarray(detections,np.float32)
        a=np.zeros(len(d),bool)if base is None else np.asarray(base,bool)
        b=np.zeros(len(d),bool)if recall is None else np.asarray(recall,bool)
        assert a.shape==b.shape==(len(d),)
        assert not (a&((d[:,4]<=.6)|(d[:,4]>=.7))).any()
        assert not (b&((d[:,4]<=.3)|(d[:,4]>=.7))).any()
        self.core.begin_frame(a|b,b)
        try:rows=super().update(sequence,frame,d,rgb,ir)
        finally:self.qualification=None;self.recall_qualification=None
        events=self.core.last_birth_events
        self.statistics['birth_opportunities']+=sum(not e['original_allowed']for e in events)
        self.statistics['extra_qualified_births']+=sum(e['extra_allowed']for e in events)
        self.statistics['low_score_births']=self.core.low_score_births
        self.statistics['weak_pending_confirmations']=self.core.weak_pending_confirmations
        return rows


class SharedCoverage:
    """Immutable original-teacher and shadow stream, shared by fixed cases."""
    def __init__(self,speed,views,heads):
        self.teacher=BirthOnlyTeacher(speed);self.shadow=ProposalConfirmation()
        self.views,self.heads=views,heads;self.sequence=None;self.frame=0

    def update(self,sequence,frame,d,rgb,ir):
        if sequence==self.sequence and frame==self.frame:
            for a,b in zip([d,rgb,ir],self.packet):
                if a is None or b is None:assert a is b
                else:np.testing.assert_array_equal(a,b)
            return self.streak.copy(),self.values
        self.streak=self.shadow.update(sequence,frame,d)
        self.teacher.update(sequence,frame,d,rgb,ir)
        unused=[e['proposal_index']for e in self.teacher.core.last_birth_events if not e['original_allowed']]
        assigned=list(self.teacher.current_proposal_assignments(frame).values())
        _,self.values=coverage_features(d,rgb,ir,self.views,self.heads,self.streak,unused,assigned)
        self.sequence,self.frame=sequence,frame
        self.packet=[None if a is None else np.asarray(a).copy()for a in [d,rgb,ir]]
        return self.streak.copy(),self.values


def policies():
    cases=[dict(id='CR_reference',kind='reference',threshold=None),
        dict(id='CR_unconditional',kind='unconditional',threshold=None)]
    for threshold in [.1,.25]:
        for kind in ['count','RGB','paired','duplicatedRGB']:
            cases.append(dict(id='CR_'+kind+'_t'+str(threshold).replace('.',''),kind=kind,threshold=threshold))
    return cases


class CoverageRecallTracker:
    """New recall controller over either a standalone or frozen full parent.

    Standalone validation does not use any late-Train-calibrated quality or
    archive threshold. Full Dev uses the frozen selected BO parent. Its earlier
    calibration exposure must not be described as held-out full-system Train.
    ROI gate interventions affect only the new controller, not the IR visual
    detector, original parent quality module, or parent's archive module.
    """
    def __init__(self,case,views,heads,thresholds,speed,archive_models=None,archive_thresholds=None,
                 *,standalone=False,shared=None):
        self.case=dict(case);self.sequence=None;self.gated_frames=0;self.qualified=0
        self.shared=SharedCoverage(speed,views,heads)if shared is None else shared
        self.teacher=CoverageRecallTeacher(speed)
        self.standalone=standalone
        if standalone:
            self.parent=ShapeTeacher(speed);self.parent.teacher=self.teacher
        else:
            c=dict(id='BO_paired_q999_c3',kind='paired',quantile='q999',confirmation=3)
            self.parent=BirthOnlyQuality(c,views,heads,thresholds,speed,archive_models,archive_thresholds)
            self.parent.teacher=self.teacher;self.parent.parent.linker.inner=self.teacher

    @property
    def statistics(self):
        return dict(self.parent.statistics,coverage_gated_frames=self.gated_frames,
            coverage_qualified_proposals=self.qualified,standalone=self.standalone,
            original_scores_preserved=True)

    def update(self,sequence,frame,d,rgb,ir=None):
        if sequence!=self.sequence:self.sequence=sequence;self.gated_frames=0;self.qualified=0
        streak,values=self.shared.update(sequence,frame,d,rgb,ir)
        kind=self.case['kind']
        if kind=='reference':gate=False
        elif kind=='unconditional':gate=True
        else:
            assert kind in ['count','RGB','paired','duplicatedRGB']
            mode=self.case.get('intervention')
            effective='IR_roll'if mode=='roll'else'IR_missing'if mode=='missing'else kind
            assert mode in [None,'roll','missing']
            gate=values[effective]['deficit']>=self.case['threshold']
        mask=(d[:,4]>.3)&(d[:,4]<.7)&(streak>=3)&gate
        self.teacher.recall_qualification=mask;self.gated_frames+=int(gate);self.qualified+=int(mask.sum())
        return self.parent.update(sequence,frame,d,rgb,ir)
