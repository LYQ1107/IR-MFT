"""Promote a low-score candidate only after causal foreground correspondence.

The original high-score birth rule remains intact. The added route admits
unclaimed, repeated observations rather than changing every detection score.
It creates a normal unconfirmed track, without emitting a fabricated box.
"""
import numpy as np

from yolox.tracker.foreground_byte_tracker import ForegroundByteTracker,ObjectFileTrack


class CandidateForegroundTracker(ForegroundByteTracker):
    def __init__(self,config):
        super().__init__(config)
        self.created=[]
        self.inner.track_factory=self.make_observation
        self.statistics.update(candidate_births=0,candidate_evidence=0,IR_supported_births=0)
        if 'global_birth_threshold' in config:
            self.inner.det_thresh=float(config['global_birth_threshold'])

    def make_observation(self,box,score,context):
        observation=ObjectFileTrack(box,score,context)
        self.created.append(observation)
        return observation

    def update(self,frame,data):
        self.created=[]
        claimed_before={t.evidence['index'] for t in self.inner.tracked_stracks+self.inner.lost_stracks}
        rows=super().update(frame,data)
        if not self.config.get('candidate_birth',False) or frame<=1:return rows
        transport=data['foreground_transport']
        source=self.config.get('birth_source_mode',self.config.get('source_mode','joint'))
        target=self.config.get('birth_target_mode',self.config.get('target_mode',source))
        a,b=map(int,data['offsets'][frame-1:frame+1]);c,d=map(int,data['offsets'][frame-2:frame])
        v0,v1=map(int,transport['vote_offsets_1'][frame-1:frame+1]);p0,p1=map(int,transport['count_offsets_1'][frame-1:frame+1])
        matrix=transport[f'vote_1_{source}_{target}'][v0:v1].reshape(d-c,b-a).astype(np.float64)
        if not matrix.size:return rows
        good=transport[f'good_1_{source}'][p0:p1];total=transport[f'total_1_{source}'][p0:p1]
        used={t.evidence['index'] for t in self.inner.tracked_stracks if t.frame_id==frame}
        quality_index=1 if source in {'ir','joint'} or target in {'ir','joint'} else 0
        for observation in self.created:
            index=observation.evidence['index'];column=index-a
            if index in used or observation.score>=self.inner.det_thresh:continue
            # The extra route shares the original primary confidence boundary.
            if observation.score<=self.inner.args.track_thresh:continue
            parent=int(matrix[:,column].argmax());strength=float(matrix[parent,column])
            if c+parent in claimed_before:continue
            if data['scores'][c+parent]<=self.inner.args.track_thresh:continue
            row=matrix[parent]
            if int(row.argmax())!=column:continue
            alternatives=np.delete(row,column);second=float(alternatives.max()) if len(alternatives) else 0.
            purity=(strength-second)/max(strength,1e-8)
            if strength<self.config.get('birth_minimum_vote',.3) or purity<.5:continue
            if good[parent]<4 or good[parent]/max(float(total[parent]),1.)<.6:continue
            def quality(i):
                q=transport['quality'][i]
                return q[quality_index] if q[quality_index]>=.15 else q[0]
            if min(quality(c+parent),quality(index))<.15:continue
            # A previous/current source-target match must be mutual, not merely
            # a point cloud shared by several candidates in the same region.
            contenders=matrix[:,column].copy();contenders[parent]=0
            if float(contenders.max())>.8*strength:continue
            self.statistics['candidate_evidence']+=1
            observation.activate(self.inner.kalman_filter,frame)
            self.inner.tracked_stracks.append(observation)
            self.statistics['candidate_births']+=1
            if quality_index==1 and min(transport['quality'][c+parent,1],transport['quality'][index,1])>=.15:
                self.statistics['IR_supported_births']+=1
            used.add(index)
        return rows
