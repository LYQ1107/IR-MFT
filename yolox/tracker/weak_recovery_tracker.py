"""Recover a retained identity from current weak but supported observations."""
import math

import numpy as np

from yolox.tracker.basetrack import TrackState
from yolox.tracker.cycle_candidate_tracker import CycleBirthMixin
from yolox.tracker.residual_foreground_tracker import ResidualForegroundTracker
from yolox.tracker.matching import ious,linear_assignment


class WeakRecoveryMixin:
    def __init__(self,config):
        super().__init__(config)
        self.statistics.update(weak_recoveries=0,weak_recovery_pairs=0,weak_recovery_IR_links=0)

    def weak_scores(self,frame,context,data):
        lag=frame-context['frame'];point=data['foreground_transport']
        a,b=map(int,data['offsets'][frame-1:frame+1]);c,d=map(int,data['offsets'][frame-lag-1:frame-lag+1])
        row=context['index']-c
        calibrated=self.config.get('weak_recovery_calibrated',hasattr(self,'predict_pairs'))
        if calibrated:
            if not hasattr(self,'predict_pairs'):raise ValueError('Calibrated recovery needs a fitted model')
            start,p,delta,vote,good,total=self.predict_pairs(frame,lag)
            if start!=c or row<0 or row>=len(p):return None
            missing=good[row]<4 or good[row]/max(float(total[row]),1.)<.25 or float(vote[row].max(initial=0))<.15
            if missing and not self.config.get('weak_allow_appearance',True):return None
            scores=p[row].astype(np.float64).copy()
            if not missing:scores[vote[row]<.15]=0
            else:
                first=np.asarray(data['embedding'][context['index']],np.float32)
                current=np.asarray(data['embedding'][a:b],np.float32)
                similarity=current@first/np.maximum(np.linalg.norm(current,axis=1)*np.linalg.norm(first),1e-8)
                scores[similarity<self.config.get('weak_appearance_cosine',.5)]=0
            ir=(self.cue=='ir')&(np.abs(delta[row])>1e-4)
            return scores,np.asarray(ir,bool)
        target=self.config.get('weak_recovery_geometry','joint')
        v0,v1=map(int,point[f'vote_offsets_{lag}'][frame-1:frame+1]);p0,p1=map(int,point[f'count_offsets_{lag}'][frame-1:frame+1])
        matrix=point[f'vote_{lag}_rgb_{target}'][v0:v1].reshape(d-c,b-a).astype(np.float64)
        good=point[f'good_{lag}_rgb'][p0:p1];total=point[f'total_{lag}_rgb'][p0:p1]
        if not matrix.size or row<0 or row>=len(matrix) or good[row]<4:return None
        value=matrix[row];maximum=float(value.max())
        if maximum<.15:return None
        second=float(np.partition(value,-2)[-2]) if len(value)>1 else 0.
        purity=max(0.,(maximum-second)/max(maximum,1e-8))
        confidence=float(good[row])/max(float(total[row]),1.)*min(1.,maximum/.4)*min(1.,purity/.5)*math.exp(-lag/32.)
        scores=value/max(maximum,1e-8)*confidence
        supported=np.zeros(b-a,bool)
        if target in {'ir','joint'}:
            supported=(point['quality'][context['index'],1]>=.15)&(point['quality'][a:b,1]>=.15)
        return scores,supported

    def update(self,frame,data):
        rows=super().update(frame,data)
        if not self.config.get('weak_recovery',False):return rows
        used={t.evidence['index'] for t in self.inner.tracked_stracks if t.frame_id==frame}
        observations=[o for o in self.created if o.evidence['index'] not in used and
                      self.config.get('weak_minimum_score',.3)<=o.score<=self.inner.args.track_thresh]
        lost=[t for t in self.inner.lost_stracks if t.state==TrackState.Lost and frame-t.end_frame<=self.inner.max_time_lost]
        if not observations or not lost:return rows
        a=int(data['offsets'][frame-1]);columns=np.array([o.evidence['index']-a for o in observations],int)
        values=np.zeros((len(lost),len(observations)),np.float64);is_ir=np.zeros_like(values,bool)
        for i,track in enumerate(lost):
            history=track.object_observations if self.config.get('object_files',True) else [track.evidence]
            contexts=[c for c in reversed(history) if frame-c['frame'] in [1,2,4,8]]
            if not contexts:continue
            evidence=self.weak_scores(frame,contexts[0],data)
            if evidence is None:continue
            scores,support=evidence
            for j,column in enumerate(columns):
                other=np.delete(scores,column);margin=float(scores[column])-(float(other.max()) if len(other) else 0.)
                if margin<self.config.get('weak_minimum_margin',.1):continue
                active=[t for t in self.inner.tracked_stracks if t.frame_id==frame]
                blocked=False
                for peer in active:
                    if float(ious(observations[j].tlbr[None],peer.tlbr[None])[0,0])<=.7:continue
                    pair=tuple(sorted((track.track_id,peer.track_id)))
                    known_distinct=frame-self.co_visible.get(pair,-100000)<=128
                    if not known_distinct:blocked=True;break
                if blocked:continue
                values[i,j]=scores[column];is_ir[i,j]=support[column]
        self.statistics['weak_recovery_pairs']+=int((values>0).sum())
        # A second retained identity with similar support is an unresolved
        # ownership conflict, not permission to pick an arbitrary old ID.
        for j in range(len(observations)):
            if len(lost)<2:continue
            order=np.argsort(values[:,j]);first,second=order[-1],order[-2]
            if values[first,j]-values[second,j]<self.config.get('weak_minimum_margin',.1):values[:,j]=0
        minimum=float(self.config.get('weak_recovery_minimum',.8))
        pairs,_,_=linear_assignment(1-values,1-minimum)
        recovered=set()
        for i,j in pairs:
            if values[i,j]<minimum:continue
            track=lost[i];observation=observations[j]
            track.re_activate(observation,frame,new_id=False)
            self.inner.tracked_stracks.append(track);recovered.add(track.track_id)
            if track.tlwh[2]*track.tlwh[3]>100:rows.append((int(frame),track.tlwh.copy(),track.track_id,track.score))
            self.trusted[(track.track_id,observation.evidence['index'])]=float(values[i,j])
            self.statistics['weak_recoveries']+=1
            self.statistics['weak_recovery_IR_links']+=int(is_ir[i,j])
        self.inner.lost_stracks=[t for t in self.inner.lost_stracks if t.track_id not in recovered]
        return rows


class RecoverableForegroundTracker(CycleBirthMixin,WeakRecoveryMixin,ResidualForegroundTracker):
    pass
