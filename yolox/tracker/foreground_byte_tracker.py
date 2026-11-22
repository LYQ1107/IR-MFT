"""RGB body-point association with pseudo-IR geometry and causal object files."""
import math
import numpy as np

from yolox.tracker.basetrack import TrackState
from yolox.tracker.evidence_byte_tracker import EvidenceByteTracker,EvidenceTrack
from yolox.tracker.matching import ious


class ObjectFileTrack(EvidenceTrack):
    def __init__(self,tlwh,score,context):
        super().__init__(tlwh,score,context)
        self.object_observations=[context]

    def observe(self,observation):
        super().observe(observation)
        self.object_observations.append(observation.evidence)
        self.object_observations=self.object_observations[-32:]


class ForegroundByteTracker(EvidenceByteTracker):
    def __init__(self,config):
        super().__init__(config)
        self.inner.track_factory=ObjectFileTrack
        self.statistics.update(point_rows=0,point_opened_edges=0,history_anchors=0,
                               duplicate_vetoes=0,IR_supported_rows=0)
        self.co_visible={}
        self.lookup={}
        self.trusted={}

    def observations(self,frame,data,indices):
        if self.config.get('appearance_penalty',0):
            contexts=super().observations(frame,data,indices)
        else:
            # Pure classical variants do not need a hidden ReID encoder.
            one=np.ones(1,np.float32)
            contexts=[dict(owner=self,index=int(i),frame=int(frame),appearance=one.copy(),
                           shape=one.copy(),color=one.copy()) for i in indices]
        for context,index in zip(contexts,indices):
            context['quality']=data['foreground_transport']['quality'][index].astype(np.float32)
        return contexts

    def rank_cost(self,tracks,observations,base,frame,stage):
        result=super().rank_cost(tracks,observations,base,frame,stage)
        if not base.size or stage not in self.config.get('point_stages',['primary','secondary']):return result
        amount=float(self.config.get('point_weight',.8))
        if amount<=0:return result
        # The base graph is also used to diagnose explicitly opened edges.
        # Never let point updates mutate that reference through an alias.
        result=result.copy()
        columns=np.array([o.evidence['index']-self.current_start for o in observations],int)
        votes=np.zeros_like(base);confidence=np.zeros(len(tracks),np.float64)
        mode=self.config.get('source_mode','joint')
        for row,track in enumerate(tracks):
            history=track.object_observations if self.config.get('object_history',False) else [track.evidence]
            contributions=[]
            for context in history:
                lag=frame-context['frame']
                if lag not in self.lookup:continue
                start,matrix,good,total=self.lookup[lag]
                index=context['index']-start
                if index<0 or index>=len(matrix):raise ValueError('Historical source index mismatch')
                if good[index]<self.config.get('minimum_points',4):continue
                value=matrix[index,columns].astype(np.float64)
                maximum=float(value.max())
                if maximum<self.config.get('minimum_vote',.15):continue
                second=float(np.partition(value,-2)[-2]) if len(value)>1 else 0.
                purity=max(0.,(maximum-second)/max(maximum,1e-8))
                cover=float(good[index])/max(float(total[index]),1.)
                strength=min(1.,maximum/.4)
                certainty=cover*strength*min(1.,purity/.5)*math.exp(-lag/32.)
                if certainty<=.05:continue
                contributions.append((value/maximum,certainty,context))
            if not contributions:continue
            weights=np.array([x[1] for x in contributions])
            aggregate=np.average(np.stack([x[0] for x in contributions]),axis=0,weights=weights)
            best=float(aggregate.max())
            second=float(np.partition(aggregate,-2)[-2]) if len(aggregate)>1 else 0.
            purity=max(0.,(best-second)/max(best,1e-8))
            votes[row]=aggregate/max(best,1e-8)
            confidence[row]=float(weights.mean())*min(1.,purity/.5)
            self.statistics['history_anchors']+=len(contributions)
            ir_mode=mode in {'ir','joint'} or self.config.get('target_mode',mode) in {'ir','joint'}
            if ir_mode and (any(x[2]['quality'][1]>=.15 for x in contributions)
                            or any(o.evidence['quality'][1]>=.15 for o in observations)):
                self.statistics['IR_supported_rows']+=1
        if self.config.get('reciprocal_gate',True) and len(tracks)>1:
            strength=votes*confidence[:,None]
            for column in range(len(observations)):
                order=np.argsort(strength[:,column])
                first,second=order[-1],order[-2]
                if strength[first,column]>.2 and strength[second,column]>.8*strength[first,column]:
                    confidence[first]*=.4;confidence[second]*=.4
        limit={'primary':.9,'secondary':.5,'pending':.7}[stage]
        for row,track in enumerate(tracks):
            if confidence[row]<self.config.get('minimum_confidence',.3):continue
            weight=amount*confidence[row]
            scores=np.array([o.score for o in observations]) if stage!='secondary' else np.ones(len(observations))
            point_cost=1-votes[row]*scores
            adjusted=(1-weight)*result[row]+weight*point_cost
            rejected=base[row]>limit
            if confidence[row]<self.config.get('opening_confidence',.7):adjusted[rejected]=result[row,rejected]
            self.statistics['point_rows']+=1
            self.statistics['point_opened_edges']+=int((rejected&(adjusted<=limit)).sum())
            result[row]=adjusted
            for column,observation in enumerate(observations):
                self.trusted[(track.track_id,observation.evidence['index'])]=confidence[row]*votes[row,column]
        return result

    def update(self,frame,data):
        transport=data['foreground_transport']
        if frame!=self.inner.frame_id+1:raise ValueError('Nonsequential frame')
        self.current_start=int(data['offsets'][frame-1]);self.lookup={};self.trusted={}
        source_mode=self.config.get('source_mode','joint');target_mode=self.config.get('target_mode',source_mode)
        lags=self.config.get('lags',[1])
        for lag in lags:
            if frame<=lag:continue
            start,end=map(int,data['offsets'][frame-lag-1:frame-lag+1])
            ncurrent=int(data['offsets'][frame]-data['offsets'][frame-1])
            a,b=map(int,transport[f'vote_offsets_{lag}'][frame-1:frame+1])
            c,d=map(int,transport[f'count_offsets_{lag}'][frame-1:frame+1])
            matrix=transport[f'vote_{lag}_{source_mode}_{target_mode}'][a:b].reshape(end-start,ncurrent)
            self.lookup[lag]=(start,matrix,transport[f'good_{lag}_{source_mode}'][c:d],transport[f'total_{lag}_{source_mode}'][c:d])
        before={t.track_id:t for t in self.inner.tracked_stracks+self.inner.lost_stracks}
        rows=super().update(frame,data)
        if self.config.get('preserve_distinct_objects',False):
            active=self.inner.tracked_stracks
            for i,first in enumerate(active):
                if first.frame_id!=frame:continue
                for second in active[i+1:]:
                    if second.frame_id!=frame:continue
                    if ious(first.tlbr[None],second.tlbr[None])[0,0]>.3:continue
                    f=self.trusted.get((first.track_id,first.evidence['index']),0)
                    s=self.trusted.get((second.track_id,second.evidence['index']),0)
                    ir_mode=source_mode in {'ir','joint'} or target_mode in {'ir','joint'}
                    quality=lambda t:float(t.evidence['quality'][1] if ir_mode and t.evidence['quality'][1]>=.15 else t.evidence['quality'][0])
                    if min(f,s)<.65 or min(quality(first),quality(second))<.15:continue
                    self.co_visible[tuple(sorted((first.track_id,second.track_id)))]=frame
            existing={t.track_id for t in self.inner.tracked_stracks+self.inner.lost_stracks}
            removed={t.track_id for t in self.inner.removed_stracks}
            for identity,track in before.items():
                if identity in existing or identity in removed or track.state!=TrackState.Lost:continue
                if frame-track.end_frame>self.inner.max_time_lost:continue
                distinct=any(frame-self.co_visible.get(tuple(sorted((identity,p.track_id))),-100000)<=128 for p in active)
                if distinct:
                    self.inner.lost_stracks.append(track);self.statistics['duplicate_vetoes']+=1
            existing={t.track_id for t in self.inner.tracked_stracks+self.inner.lost_stracks}
            for identity,track in before.items():
                if identity in existing or identity in removed or track.state!=TrackState.Tracked:continue
                distinct=any(frame-self.co_visible.get(tuple(sorted((identity,p.track_id))),-100000)<=128 for p in self.inner.lost_stracks)
                if distinct:
                    self.inner.tracked_stracks.append(track);self.statistics['duplicate_vetoes']+=1
                    if track.is_activated and track.frame_id==frame and track.tlwh[2]*track.tlwh[3]>100:
                        rows.append((int(frame),track.tlwh.copy(),track.track_id,track.score))
            self.co_visible={p:t for p,t in self.co_visible.items() if frame-t<=128}
        return rows
