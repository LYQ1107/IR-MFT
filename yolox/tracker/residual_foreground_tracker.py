"""RGB-preserving geometry increments, verified by causal object observations.

Generated IR is allowed to resolve uncertain RGB point ownership. It does not
remove RGB source points or replace unambiguous RGB evidence everywhere.
"""
import math
import numpy as np

from yolox.tracker.candidate_foreground_tracker import CandidateForegroundTracker
from yolox.tracker.evidence_byte_tracker import EvidenceByteTracker


def normalized_evidence(history,lookup,columns,frame,minimum_points=4):
    contributions=[]
    for context in history:
        lag=frame-context['frame']
        if lag not in lookup:continue
        start,matrix,good,total=lookup[lag];index=context['index']-start
        if index<0 or index>=len(matrix):raise ValueError('Historical point index mismatch')
        if good[index]<minimum_points:continue
        value=matrix[index,columns].astype(np.float64);maximum=float(value.max())
        if maximum<.15:continue
        second=float(np.partition(value,-2)[-2]) if len(value)>1 else 0.
        purity=max(0.,(maximum-second)/max(maximum,1e-8))
        certainty=float(good[index])/max(float(total[index]),1.)*min(1.,maximum/.4)*min(1.,purity/.5)*math.exp(-lag/32.)
        if certainty<=.05:continue
        contributions.append((value/maximum,certainty,context,purity))
    if not contributions:return None
    weights=np.array([x[1] for x in contributions]);aggregate=np.average(np.stack([x[0] for x in contributions]),axis=0,weights=weights)
    best=float(aggregate.max());second=float(np.partition(aggregate,-2)[-2]) if len(aggregate)>1 else 0.
    purity=max(0.,(best-second)/max(best,1e-8))
    confidence=float(weights.mean())*min(1.,purity/.5)
    return aggregate/max(best,1e-8),confidence,purity,contributions


class ResidualForegroundTracker(CandidateForegroundTracker):
    def __init__(self,config):
        config=dict(dict(source_mode='rgb',target_mode='rgb'),**config)
        super().__init__(config)
        if config.get('source_mode','rgb')!='rgb' or config.get('target_mode','rgb')!='rgb':
            raise ValueError('The preserved point stream must be RGB')
        self.statistics.update(IR_increment_rows=0,IR_conflicts=0,IR_temporally_rejected=0,IR_applied_rows=0,motion_preserved_rows=0)

    def update(self,frame,data):
        self.auxiliary_lookup={};self.current_transport=data['foreground_transport']
        transport=data['foreground_transport']
        auxiliary=self.config.get('ir_target_mode','joint')
        for lag in (self.config.get('ir_validation_lags',[1,2,4,8]) if self.config.get('IR_increment',True) else []):
            if frame<=lag:continue
            start,end=map(int,data['offsets'][frame-lag-1:frame-lag+1])
            ncurrent=int(data['offsets'][frame]-data['offsets'][frame-1])
            a,b=map(int,transport[f'vote_offsets_{lag}'][frame-1:frame+1]);c,d=map(int,transport[f'count_offsets_{lag}'][frame-1:frame+1])
            key=f'vote_{lag}_rgb_{auxiliary}'
            self.auxiliary_lookup[lag]=(start,transport[key][a:b].reshape(end-start,ncurrent),transport[f'good_{lag}_rgb'][c:d],transport[f'total_{lag}_rgb'][c:d])
        return super().update(frame,data)

    def rank_cost(self,tracks,observations,base,frame,stage):
        result=EvidenceByteTracker.rank_cost(self,tracks,observations,base,frame,stage)
        if not base.size or stage not in self.config.get('point_stages',['primary','secondary']):return result
        amount=float(self.config.get('point_weight',.8))
        if amount<=0:return result
        result=result.copy();columns=np.array([o.evidence['index']-self.current_start for o in observations],int)
        votes=np.zeros_like(base);confidence=np.zeros(len(tracks),np.float64);incremented=np.zeros(len(tracks),bool)
        for row,track in enumerate(tracks):
            history=track.object_observations if self.config.get('object_history',False) else [track.evidence]
            rgb=normalized_evidence(history,self.lookup,columns,frame)
            if rgb is not None:
                votes[row],confidence[row]=rgb[:2]
                self.statistics['history_anchors']+=len(rgb[3])
            if not self.config.get('IR_increment',True):continue
            auxiliary=normalized_evidence(track.object_observations,self.auxiliary_lookup,columns,frame)
            if auxiliary is None:continue
            av,ac,ap,anchors=auxiliary;choice=int(av.argmax())
            if observations[choice].evidence['quality'][1]<.15:continue
            supported=[a for a in anchors if a[2]['quality'][1]>=.15 and int(a[0].argmax())==choice]
            required=int(self.config.get('minimum_IR_anchors',2))
            if len(supported)<required:
                self.statistics['IR_temporally_rejected']+=1;continue
            if self.config.get('IR_unanimous',True) and len(supported)!=len(anchors):
                self.statistics['IR_temporally_rejected']+=1;continue
            rp=rgb[2] if rgb is not None else 0.
            if rgb is not None and int(rgb[0].argmax())!=choice:self.statistics['IR_conflicts']+=1
            if rp>self.config.get('RGB_ambiguity_ceiling',.8):continue
            if ap<rp+self.config.get('IR_purity_increment',.15):continue
            if ac<confidence[row]+self.config.get('IR_confidence_increment',.1):continue
            # Multiple lags confirm a hypothesis; correlated views do not get
            # multiplied into an artificial near-one probability.
            ac=min(ac,min(x[1] for x in supported))
            if ac<confidence[row]+self.config.get('IR_confidence_increment',.1):continue
            alpha=float(self.config.get('IR_residual_weight',.5))
            if rgb is None:votes[row]=av;confidence[row]=alpha*ac
            else:
                votes[row]=(1-alpha)*votes[row]+alpha*av
                votes[row]/=max(float(votes[row].max()),1e-8)
                confidence[row]=(1-alpha)*confidence[row]+alpha*ac
            self.statistics['IR_increment_rows']+=1;self.statistics['IR_supported_rows']+=1
            incremented[row]=True
        if self.config.get('reciprocal_gate',True) and len(tracks)>1:
            strength=votes*confidence[:,None]
            for column in range(len(observations)):
                order=np.argsort(strength[:,column]);first,second=order[-1],order[-2]
                if strength[first,column]>.2 and strength[second,column]>.8*strength[first,column]:
                    confidence[first]*=.4;confidence[second]*=.4
        limit={'primary':.9,'secondary':.5,'pending':.7}[stage]
        ambiguity=None
        if self.config.get('preserve_clear_motion',False):
            feasible=np.isfinite(base)&(base<=limit)
            masked=np.where(feasible,base,np.inf)
            row_best=np.min(masked,axis=1,keepdims=True);col_best=np.min(masked,axis=0,keepdims=True)
            margin=float(self.config.get('motion_ambiguity_margin',.12))
            close_row=feasible&(base-row_best<margin);close_col=feasible&(base-col_best<margin)
            ambiguity=(close_row&(close_row.sum(1,keepdims=True)>1))|(close_col&(close_col.sum(0,keepdims=True)>1))
        for row,track in enumerate(tracks):
            if confidence[row]<self.config.get('minimum_confidence',.3):continue
            if ambiguity is not None:
                original_has_match=bool((base[row]<=limit).any())
                strong_gap=not original_has_match and confidence[row]>=self.config.get('gap_point_confidence',.9)
                if not ambiguity[row].any() and not strong_gap:
                    best=int(votes[row].argmax())
                    if best==int(base[row].argmin()) and confidence[row]>=.65:
                        index=observations[best].evidence['index']
                        self.trusted[(track.track_id,index)]=confidence[row]*votes[row,best]
                    self.statistics['motion_preserved_rows']+=1;continue
            weight=amount*confidence[row]
            scores=np.array([o.score for o in observations]) if stage!='secondary' else np.ones(len(observations))
            adjusted=(1-weight)*result[row]+weight*(1-votes[row]*scores)
            rejected=base[row]>limit
            if confidence[row]<self.config.get('opening_confidence',.7):adjusted[rejected]=result[row,rejected]
            self.statistics['point_rows']+=1;self.statistics['point_opened_edges']+=int((rejected&(adjusted<=limit)).sum())
            if incremented[row]:self.statistics['IR_applied_rows']+=1
            result[row]=adjusted
            for column,observation in enumerate(observations):
                self.trusted[(track.track_id,observation.evidence['index'])]=confidence[row]*votes[row,column]
        return result
