"""Consult learned appearance evidence when current body points are missing."""
import numpy as np

from yolox.tracker.association_features import cosine
from yolox.tracker.prebirth_identity_tracker import PrebirthStrongRGBTracker


class MissingPointEvidenceTracker(PrebirthStrongRGBTracker):
    def __init__(self,config,**kwargs):
        super().__init__(config,**kwargs)
        self.statistics.update(semantic_rows=0,semantic_opened_edges=0,semantic_IR_increment_rows=0)

    def rank_cost(self,tracks,observations,base,frame,stage):
        result=super().rank_cost(tracks,observations,base,frame,stage)
        if not self.config.get('semantic_missing_points',False) or not base.size or stage not in ['primary','secondary']:return result
        columns=np.array([o.evidence['index']-self.current_start for o in observations],int)
        limit={'primary':.9,'secondary':.5}[stage];values=np.zeros_like(base);changes=np.zeros_like(base);similarity=np.zeros_like(base)
        for row,track in enumerate(tracks):
            context=track.evidence;lag=frame-context['frame']
            if lag<=0 or lag>8:continue
            if lag in self.config.get('lags',[1,2,4,8]):
                start,p,delta,vote,good,total=self.predict_pairs(frame,lag);index=context['index']-start
                if index<0 or index>=len(p):raise ValueError('Missing-point source index mismatch')
                reliable=good[index]>=4 and good[index]/max(float(total[index]),1.)>=.25 and vote[index].max(initial=0)>=.15
                if reliable:continue
            scores,increment=self.appearance_scores(frame,context,self.data)
            a,b=map(int,self.data['offsets'][frame-1:frame+1])
            raw=cosine(self.data['embedding'][context['index']:context['index']+1],self.data['embedding'][a:b])[0]
            for col,column in enumerate(columns):
                if raw[column]<.5:continue
                other=np.delete(scores,column)
                if scores[column]-(float(other.max()) if len(other) else 0.)<.1:continue
                values[row,col]=scores[column];changes[row,col]=increment[column];similarity[row,col]=raw[column]
        # A competing retained identity is evidence against a confident choice.
        if len(tracks)>1:
            ordered=np.sort(values,axis=0)
            values[:,ordered[-1]-ordered[-2]<.1]=0
        for row,track in enumerate(tracks):
            best=int(values[row].argmax());maximum=float(values[row,best])
            if maximum<self.config.get('semantic_minimum_probability',.9):continue
            feasible=base[row]<=limit
            finite=np.sort(base[row,feasible])
            clear=len(finite)==1 or (len(finite)>1 and finite[1]-finite[0]>=.12)
            if self.config.get('semantic_protect_motion',True) and clear:continue
            confidence=maximum
            weight=float(self.config.get('semantic_weight',.5))*confidence
            scores=np.array([o.score for o in observations]) if stage!='secondary' else np.ones(len(observations))
            adjusted=(1-weight)*base[row]+weight*(1-values[row]*scores)
            rejected=~feasible
            allowed=self.config.get('semantic_opening',False)&(values[row]>=.98)&(similarity[row]>=.85)
            adjusted[rejected&~allowed]=result[row,rejected&~allowed]
            result[row]=adjusted
            self.trusted[(track.track_id,observations[best].evidence['index'])]=confidence
            self.statistics['semantic_rows']+=1
            self.statistics['semantic_opened_edges']+=int((rejected&(adjusted<=limit)).sum())
            self.statistics['semantic_IR_increment_rows']+=int(self.cue=='ir' and abs(changes[row,best])>1e-4)
        return result
