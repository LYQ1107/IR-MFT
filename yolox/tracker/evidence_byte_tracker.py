"""Evidence ranking within original ByteTrack's feasible association graph.

Lifecycle, score thresholds, low-score stage, duplicate removal and Kalman
updates remain the original implementation. Cues can penalize ambiguous
existing edges; they cannot silently open rejected nonoverlapping edges.
"""
from types import SimpleNamespace

import numpy as np
from yolox.tracker.byte_tracker import BYTETracker,STrack
from yolox.tracker.perceptual_features import unit


class EvidenceTrack(STrack):
    def __init__(self,tlwh,score,context):
        super().__init__(tlwh,score)
        self.owner=context['owner']
        self.evidence=context
        self.appearance=context['appearance'].copy()
        self.shape_feature=context['shape'].copy()
        self.color_feature=context['color'].copy()

    def observe(self,observation):
        context=observation.evidence
        protected=self.owner.config.get('protected_memory',False)
        ambiguous=context['index'] in self.owner.ambiguous_indices
        if protected and ambiguous:
            self.owner.statistics['protected_updates']+=1
        else:
            alpha=float(self.owner.config.get('template_ema',.9))
            self.appearance=unit(alpha*self.appearance+(1-alpha)*context['appearance'])
            self.shape_feature=unit(alpha*self.shape_feature+(1-alpha)*context['shape'])
            self.color_feature=unit(alpha*self.color_feature+(1-alpha)*context['color'])
        self.evidence=context

    def update(self,new_track,frame_id):
        super().update(new_track,frame_id)
        self.observe(new_track)

    def re_activate(self,new_track,frame_id,new_id=False):
        super().re_activate(new_track,frame_id,new_id)
        self.observe(new_track)


class EvidenceByteTracker:
    def __init__(self,config):
        self.config=dict(config)
        self.statistics=dict(ambiguous_edges=0,adjusted_edges=0,protected_updates=0)
        self.ambiguous_indices=set()
        self.normalizer=None
        if config.get('normalizer'):
            with np.load(config['normalizer']) as z:self.normalizer={k:z[k] for k in z.files}
        self.inner=BYTETracker(SimpleNamespace(track_thresh=.6,track_buffer=30,match_thresh=.9,mot20=False),
                               frame_rate=25,track_factory=EvidenceTrack,association_hook=self.rank_cost)

    def transform(self,values,field):
        values=np.asarray(values,np.float32)
        if self.normalizer is not None:
            values=values-self.normalizer[field+'_mean']
            projection=self.normalizer[field+'_projection']
            if projection.size:values=values@projection
        return values/np.maximum(np.linalg.norm(values,axis=1,keepdims=True),1e-8)

    def observations(self,frame,data,indices):
        raw=self.transform(data['embedding'][indices],'embedding')
        view=self.config.get('appearance_view','raw')
        if view!='raw':
            name={'ir_mask':'ir_mask_embedding','rgb_mask':'rgb_mask_embedding','double_rgb':'embedding'}[view]
            other=self.transform(data[name][indices],name)
            alpha=self.config.get('mask_weight',.5)
            appearance=np.concatenate([raw*np.sqrt(1-alpha),other*np.sqrt(alpha)],axis=1)
        else:appearance=raw
        shape_name='ir_hog' if self.config.get('shape','rgb')=='ir' else 'rgb_hog'
        shape=np.asarray(data[shape_name][indices],np.float32)
        shape/=np.maximum(np.linalg.norm(shape,axis=1,keepdims=True),1e-8)
        color=np.asarray(data['color'][indices],np.float32)
        color/=np.maximum(np.linalg.norm(color,axis=1,keepdims=True),1e-8)
        return [dict(owner=self,index=int(i),frame=int(frame),appearance=a.copy(),shape=s.copy(),color=c.copy())
                for i,a,s,c in zip(indices,appearance,shape,color)]

    def rank_cost(self,tracks,observations,base,frame,stage):
        if not base.size:return base
        if stage!='primary' and not self.config.get('rank_secondary',False):return base
        weights=[float(self.config.get(k,0)) for k in ['appearance_penalty','shape_penalty','color_penalty']]
        if not any(weights):return base
        limit={'primary':.9,'secondary':.5,'pending':.7}[stage]
        feasible=np.isfinite(base)&(base<=limit)
        masked=np.where(feasible,base,np.inf)
        row_best=np.min(masked,axis=1,keepdims=True)
        col_best=np.min(masked,axis=0,keepdims=True)
        margin=float(self.config.get('ambiguity_margin',.12))
        close_row=feasible&(base-row_best<margin)
        close_col=feasible&(base-col_best<margin)
        ambiguous=(close_row&(close_row.sum(1,keepdims=True)>1))|(close_col&(close_col.sum(0,keepdims=True)>1))
        if not ambiguous.any():return base
        self.statistics['ambiguous_edges']+=int(ambiguous.sum())
        for column in np.flatnonzero(ambiguous.any(axis=0)):
            self.ambiguous_indices.add(observations[column].evidence['index'])
        cue=np.zeros_like(base)
        for weight,track_field,obs_field in zip(weights,['appearance','shape_feature','color_feature'],['appearance','shape','color']):
            if not weight:continue
            first=np.stack([getattr(t,track_field) for t in tracks])
            second=np.stack([o.evidence[obs_field] for o in observations])
            cue+=weight*np.clip(1-first@second.T,0,2)
        best=np.min(np.where(feasible,cue,np.inf),axis=1,keepdims=True)
        relative=np.maximum(cue-best,0)
        result=base.copy()
        result[ambiguous]+=relative[ambiguous]
        # Keep the original rejected graph rejected, irrespective of cue scale.
        result[~feasible]=base[~feasible]
        self.statistics['adjusted_edges']+=int((result!=base).sum())
        return result

    def update(self,frame,data):
        if frame!=self.inner.frame_id+1:raise ValueError('Nonsequential frame')
        self.ambiguous_indices=set()
        a,b=data['offsets'][frame-1:frame+1];indices=np.arange(a,b,dtype=int)
        evidence=self.observations(frame,data,indices)
        detections=np.column_stack([data['boxes'][a:b],data['scores'][a:b]]).astype(np.float32)
        tracks=self.inner.update(detections,(1,1),(1,1),evidence=evidence)
        return [(int(frame),t.tlwh.copy(),t.track_id,t.score) for t in tracks if t.tlwh[2]*t.tlwh[3]>100]
