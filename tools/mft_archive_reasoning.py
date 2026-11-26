"""Causal archive-to-tracklet evidence; labels never enter the online linker.

The inner tracker keeps its original lifetime and associations. A link changes
only this frame and later output identities; previously emitted rows stay fixed.
Archives are created after actual departure from tracked_stracks/lost_stracks.
"""
from dataclasses import dataclass
import numpy as np
from tools.mft_candidate_risk_features import _margins
from tools.mft_candidate_risk_model import sigmoid
from tools.mft_candidate_risk_labels import current_detection_labels,PastLabelHistory

FEATURE_NAMES=(
    'log_gap','log_new_public_age','log_old_template_age','dx_body','dy_body',
    'distance_body','abs_log_width','abs_log_height','abs_log_aspect',
    'RGB_cosine','IR_cosine','IR_available','RGB_row_margin','RGB_column_margin',
    'IR_row_margin','IR_column_margin','old_template_quality','new_template_quality',
    'RGB_quality_support','IR_quality_support','cue_disagreement',
    'log_old_reliable_count','log_new_reliable_count','new_detection_score',
    'relative_displacement_per_frame','log_new_template_recency')
IR_FEATURES=tuple(FEATURE_NAMES.index(k)for k in[
    'IR_cosine','IR_available','IR_row_margin','IR_column_margin',
    'IR_quality_support','cue_disagreement'])


def snapshot(track):
    assert track.last_box is not None and track.memory_frame is not None
    return dict(internal=int(track.track_id),rgb=track.rgb.copy(),
        ir=None if track.ir is None else track.ir.copy(),box=track.last_box.copy(),
        last_frame=int(track.frame_id),template_frame=int(track.memory_frame),
        quality=float(track.memory_quality),accepted=int(track.accepted),score=float(track.score))


def features(new,old,frame,speed):
    """Rows are current tracklets; columns are expired identity templates."""
    assert type(frame)is int and frame>0 and 0<=speed<=1
    x=np.zeros((len(new),len(old),len(FEATURE_NAMES)),np.float32)
    valid=np.zeros(x.shape[:2],bool);rgb=np.zeros(valid.shape,np.float64)
    ir=np.zeros_like(rgb);ir_valid=np.zeros_like(valid)
    for i,a in enumerate(new):
        assert a['last_frame']==frame and a['template_frame']<=frame and a['accepted']>=2
        size=np.maximum(a['box'][2:]-a['box'][:2],1.)
        center=(a['box'][:2]+a['box'][2:])/2
        for j,b in enumerate(old):
            gap=frame-b['last_frame'];assert gap>0 and b['template_frame']<=b['last_frame']
            past_size=np.maximum(b['box'][2:]-b['box'][:2],1.);scale=size/past_size
            delta=(center-(b['box'][:2]+b['box'][2:])/2)/np.linalg.norm(past_size)
            distance=float(np.linalg.norm(delta))
            if not 2<=gap<=250 or (scale<.5).any()or(scale>2).any()or distance>.5+speed*gap:continue
            valid[i,j]=True;rgb[i,j]=float(np.clip(a['rgb']@b['rgb'],-1,1))
            available=a['ir']is not None and b['ir']is not None
            if available:ir[i,j]=float(np.clip(a['ir']@b['ir'],-1,1));ir_valid[i,j]=True
            quality=float(np.sqrt(a['quality']*b['quality']))
            x[i,j]=[
                np.log1p(gap),np.log1p(frame-a['first_public_frame']),
                np.log1p(frame-b['template_frame']),*np.clip(delta,-8,8),min(distance,8),
                abs(float(np.log(scale[0]))),abs(float(np.log(scale[1]))),
                abs(float(np.log(scale[0]/scale[1]))),rgb[i,j],ir[i,j],float(available),0,0,0,0,
                b['quality'],a['quality'],rgb[i,j]*quality,ir[i,j]*quality if available else 0.,
                abs(rgb[i,j]-ir[i,j])if available else 0.,np.log1p(b['accepted']),
                np.log1p(a['accepted']),a['score'],distance/gap,np.log1p(frame-a['template_frame'])]
    for name,values in zip(['RGB_row_margin','RGB_column_margin','IR_row_margin','IR_column_margin'],
            (*_margins(rgb,valid),*_margins(ir,ir_valid))):
        x[...,FEATURE_NAMES.index(name)]=values
    assert np.isfinite(x).all()
    return x,valid


class ArchiveLogistic:
    def __init__(self,mean,scale,kind):
        assert kind in ['RGB','paired'];self.kind=kind
        self.mean=np.asarray(mean,np.float64).copy();self.scale=np.asarray(scale,np.float64).copy()
        assert self.mean.shape==self.scale.shape==(len(FEATURE_NAMES),)
        assert np.isfinite(self.mean).all()and np.isfinite(self.scale).all()and(self.scale>0).all()
        if kind=='RGB':self.mean[list(IR_FEATURES)]=0;self.scale[list(IR_FEATURES)]=1
        self.coefficients=np.zeros(len(FEATURE_NAMES)+1,np.float64)

    def design(self,values):
        a=np.asarray(values,np.float64).copy()
        assert a.ndim==2 and a.shape[1]==len(FEATURE_NAMES)and np.isfinite(a).all()
        if self.kind=='RGB':a[:,IR_FEATURES]=0
        z=np.empty((len(a),len(FEATURE_NAMES)+1),np.float64)
        z[:,:-1]=np.clip((a-self.mean)/self.scale,-8,8);z[:,-1]=1
        return z

    def probabilities(self,values):return sigmoid(self.design(values)@self.coefficients)

    def to_dict(self):
        return dict(feature_names=list(FEATURE_NAMES),kind=self.kind,mean=self.mean.tolist(),
            scale=self.scale.tolist(),coefficients=self.coefficients.tolist(),seed=0,standardized_clip=8.)

    @classmethod
    def from_dict(cls,d):
        assert d['feature_names']==list(FEATURE_NAMES)and d['seed']==0 and d['standardized_clip']==8.
        m=cls(d['mean'],d['scale'],d['kind']);v=np.asarray(d['coefficients'],np.float64)
        assert v.shape==m.coefficients.shape and np.isfinite(v).all();m.coefficients=v.copy()
        return m


def choose_links(probabilities,eligible,threshold,margin=.05):
    """Weak eligible competitors still count in both uniqueness margins."""
    p=np.asarray(probabilities,np.float64);eligible=np.asarray(eligible,bool)
    assert p.shape==eligible.shape and 0<=threshold<=1 and margin>0
    scored=np.where(eligible,p,-np.inf);result=[]
    for i in range(len(p)):
        if not np.isfinite(scored[i]).any():continue
        j=int(np.argmax(scored[i]))
        if int(np.argmax(scored[:,j]))!=i or p[i,j]<threshold:continue
        row=np.delete(scored[i],j);column=np.delete(scored[:,j],i)
        competitor=max(float(row.max())if row.size else -np.inf,float(column.max())if column.size else -np.inf)
        if p[i,j]-competitor>=margin:result.append((i,j))
    assert len({j for i,j in result})==len(result)
    return result


class ArchiveLinker:
    def __init__(self,inner,speed,*,kind='off',models=None,thresholds=None,observer=None):
        assert kind in ['off','RGB','paired']and 0<=speed<=1
        if kind!='off':assert set(models)==set(thresholds)=={'RGB','paired'}
        self.inner,self.speed,self.kind=inner,speed,kind
        self.models,self.thresholds,self.observer=models,thresholds,observer;self.sequence=None

    def reset(self,sequence):
        self.sequence=sequence;self.previous_pool={};self.closed={};self.published={}
        self.first_public={};self.linked=set();self.statistics=dict(frames=0,departures=0,
            archived_templates=0,candidate_pairs=0,identity_links=0,delayed_identity_links=0,
            paired_calls=0,RGB_calls=0,maximum_live_archives=0)

    def update(self,sequence,frame,detections,rgb,ir=None):
        if sequence!=self.sequence:
            assert frame==1;self.reset(sequence)
        assert frame==self.statistics['frames']+1
        rows=self.inner.update(sequence,frame,detections,rgb,ir)
        core=self.inner.core;pool=core.tracked_stracks+core.lost_stracks
        pool_ids={int(t.track_id)for t in pool}
        for internal,memory in self.previous_pool.items():
            if internal in pool_ids:continue
            self.statistics['departures']+=1
            if internal in self.published and memory['accepted']>=3:
                root=self.published[internal];self.closed[root]=memory
                self.statistics['archived_templates']+=1
        self.closed={k:v for k,v in self.closed.items()if frame-v['last_frame']<=250}
        active_roots={self.published[i]for i in pool_ids if i in self.published}
        visible={int(t.track_id):t for t in core.tracked_stracks
            if t.is_activated and t.frame_id==frame and t.measurement_frame==frame}
        output_ids={int(row[2])for row in rows};assert output_ids<=visible.keys()
        # Published IDs may be resolved later, for at most32 frames after first
        # output. Historical rows are never revisited or buffered for rewriting.
        pending=[]
        for internal in sorted(output_ids):
            self.first_public.setdefault(internal,frame)
            t=visible[internal]
            if internal in self.linked or frame-self.first_public[internal]>32 or t.accepted<2 or t.rgb is None:continue
            value=snapshot(t);value['first_public_frame']=self.first_public[internal];pending.append(value)
        old=[(root,value)for root,value in sorted(self.closed.items())if root not in active_roots]
        if pending and old:
            x,eligible=features(pending,[v for k,v in old],frame,self.speed)
            self.statistics['candidate_pairs']+=int(eligible.sum())
            if self.observer is not None:
                self.observer(x.copy(),eligible.copy(),tuple(v['internal']for v in pending),
                    tuple(v['internal']for k,v in old),
                    tuple(int(visible[v['internal']].proposal_index)for v in pending),frame)
            if self.kind!='off'and eligible.any():
                kind=self.kind
                if kind=='paired'and not all(v['ir']is not None for v in pending+[v for k,v in old]):kind='RGB'
                self.statistics['paired_calls'if kind=='paired'else'RGB_calls']+=1
                p=np.zeros(eligible.shape,np.float64);p[eligible]=self.models[kind].probabilities(x[eligible])
                for i,j in choose_links(p,eligible,self.thresholds[kind]):
                    internal=pending[i]['internal'];root=old[j][0]
                    assert root not in active_roots and root in self.closed
                    self.published[internal]=root;self.linked.add(internal);del self.closed[root]
                    active_roots.add(root);self.statistics['identity_links']+=1
                    self.statistics['delayed_identity_links']+=int(frame>self.first_public[internal])
        for internal in output_ids:self.published.setdefault(internal,internal)
        result=[(f,box,self.published[internal],score)for f,box,internal,score in rows]
        assert len({row[2]for row in result})==len(result)
        self.previous_pool={int(t.track_id):snapshot(t)for t in pool if t.accepted>=3 and t.rgb is not None
            and t.last_box is not None and t.memory_frame is not None}
        self.statistics['frames']+=1
        self.statistics['maximum_live_archives']=max(self.statistics['maximum_live_archives'],len(self.closed))
        return result


@dataclass(frozen=True)
class ArchiveBatch:
    sequence:str
    frame:int
    x:np.ndarray
    y:np.ndarray


class ArchiveTrainCollector:
    def __init__(self,speed,sink):
        from tools.mft_candidate_risk_teacher import CandidateTeacher
        self.sink=sink;self.sequence=None;self.current_labels=None
        self.linker=ArchiveLinker(CandidateTeacher(speed),speed,observer=self.observe)

    def observe(self,x,eligible,new_ids,old_ids,indices,frame):
        assert self.current_labels is not None
        known=np.zeros(eligible.shape,bool);targets=np.zeros(eligible.shape,np.float32)
        for j,internal in enumerate(old_ids):
            prior=self.history.prior(internal,frame)
            if prior is None:continue
            for i,index in enumerate(indices):
                label=int(self.current_labels[index])
                if eligible[i,j]and label>=0:known[i,j]=True;targets[i,j]=float(label==prior)
        self.statistics['eligible_pairs']+=int(eligible.sum());self.statistics['known_pairs']+=int(known.sum())
        self.statistics['positive_pairs']+=int(targets[known].sum())
        self.statistics['negative_pairs']+=int(known.sum()-targets[known].sum())
        self.statistics['unknown_pairs']+=int((eligible&~known).sum())
        if known.any():self.sink(ArchiveBatch(self.sequence,frame,x[known].copy(),targets[known].copy()))

    def step(self,sequence,frame,detections,rgb,ir,frame_GT,*,split):
        assert split=='train','Archive supervision is forbidden on Dev/Test'
        if sequence!=self.sequence:
            assert frame==1;self.sequence=sequence;self.history=PastLabelHistory()
            self.statistics=dict(frames=0,proposals=0,eligible_pairs=0,known_pairs=0,
                positive_pairs=0,negative_pairs=0,unknown_pairs=0)
        assert frame==self.history.frame+1 and self.current_labels is None
        labels=current_detection_labels(detections,frame_GT,frame=frame,split=split)
        self.current_labels=labels
        try:
            rows=self.linker.update(sequence,frame,detections,rgb,ir)
            assigned=self.linker.inner.current_proposal_assignments(frame)
            self.history.commit_after_update(frame,{i:int(labels[j])for i,j in assigned.items()})
        finally:self.current_labels=None
        self.statistics['frames']+=1;self.statistics['proposals']+=len(detections)
        return rows
