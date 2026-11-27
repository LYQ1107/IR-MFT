"""Causal logical identity evidence, independent of physical ByteTrack state.

The ledger sees current detector observations and its own past only. Training
origin labels live in a separate caller. No method accepts ground truth.
"""
from dataclasses import dataclass, field
import numpy as np
from tools.mft_candidate_risk_teacher import CandidateTeacher, observations
from tools.mft_candidate_risk_features import _margins
from tools.mft_candidate_risk_model import sigmoid
from tools.mft_tracklet_permanence import unit

FEATURE_NAMES=(
    'RGB_anchor','RGB_adaptive','IR_anchor','IR_adaptive','IR_available',
    'RGB_row_margin','RGB_column_margin','IR_row_margin','IR_column_margin',
    'log_trusted_gap','log_identity_age','log_trusted_support',
    'dx_body','dy_body','distance_body','absolute_log_diagonal_scale',
    'absolute_log_aspect_ratio','distance_per_frame','old_quality','new_quality',
    'new_score','RGB_quality_support','IR_quality_support','cue_disagreement',
    'RGB_anchor_adaptive_agreement','IR_anchor_adaptive_agreement')
IR_FEATURES=tuple(FEATURE_NAMES.index(k)for k in[
    'IR_anchor','IR_adaptive','IR_available','IR_row_margin','IR_column_margin',
    'IR_quality_support','cue_disagreement','IR_anchor_adaptive_agreement'])


def uniqueness_margins(scores,valid):
    """Include an explicit zero-score null match when there is no competitor."""
    row,col=_margins(scores,valid)
    row=np.where(valid&(valid.sum(axis=1,keepdims=True)==1),scores,row)
    col=np.where(valid&(valid.sum(axis=0,keepdims=True)==1),scores,col)
    return row,col


@dataclass
class Memory:
    identity:int
    owner:int
    born:int
    last_owner_seen:int
    seeds:list=field(default_factory=list)
    RGB:object=None
    IR:object=None
    adaptive_RGB:object=None
    adaptive_IR:object=None
    box:object=None
    trusted_frame:int=0
    support:int=0
    quality:float=0.

    @property
    def ready(self):return self.RGB is not None

    def accept(self,frame,obs,*,adaptive=True):
        """Freeze origin after3 reliable views; later trust needs anchor agreement."""
        assert frame>=self.born and frame>self.trusted_frame
        self.last_owner_seen=frame
        if not obs['reliable']:return False
        if not self.ready:
            self.seeds.append((obs['rgb'].copy(),None if obs['ir']is None else obs['ir'].copy()))
            assert len(self.seeds)<=3
            if len(self.seeds)==3:
                self.RGB=unit(sum(v[0]for v in self.seeds));assert self.RGB is not None
                self.IR=unit(sum(v[1]for v in self.seeds))if all(v[1]is not None for v in self.seeds)else None
                self.adaptive_RGB=self.RGB.copy()
                self.adaptive_IR=None if self.IR is None else self.IR.copy()
            seed=True
        else:
            if float(self.RGB@obs['rgb'])<.5:return False
            if adaptive:
                self.adaptive_RGB=unit(.9*self.adaptive_RGB+.1*obs['rgb'])
                self.adaptive_IR=unit(.9*self.adaptive_IR+.1*obs['ir'])if self.adaptive_IR is not None and obs['ir']is not None else None
            seed=False
        self.box=obs['box'].copy();self.trusted_frame=frame
        self.support+=1;self.quality=float(obs['quality'])
        return seed


class Ledger:
    def __init__(self,speed,horizon=1000,adaptive=True):
        assert 0<=speed<=1 and type(horizon)is int and horizon>=25
        self.speed,self.horizon,self.adaptive=speed,horizon,adaptive
        self.frame=0;self.memories={};self.bindings={};self.generated=1_000_000_000

    def graph(self,frame,obs):
        assert frame==self.frame+1
        old=[m for _,m in sorted(self.memories.items())if m.ready and 1<=frame-m.trusted_frame<=self.horizon]
        n,r=len(obs),len(old);x=np.zeros((n,r,len(FEATURE_NAMES)),np.float32)
        eligible=np.zeros((n,r),bool)
        if not n or not r:return x,eligible,tuple(m.identity for m in old)
        boxes=np.asarray([v['box']for v in obs],np.float64)
        past=np.asarray([m.box for m in old],np.float64)
        size=np.maximum(boxes[:,2:]-boxes[:,:2],1.)
        previous=np.maximum(past[:,2:]-past[:,:2],1.)
        diagonal=np.linalg.norm(previous,axis=1)
        delta=((boxes[:,:2]+boxes[:,2:])[:,None,:]-(past[:,:2]+past[:,2:])[None,:,:])/2/diagonal[None,:,None]
        distance=np.linalg.norm(delta,axis=2)
        scale=np.linalg.norm(size,axis=1)[:,None]/diagonal[None,:]
        gaps=np.asarray([frame-m.trusted_frame for m in old],np.float64)
        eligible=(scale>=.25)&(scale<=4)&(distance<=.5+self.speed*gaps[None,:])
        rgb=np.asarray([o['rgb']for o in obs],np.float32)
        ar=rgb@np.asarray([m.RGB for m in old],np.float32).T
        er=rgb@np.asarray([m.adaptive_RGB for m in old],np.float32).T
        have=np.asarray([o['ir']is not None for o in obs])[:,None]&np.asarray([m.IR is not None and m.adaptive_IR is not None for m in old])[None,:]
        ai=np.zeros_like(ar);ei=np.zeros_like(ar)
        if have.any():
            shape=rgb.shape[1]
            ir=np.asarray([o['ir']if o['ir']is not None else np.zeros(shape,np.float32)for o in obs],np.float32)
            ai=ir@np.asarray([m.IR if m.IR is not None else np.zeros(shape,np.float32)for m in old],np.float32).T
            ei=ir@np.asarray([m.adaptive_IR if m.adaptive_IR is not None else np.zeros(shape,np.float32)for m in old],np.float32).T
        ar,er,ai,ei=[np.clip(v,-1,1)for v in [ar,er,ai,ei]]
        qr=np.asarray([m.quality for m in old],np.float64)[None,:]
        qn=np.asarray([o['quality']for o in obs],np.float64)[:,None]
        support=np.sqrt(qr*qn)
        raw=[ar,er,ai,ei,have,*uniqueness_margins(ar,eligible),*uniqueness_margins(ai,eligible&have),
            np.log1p(gaps)[None,:],np.log1p([frame-m.born for m in old])[None,:],
            np.log1p([m.support for m in old])[None,:],np.clip(delta[...,0],-8,8),np.clip(delta[...,1],-8,8),
            np.minimum(distance,8),np.abs(np.log(scale)),
            np.abs(np.log((size[:,0]/size[:,1])[:,None]/(previous[:,0]/previous[:,1])[None,:])),
            distance/gaps[None,:],qr,qn,np.asarray([o['score']for o in obs])[:,None],
            ar*support,ai*support*have,np.abs(ar-ai)*have,
            np.asarray([m.RGB@m.adaptive_RGB for m in old])[None,:],
            np.asarray([m.IR@m.adaptive_IR if m.IR is not None and m.adaptive_IR is not None else 0. for m in old])[None,:]]
        assert len(raw)==len(FEATURE_NAMES)
        for j,v in enumerate(raw):x[...,j]=v
        assert np.isfinite(x).all()
        return x,eligible,tuple(m.identity for m in old)

    def new_identity(self):
        self.generated+=1
        assert self.generated not in self.memories
        return self.generated

    def commit(self,frame,obs,assignments,live):
        assert frame==self.frame+1 and len(set(assignments.values()))==len(assignments)
        seeds=[]
        for internal,index in sorted(assignments.items()):
            root=self.bindings.setdefault(internal,internal)
            if root not in self.memories:self.memories[root]=Memory(root,internal,frame,frame)
            memory=self.memories[root];memory.owner=internal
            if memory.accept(frame,obs[index],adaptive=self.adaptive):seeds.append((root,index))
        self.memories={i:m for i,m in self.memories.items()if m.owner in live or frame-m.last_owner_seen<=self.horizon}
        self.bindings={i:r for i,r in self.bindings.items()if i in live or r in self.memories}
        self.frame=frame
        return seeds

    @property
    def descriptor_bytes(self):
        return sum(v.nbytes for m in self.memories.values()for v in
            [m.RGB,m.IR,m.adaptive_RGB,m.adaptive_IR,*[a for pair in m.seeds for a in pair]]if v is not None)


class LedgerLogistic:
    def __init__(self,mean,scale,kind):
        assert kind in ['RGB','paired']
        self.kind=kind;self.mean=np.asarray(mean,np.float64).copy();self.scale=np.asarray(scale,np.float64).copy()
        assert self.mean.shape==self.scale.shape==(len(FEATURE_NAMES),)and np.isfinite(self.mean).all()
        assert np.isfinite(self.scale).all()and(self.scale>0).all()
        if kind=='RGB':self.mean[list(IR_FEATURES)]=0;self.scale[list(IR_FEATURES)]=1
        self.coefficients=np.zeros(len(FEATURE_NAMES)+1,np.float64)

    def design(self,x):
        a=np.asarray(x,np.float64).copy()
        assert a.ndim==2 and a.shape[1]==len(FEATURE_NAMES)and np.isfinite(a).all()
        if self.kind=='RGB':a[:,IR_FEATURES]=0
        return np.column_stack([np.clip((a-self.mean)/self.scale,-8,8),np.ones(len(a))])

    def probabilities(self,x):return sigmoid(self.design(x)@self.coefficients)

    def to_dict(self):
        return dict(feature_names=list(FEATURE_NAMES),mean=self.mean.tolist(),scale=self.scale.tolist(),
            coefficients=self.coefficients.tolist(),kind=self.kind,seed=0,clip=8.)

    @classmethod
    def from_dict(cls,v):
        assert v['feature_names']==list(FEATURE_NAMES)and v['seed']==0 and v['clip']==8.
        m=cls(v['mean'],v['scale'],v['kind']);m.coefficients=np.asarray(v['coefficients'],np.float64)
        assert m.coefficients.shape==(len(FEATURE_NAMES)+1,)and np.isfinite(m.coefficients).all()
        return m


class LedgerTracker:
    """Delayed, exclusive remapping; physical geometry is never modified."""
    def __init__(self,speed,*,kind='off',models=None,thresholds=None,quantile='q99',
                 adaptive=True,shape=False,observer=None,intervention=None):
        assert kind in ['off','RGB','critic']and intervention in [None,'missing','roll']
        self.speed,self.kind,self.models,self.thresholds=speed,kind,models,thresholds
        self.quantile,self.adaptive,self.shape=quantile,adaptive,shape
        self.observer,self.intervention=observer,intervention;self.sequence=None

    def update(self,sequence,frame,detections,rgb,ir=None):
        if self.sequence!=sequence:
            assert frame==1;self.sequence=sequence;self.ledger=Ledger(self.speed,adaptive=self.adaptive)
            self.teacher=CandidateTeacher(self.speed);self.teacher.reset(sequence);self.pending={}
            self.statistics=dict(frames=0,proposals=0,eligible_pairs=0,ready_roots_peak=0,
                descriptor_bytes_peak=0,rebind_proposals=0,IR_vetoes=0,rebindings=0,
                displaced_owners=0,blocked_occupied_roots=0,shape_rows=0)
        if self.intervention=='missing':ir=None
        elif self.intervention=='roll'and ir is not None:ir=np.roll(ir,1,axis=0).copy()
        obs=observations(detections,rgb,ir)
        for o,d in zip(obs,detections):o['score']=float(d[4])
        x,eligible,roots=self.ledger.graph(frame,obs)
        if self.observer is not None:self.observer(x,eligible,roots,frame)
        rows=self.teacher.update(sequence,frame,detections,rgb,ir)
        assignments=self.teacher.current_proposal_assignments(frame)
        # Binding defaults are materialized before remapping, without descriptors.
        for internal in assignments:self.ledger.bindings.setdefault(internal,internal)
        if self.kind!='off'and eligible.any():self.remap(frame,x,eligible,roots,assignments,obs)
        live={int(t.track_id)for t in self.teacher.core.tracked_stracks+self.teacher.core.lost_stracks}
        self.last_seed_events=self.ledger.commit(frame,obs,assignments,live)
        out=[];tracks={int(t.track_id):t for t in self.teacher.core.tracked_stracks}
        for f,box,internal,score in rows:
            if self.shape:
                measured=tracks[internal].current_measurement
                if measured[2]*measured[3]>100:
                    measured=measured.copy();measured[:2]=box[:2]+box[2:]/2-measured[2:]/2
                    box=measured;self.statistics['shape_rows']+=1
            out.append((f,box,self.ledger.bindings[internal],score))
        assert len({r[2]for r in out})==len(out),'A logical ID cannot be emitted twice in one frame'
        self.statistics['frames']+=1;self.statistics['proposals']+=len(detections)
        self.statistics['eligible_pairs']+=int(eligible.sum())
        self.statistics['ready_roots_peak']=max(self.statistics['ready_roots_peak'],len(roots))
        self.statistics['descriptor_bytes_peak']=max(self.statistics['descriptor_bytes_peak'],self.ledger.descriptor_bytes)
        return out

    def remap(self,frame,x,eligible,roots,assignments,obs):
        p=np.zeros(eligible.shape,np.float64);p[eligible]=self.models['RGB'].probabilities(x[eligible])
        row,col=uniqueness_margins(p,eligible);scored=np.where(eligible,p,-np.inf)
        threshold=self.thresholds['RGB'][self.quantile]
        paired=None
        if self.kind=='critic':
            paired=np.zeros_like(p);paired[eligible]=self.models['paired'].probabilities(x[eligible])
        current={self.ledger.bindings[t]:t for t in assignments}
        assert len(current)==len(assignments)
        proposals={};pending={}
        for internal,index in sorted(assignments.items()):
            if not len(roots)or not eligible[index].any()or not obs[index]['reliable']:continue
            j=int(np.argmax(scored[index]));root=roots[j]
            if root==self.ledger.bindings[internal]or not(p[index,j]>threshold and row[index,j]>=.05 and col[index,j]>=.05):continue
            self.statistics['rebind_proposals']+=1
            # Weak/missing IR does not dilute RGB. Only a strongly reliable
            # contrary paired posterior may veto; it never creates proposals.
            if paired is not None and x[index,j,FEATURE_NAMES.index('IR_available')]>0 and paired[index,j]<self.thresholds['paired']['veto']:
                self.statistics['IR_vetoes']+=1;continue
            old=self.pending.get(internal);count=old[2]+1 if old is not None and old[:2]==(root,frame-1)else 1
            pending[internal]=(root,frame,count)
            if count>=3:proposals[internal]=(root,j,index)
        self.pending=pending
        # A proposed root can displace a current owner only if its own current
        # appearance is confidently contradictory. Otherwise preserve binding.
        accepted={}
        for internal,(root,j,index)in proposals.items():
            owner=current.get(root)
            if owner is not None and owner!=internal and owner not in proposals:
                ownindex=assignments[owner]
                if not eligible[ownindex,j]or p[ownindex,j]>.5:
                    self.statistics['blocked_occupied_roots']+=1;continue
            accepted[internal]=(root,j,index)
        # Rejection can invalidate a dependency in a longer permutation.
        changed=True
        while changed:
            changed=False
            for internal,(root,j,index)in list(accepted.items()):
                owner=current.get(root)
                if owner is not None and owner!=internal and owner not in accepted and p[assignments[owner],j]>.5:
                    del accepted[internal];changed=True
        destinations={v[0]for v in accepted.values()}
        assert len(destinations)==len(accepted)
        # Detach all stale bindings (including lost physical owners) first.
        for owner,root in list(self.ledger.bindings.items()):
            if root in destinations and owner not in accepted:
                self.ledger.bindings[owner]=self.ledger.new_identity()
                self.statistics['displaced_owners']+=1
        for internal,(root,_,_)in accepted.items():
            self.ledger.bindings[internal]=root;self.pending.pop(internal,None)
            self.statistics['rebindings']+=1
        assert len(set(self.ledger.bindings.values()))==len(self.ledger.bindings)
