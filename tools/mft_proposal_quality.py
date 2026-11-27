"""Current-frame crop quality and independent causal proposal confirmation.

No GT enters these interfaces. Only uncertain detector scores may be promoted;
detector boxes/order are preserved. Confirmation sees original scores only.
"""
from types import SimpleNamespace
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker,STrack
from yolox.tracker import matching
from tools.mft_candidate_risk_teacher import normalized_descriptors
from tools.mft_candidate_risk_model import sigmoid

FEATURE_NAMES=('score','logit_score','log_area','log_aspect','max_other_IoU',
    'log_nearby_count','quality','RGB_foreground_margin','IR_foreground_margin',
    'IR_available','cue_disagreement')
IR_FEATURES=(8,9,10)


class ProposalTrack(STrack):
    def __init__(self,box,score,observation,counter):
        super().__init__(box,score);self.observation,self.counter=observation,counter
        self.proposal_index=None;self.measurement_frame=0;self.streak=0

    def next_id(self):self.counter[0]+=1;return self.counter[0]

    def accept(self,obs):
        assert obs is not None
        self.streak=self.streak+1 if self.measurement_frame==self.frame_id-1 else 1
        self.measurement_frame=self.frame_id;self.proposal_index=int(obs['proposal_index'])

    def activate(self,kf,frame):super().activate(kf,frame);self.accept(self.observation)
    def update(self,new,frame):super().update(new,frame);self.accept(new.observation)
    def re_activate(self,new,frame,new_id=False):super().re_activate(new,frame,new_id);self.accept(new.observation)


class ProposalConfirmation:
    """Low-threshold geometric observer; never receives promoted scores."""
    def __init__(self):self.sequence=None

    def update(self,sequence,frame,detections):
        if sequence!=self.sequence:
            assert frame==1;self.sequence=sequence;counter=[0]
            args=SimpleNamespace(track_thresh=.1,track_buffer=30,match_thresh=.9,mot20=True)
            self.core=BYTETracker(args,frame_rate=25,track_factory=lambda b,s,o:ProposalTrack(b,s,o,counter))
        assert frame==self.core.frame_id+1
        d=np.asarray(detections,np.float32)
        assert d.ndim==2 and d.shape[1]==5 and np.isfinite(d).all()
        self.core.update(d.copy(),(1,1),(1,1),evidence=[dict(proposal_index=i)for i in range(len(d))])
        result=np.zeros(len(d),np.int32);seen=set()
        for t in self.core.tracked_stracks:
            if t.measurement_frame==frame:
                assert t.proposal_index not in seen;seen.add(t.proposal_index)
                result[t.proposal_index]=t.streak
        for t in self.core.tracked_stracks+self.core.lost_stracks+self.core.removed_stracks:t.observation=None
        return result


class ForegroundView:
    def __init__(self,center,direction):
        self.center=np.asarray(center,np.float64).copy();self.direction=np.asarray(direction,np.float64).copy()
        assert self.center.shape==self.direction.shape==(512,)
        assert np.isfinite(self.center).all()and np.isfinite(self.direction).all()

    def scores(self,values):
        x=normalized_descriptors(values,len(values)).astype(np.float64)
        return (x-self.center)@self.direction

    def to_dict(self):return dict(center=self.center.tolist(),direction=self.direction.tolist())
    @classmethod
    def from_dict(cls,v):return cls(v['center'],v['direction'])


def geometry_features(detections):
    d=np.asarray(detections,np.float32);n=len(d)
    assert d.shape==(n,5)and np.isfinite(d).all()
    x=np.zeros((n,7),np.float32)
    if not n:return x
    assert ((d[:,2:4]-d[:,:2])>0).all()
    score=np.clip(d[:,4].astype(np.float64),1e-5,1-1e-5)
    size=d[:,2:4]-d[:,:2];center=(d[:,:2]+d[:,2:4])/2
    overlap=matching.ious(d[:,:4],d[:,:4]);np.fill_diagonal(overlap,0)
    maximum=overlap.max(axis=1)
    diagonal=np.linalg.norm(size,axis=1)
    distance=np.linalg.norm(center[:,None,:]-center[None,:,:],axis=2)/diagonal[:,None]
    nearby=(distance<=2).sum(axis=1)-1
    x[:]=np.column_stack([score,np.log(score/(1-score)),np.log(size.prod(axis=1)),
        np.log(size[:,0]/size[:,1]),maximum,np.log1p(nearby),score*(1-maximum)])
    assert np.isfinite(x).all()
    return x


def current_features(detections,rgb,ir,views):
    geometry=geometry_features(detections);n=len(geometry)
    if not n:return np.empty((0,len(FEATURE_NAMES)),np.float32)
    r=views['RGB'].scores(rgb);available=ir is not None
    i=views['IR'].scores(ir)if available else np.zeros(n)
    x=np.column_stack([geometry,r,i,np.full(n,float(available)),abs(r-i)if available else np.zeros(n)]).astype(np.float32)
    assert np.isfinite(x).all();return x


def prepare_kind(x,kind):
    a=np.asarray(x,np.float64).copy();assert a.ndim==2 and a.shape[1]==len(FEATURE_NAMES)
    if kind=='RGB':a[:,IR_FEATURES]=0
    elif kind=='duplicatedRGB':a[:,8]=a[:,7];a[:,9]=1;a[:,10]=0
    else:assert kind=='paired'
    return a


class QualityHead:
    def __init__(self,mean,scale,kind):
        assert kind in ['RGB','paired','duplicatedRGB'];self.kind=kind
        self.mean=np.asarray(mean,np.float64).copy();self.scale=np.asarray(scale,np.float64).copy()
        assert self.mean.shape==self.scale.shape==(len(FEATURE_NAMES),)
        assert np.isfinite(self.mean).all()and np.isfinite(self.scale).all()and(self.scale>0).all()
        if kind=='RGB':self.mean[list(IR_FEATURES)]=0;self.scale[list(IR_FEATURES)]=1
        self.coefficients=np.zeros(len(FEATURE_NAMES)+1,np.float64)

    def design(self,x):
        a=prepare_kind(x,self.kind)
        return np.column_stack([np.clip((a-self.mean)/self.scale,-8,8),np.ones(len(a))])

    def probabilities(self,x):return sigmoid(self.design(x)@self.coefficients)

    def to_dict(self):
        return dict(feature_names=list(FEATURE_NAMES),kind=self.kind,mean=self.mean.tolist(),scale=self.scale.tolist(),
            coefficients=self.coefficients.tolist(),seed=0,clip=8.)

    @classmethod
    def from_dict(cls,v):
        assert v['feature_names']==list(FEATURE_NAMES)and v['seed']==0 and v['clip']==8.
        m=cls(v['mean'],v['scale'],v['kind']);m.coefficients=np.asarray(v['coefficients'],np.float64)
        assert m.coefficients.shape==(len(FEATURE_NAMES)+1,)and np.isfinite(m.coefficients).all()
        return m


def promoted_scores(detections,p,streak,threshold,confirmation):
    d=np.asarray(detections,np.float32);p=np.asarray(p,np.float64);streak=np.asarray(streak)
    assert p.shape==streak.shape==(len(d),)and 0<=threshold<=1 and confirmation in [0,3]
    assert np.isfinite(p).all()and((p>=0)&(p<=1)).all()
    selected=(d[:,4]>.3)&(d[:,4]<=.7)&(p>threshold)&(streak>=confirmation)
    result=d.copy();result[selected,4]=.75
    assert np.array_equal(result[:,:4],d[:,:4])and np.array_equal(result[~selected],d[~selected])
    # OriginalByteTrack det_thresh=.6+.1=.7: .75 permits actual births.
    assert not selected.any()or(result[selected,4]>.7).all()
    return result,selected


class ProposalQualityTracker:
    def __init__(self,case,views,heads,thresholds,parent):
        self.case=dict(case);self.views,self.heads,self.thresholds=views,heads,thresholds
        self.parent=parent;self.sequence=None;self.confirmation=ProposalConfirmation()

    def update(self,sequence,frame,detections,rgb,ir=None):
        if self.sequence!=sequence:
            assert frame==1;self.sequence=sequence
            self.statistics=dict(frames=0,proposals=0,eligible_uncertain=0,confirmed_uncertain=0,promoted=0,
                RGB_fallback_frames=0,RGB_fallback_proposals=0)
            import hashlib
            self.calibrated_packet_hash=hashlib.sha256()
        d=np.asarray(detections,np.float32)
        mode=self.case.get('intervention')
        if mode=='missing':ir=None
        elif mode=='roll':ir=np.roll(ir,1,axis=0).copy()if ir is not None else None
        else:assert mode is None
        required=self.case['confirmation']
        streak=self.confirmation.update(sequence,frame,d)if required else np.full(len(d),3,np.int32)
        kind=self.case['kind']
        if kind=='unconditional':p=np.ones(len(d));threshold=0.
        else:
            if kind=='paired'and ir is None:
                kind='RGB';self.statistics['RGB_fallback_frames']+=1;self.statistics['RGB_fallback_proposals']+=len(d)
            x=current_features(d,rgb,ir,self.views);p=self.heads[kind].probabilities(x)
            condition=self.case.get('calibration_condition','confirmed'if required else'all')
            assert condition in ['all','confirmed']
            threshold=self.thresholds[kind][self.case['quantile']][condition]
        changed,selected=promoted_scores(d,p,streak,threshold,required)
        self.calibrated_packet_hash.update(np.asarray([frame,len(d)],dtype='<i8').tobytes())
        self.calibrated_packet_hash.update(np.asarray(changed,dtype='<f4').tobytes())
        self.statistics['frames']+=1;self.statistics['proposals']+=len(d)
        weak=(d[:,4]>.3)&(d[:,4]<=.7)
        self.statistics['eligible_uncertain']+=int(weak.sum());self.statistics['confirmed_uncertain']+=int((weak&(streak>=3)).sum())
        self.statistics['promoted']+=int(selected.sum())
        return self.parent.update(sequence,frame,changed,rgb,ir)
