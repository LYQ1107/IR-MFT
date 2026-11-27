"""Train-only frozen origin labels; never imported by the online ledger."""
import numpy as np
from tools.mft_candidate_risk_labels import current_detection_labels
from tools.mft_identity_ledger import LedgerTracker


class OriginLabels:
    def __init__(self):self.seeds={};self.labels={};self.frame=0

    def origin(self,identity,frame):
        assert frame==self.frame+1
        return self.labels.get(identity)

    def commit(self,frame,events,labels):
        assert frame==self.frame+1
        for identity,index in events:
            values=self.seeds.setdefault(identity,[]);values.append(int(labels[index]));assert len(values)<=3
            if len(values)==3:self.labels[identity]=values[0]if values[0]>0 and len(set(values))==1 else None
        self.frame=frame


class LedgerCollector:
    def __init__(self,speed,sink):
        self.speed,self.sink=speed,sink;self.sequence=None;self.current_labels=None

    def observe(self,x,eligible,roots,frame):
        assert self.current_labels is not None
        known=np.zeros(eligible.shape,bool);targets=np.zeros(eligible.shape,np.float32)
        for j,root in enumerate(roots):
            origin=self.origins.origin(root,frame)
            if origin is None:continue
            known[:,j]=eligible[:,j]&(self.current_labels>=0)
            targets[:,j]=(self.current_labels==origin).astype(np.float32)
        self.statistics['eligible_pairs']+=int(eligible.sum());self.statistics['known_pairs']+=int(known.sum())
        self.statistics['unknown_pairs']+=int((eligible&~known).sum())
        self.statistics['positive_pairs']+=int(targets[known].sum())
        self.statistics['negative_pairs']+=int(known.sum()-targets[known].sum())
        if known.any():self.sink(self.sequence,frame,x[known].copy(),targets[known].copy())

    def step(self,sequence,frame,d,rgb,ir,gt,*,split):
        assert split=='train','Ledger supervision cannot consume Dev/Test'
        if self.sequence!=sequence:
            assert frame==1;self.sequence=sequence;self.origins=OriginLabels()
            self.tracker=LedgerTracker(self.speed,observer=self.observe)
            self.statistics=dict(frames=0,proposals=0,eligible_pairs=0,known_pairs=0,
                unknown_pairs=0,positive_pairs=0,negative_pairs=0)
        assert frame==self.origins.frame+1 and self.current_labels is None
        labels=current_detection_labels(d,gt,frame=frame,split=split);self.current_labels=labels
        try:
            rows=self.tracker.update(sequence,frame,d,rgb,ir)
            self.origins.commit(frame,self.tracker.last_seed_events,labels)
        finally:self.current_labels=None
        self.statistics['frames']+=1;self.statistics['proposals']+=len(d)
        return rows
