"""Streaming Train supervision around an unchanged causal ByteTrack teacher."""
from dataclasses import dataclass
import numpy as np
from tools.mft_candidate_risk_teacher import CandidateTeacher
from tools.mft_candidate_risk_labels import current_detection_labels,PastLabelHistory,edge_targets


@dataclass(frozen=True)
class FrameEdgeBatch:
    sequence:str
    frame:int
    x:np.ndarray
    y:np.ndarray


class TrainEdgeCollector:
    def __init__(self,speed_upper,sink):
        self.sink=sink;self.sequence=None;self.current_labels=None
        self.teacher=CandidateTeacher(speed_upper,self.observe)

    def reset(self,sequence):
        self.sequence=sequence;self.history=PastLabelHistory()
        self.statistics=dict(frames=0,proposals=0,eligible_pairs=0,known_pairs=0,
            positive_pairs=0,negative_pairs=0,unknown_pairs=0,
            known_identity_detections=0,known_background_detections=0,unknown_detections=0)

    def observe(self,x,eligible,track_ids,proposal_indices,frame):
        assert self.current_labels is not None
        prior=[self.history.prior(i,frame)for i in track_ids]
        labels=self.current_labels[np.asarray(proposal_indices,dtype=np.int64)]
        targets,known=edge_targets(prior,labels,eligible)
        self.statistics['eligible_pairs']+=int(eligible.sum())
        self.statistics['known_pairs']+=int(known.sum())
        self.statistics['unknown_pairs']+=int((eligible&~known).sum())
        self.statistics['positive_pairs']+=int(targets[known].sum())
        self.statistics['negative_pairs']+=int(known.sum()-targets[known].sum())
        if known.any():self.sink(FrameEdgeBatch(self.sequence,frame,x[known].copy(),targets[known].copy()))

    def step(self,sequence,frame,detections,rgb,ir,frame_GT,*,split):
        assert split=='train','Supervised collection is forbidden on Dev/Test'
        if sequence!=self.sequence:
            assert frame==1;self.reset(sequence)
        assert frame==self.history.frame+1 and self.current_labels is None
        labels=current_detection_labels(detections,frame_GT,frame=frame,split=split)
        self.current_labels=labels
        try:
            rows=self.teacher.update(sequence,frame,detections,rgb,ir)
            assignments=self.teacher.current_proposal_assignments(frame)
            self.history.commit_after_update(frame,{identity:int(labels[index])for identity,index in assignments.items()})
        finally:self.current_labels=None
        self.statistics['frames']+=1;self.statistics['proposals']+=len(detections)
        self.statistics['known_identity_detections']+=int((labels>0).sum())
        self.statistics['known_background_detections']+=int((labels==0).sum())
        self.statistics['unknown_detections']+=int((labels<0).sum())
        return rows
