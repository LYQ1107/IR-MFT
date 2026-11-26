"""Train-only targets, kept separate from GT-free inference features."""
from collections import Counter
import numpy as np
from yolox.tracker import matching


def current_detection_labels(detections,frame_GT,*,frame,split):
    """Return positive GT ID,0 known background,-1 unknown/ignored."""
    assert split=='train','This supervision interface cannot label Dev/Test for model fitting'
    assert type(frame)is int and frame>0
    detections=np.asarray(detections,np.float64);gt=np.asarray(frame_GT,np.float64)
    assert detections.ndim==2 and detections.shape[1]==5
    assert gt.ndim==2 and gt.shape[1]>=8 and np.isfinite(gt).all()and np.isfinite(detections).all()
    assert not len(gt)or(gt[:,0]==frame).all(),'Only the declared current frame can supply labels'
    labels=np.full(len(detections),-1,np.int64)
    if not len(detections):return labels
    valid=(gt[:,6]>0)&(gt[:,7]==1)
    boxes=gt[:,2:6].copy()
    assert (boxes[:,2:]>0).all()
    boxes[:,2:]+=boxes[:,:2]
    if valid.any():
        ids=gt[valid,1];assert(ids>0).all()and np.all(ids==ids.astype(np.int64))and len(np.unique(ids))==len(ids)
        overlap=matching.ious(boxes[valid],detections[:,:4])
        pairs=matching.linear_assignment(1-overlap,thresh=.5)[0]
        labels[overlap.max(axis=0)<.2]=0
        for i,j in pairs:labels[j]=int(ids[i])
    else:labels[:]=0
    if (~valid).any():
        ignored=matching.ious(boxes[~valid],detections[:,:4])
        labels[ignored.max(axis=0)>=.2]=-1
    return labels


class PastLabelHistory:
    """One sequence per instance; no current-frame label enters a prior."""
    def __init__(self,minimum=3,purity=.8):
        assert type(minimum)is int and minimum>=1 and .5<purity<=1
        self.minimum,self.purity=minimum,purity
        self.frame=0
        self.counts={}

    def prior(self,track_id,current_frame):
        assert current_frame==self.frame+1,'Prior labels must precede the current association'
        counts=self.counts.get(track_id,Counter());total=sum(counts.values())
        if total<self.minimum:return None
        label,count=max(counts.items(),key=lambda p:(p[1],-p[0]))
        return int(label)if label>0 and count/total>=self.purity else None

    def commit_after_update(self,frame,track_labels):
        assert frame==self.frame+1
        for track_id,label in track_labels.items():
            assert type(track_id)is int and track_id>0
            label=int(label)
            if label>=0:self.counts.setdefault(track_id,Counter())[label]+=1
        self.frame=frame


def edge_targets(prior_labels,detection_labels,eligible):
    """Unknown rows/columns do not become false negative training labels."""
    eligible=np.asarray(eligible,bool)
    assert eligible.shape==(len(prior_labels),len(detection_labels))
    targets=np.zeros(eligible.shape,np.float32);known=np.zeros_like(eligible)
    for i,label in enumerate(prior_labels):
        if label is None:continue
        assert label>0
        for j,current in enumerate(detection_labels):
            if eligible[i,j]and current>=0:
                known[i,j]=True;targets[i,j]=float(label==current)
    return targets,known
