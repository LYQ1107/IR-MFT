"""GT-free coverage proxies and a separate, read-only Dev target adapter.

Foreground-head outputs are uncalibrated confidence proxies. Their sums do not
estimate a fish count or an error probability. This diagnostic never changes
the detector, tracker, birth threshold, or emitted trajectories.
"""
import numpy as np
from yolox.tracker import matching
from tools.mft_proposal_quality import current_features

KINDS=('count','score','RGB','paired','duplicatedRGB','IR_roll','IR_missing')


def coverage_features(d,rgb,ir,views,heads,streak,unused,assigned):
    """Only original current proposals and current/past tracker observations."""
    d=np.asarray(d,np.float32);n=len(d)
    streak=np.asarray(streak);unused=np.asarray(unused,np.int64);assigned=np.asarray(assigned,np.int64)
    assert d.shape==(n,5) and streak.shape==(n,)
    for indices in [unused,assigned]:
        assert indices.ndim==1 and len(np.unique(indices))==len(indices)
        assert not len(indices) or (indices.min()>=0 and indices.max()<n)
    assert not np.intersect1d(unused,assigned).size
    assert not len(unused) or ((d[unused,4]>.6)&(d[unused,4]<.7)).all()
    confirmed=unused[streak[unused]>=3]
    x=current_features(d,rgb,ir,views)
    probabilities={k:heads[k].probabilities(x)for k in ['RGB','paired','duplicatedRGB']}
    if ir is None:probabilities['paired']=probabilities['RGB'].copy()
    rolled=current_features(d,rgb,np.roll(ir,1,axis=0)if ir is not None else None,views)
    probabilities['IR_roll']=heads['paired'].probabilities(rolled)if ir is not None else probabilities['RGB'].copy()
    probabilities.update(IR_missing=probabilities['RGB'].copy(),count=np.ones(n),score=d[:,4].astype(np.float64))
    values={}
    for kind in KINDS:
        p=probabilities[kind];assert p.shape==(n,) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        unmeasured=float(p[confirmed].sum());measured=float(p[assigned].sum())
        values[kind]=dict(uncovered_mass=unmeasured,measured_mass=measured,
            deficit=unmeasured/(unmeasured+measured)if unmeasured+measured>0 else 0.)
    assert values['RGB']==values['IR_missing']
    return confirmed,values


def Dev_analysis_labels(detections,frame_GT,*,frame,split,purpose):
    """Read-only Dev diagnostic labels, never a fitting/supervision interface.

    Same IoU/Hungarian/ignored-region semantics as current_detection_labels;
    deliberately leaves that Train-only API and its guard unchanged.
    """
    assert split=='dev' and purpose=='read_only_coverage_diagnostic'
    assert type(frame)is int and frame>0
    d=np.asarray(detections,np.float64);gt=np.asarray(frame_GT,np.float64)
    assert d.ndim==2 and d.shape[1]==5 and np.isfinite(d).all()
    assert gt.ndim==2 and gt.shape[1]>=8 and np.isfinite(gt).all()
    assert not len(gt)or(gt[:,0]==frame).all()
    labels=np.full(len(d),-1,np.int64)
    if not len(d):return labels
    valid=(gt[:,6]>0)&(gt[:,7]==1)
    boxes=gt[:,2:6].copy();assert (boxes[:,2:]>0).all();boxes[:,2:]+=boxes[:,:2]
    if valid.any():
        ids=gt[valid,1];assert(ids>0).all()and np.all(ids==ids.astype(np.int64))and len(np.unique(ids))==len(ids)
        overlap=matching.ious(boxes[valid],d[:,:4])
        pairs=matching.linear_assignment(1-overlap,thresh=.5)[0]
        labels[overlap.max(axis=0)<.2]=0
        for i,j in pairs:labels[j]=int(ids[i])
    else:labels[:]=0
    if (~valid).any():labels[matching.ious(boxes[~valid],d[:,:4]).max(axis=0)>=.2]=-1
    return labels


def uncovered_target(d,labels,frame_GT,outputs,confirmed):
    """Join GT only after an entire sequence's GT-free features are complete."""
    gt=np.asarray(frame_GT,np.float64);valid=(gt[:,6]>0)&(gt[:,7]==1)
    boxes=gt[valid,2:6].copy();boxes[:,2:]+=boxes[:,:2];ids=gt[valid,1].astype(np.int64)
    outputs=np.asarray(outputs,np.float32).reshape(-1,4)
    covered=matching.ious(boxes,outputs).max(axis=1)>=.5 if len(outputs)and len(boxes)else np.zeros(len(boxes),bool)
    missed=set(int(v)for v in ids[~covered]);selected=labels[confirmed]
    positive=int((selected>0).sum());background=int((selected==0).sum());unknown=int((selected<0).sum())
    uncovered=sum(int(v)in missed for v in selected if v>0)
    assert uncovered<=positive<=len(confirmed)
    return dict(valid_GT=len(boxes),confirmed_opportunities=len(confirmed),known_positive=positive,
        known_background=background,unknown=unknown,uncovered_known_positive=int(uncovered),
        need_ratio=uncovered/len(boxes)if len(boxes)else 0.)


def threshold_summary(features,targets,threshold):
    features=np.asarray(features,np.float64);targets=np.asarray(targets,np.float64)
    assert features.shape==targets.shape and np.isfinite(features).all()and np.isfinite(targets).all()
    selected=features>=threshold;needed=targets>=.1
    tp=int((selected&needed).sum());fp=int((selected&~needed).sum());fn=int((~selected&needed).sum())
    return dict(frames=len(features),selected_frames=int(selected.sum()),needed_frames=int(needed.sum()),
        TP=tp,FP=fp,FN=fn,TN=int((~selected&~needed).sum()),
        precision=tp/(tp+fp)if tp+fp else None,recall=tp/(tp+fn)if tp+fn else None,
        selected_mean_target=float(targets[selected].mean())if selected.any()else None,
        unselected_mean_target=float(targets[~selected].mean())if (~selected).any()else None)
