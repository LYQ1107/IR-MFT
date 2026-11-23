"""Preserve established global observations while adding crop evidence."""
import numpy as np

from yolox.tracker.foveated_observations import integrate_crops
from yolox.tracker.matching import ious


def intersects_existing(box, existing):
    if not len(existing):
        return False
    overlap=np.maximum(np.minimum(box[None,2:],existing[:,2:])-np.maximum(box[None,:2],existing[:,:2]),0).prod(1)
    area=np.maximum((box[2:]-box[:2]).prod(),1)
    other=np.maximum((existing[:,2:]-existing[:,:2]).prod(1),1)
    # Only added observations are suppressed; already distinct global objects
    # remain distinct even when their boxes overlap heavily.
    return bool(np.any((ious(box[None],existing)[0]>.3)|(overlap/np.minimum(area,other)>.7)))


def priority_union(original, views):
    original=np.asarray(original,np.float32).reshape(-1,5)
    result=original.copy()
    # High-confidence global coordinates AND scores are immutable. A crop may
    # promote an original weak observation without moving its established box.
    for detection,resolution in views:
        detection=np.asarray(detection,np.float32).reshape(-1,5)
        if not len(original) or not len(detection):
            continue
        overlap=ious(original[:,:4],detection[:,:4])
        for column in range(len(detection)):
            row=int(overlap[:,column].argmax())
            if overlap[row,column]>=.5 and original[row,4]<.6:
                result[row,4]=max(result[row,4],detection[column,4])
    candidates=integrate_crops(np.empty((0,5),np.float32),views,refine=True)
    added=[]
    for candidate in sorted(candidates,key=lambda row:-float(row[4])):
        if candidate[4]<.7 or intersects_existing(candidate[:4],original[:,:4]):
            continue
        if added and intersects_existing(candidate[:4],np.asarray(added)[:,:4]):
            continue
        added.append(candidate)
    if added:
        result=np.r_[result,np.asarray(added,np.float32)]
    return result.astype(np.float32),np.r_[np.arange(len(original)),np.full(len(added),-1)].astype(np.int64)
