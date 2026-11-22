import numpy as np
import torch

from yolox.tracker.association_calibrator import AssociationCalibrator
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.calibrated_foreground_tracker import CalibratedForegroundTracker
from test_foreground_tracker import crossing_data,assert_same


def data_with_descriptors():
    data=crossing_data();features=np.tile(np.eye(2,dtype=np.float32),(3,1))
    for k in ['embedding','rgb_mask_embedding','ir_mask_embedding','rgb_hog','ir_hog','color']:data[k]=features.copy()
    data['quality']=np.ones((6,2),np.float32)
    return data


def run(data,model,cue,frames):
    BaseTrack._count=0
    tracker=CalibratedForegroundTracker(dict(preserve_distinct_objects=True),model=model,cue=cue)
    return tracker,[tracker.update(f,data) for f in range(1,frames+1)]


def test_zero_signed_increment_keeps_body_identity_with_both_cue_controls():
    data=data_with_descriptors();model=AssociationCalibrator()
    a,oa=run(data,model,'rgb',3);b,ob=run(data,model,'ir',3)
    assert_same(oa,ob)
    centres={r[2]:r[1][0]+r[1][2]/2 for r in oa[1]}
    assert centres[1]>centres[2]
    assert a.statistics['IR_applied_rows']==0 and b.statistics['IR_applied_rows']==0
    del data['ir_hog'];del data['ir_mask_embedding']
    for k in list(data['foreground_transport']):
        if k.startswith('vote_') and ('joint' in k or '_ir' in k):del data['foreground_transport'][k]
    _,without=run(data,model,'rgb',3);assert_same(oa,without)


def test_future_descriptors_and_votes_do_not_change_already_emitted_outputs():
    first=data_with_descriptors()
    second={k:(v.copy() if isinstance(v,np.ndarray) else {name:arr.copy() for name,arr in v.items()}) for k,v in first.items()}
    for k in ['embedding','rgb_mask_embedding','ir_mask_embedding','rgb_hog','ir_hog','color']:
        second[k][4:]=-10000
    for lag in [1,2,4,8]:
        for key in second['foreground_transport']:
            if key.startswith(f'vote_{lag}_'):
                boundary=second['foreground_transport'][f'vote_offsets_{lag}'][2]
                second['foreground_transport'][key][boundary:]=100
    torch.manual_seed(2);model=AssociationCalibrator()
    with torch.no_grad():model.net[-1].weight.normal_(std=.1)
    a,oa=run(first,model,'ir',2);b,ob=run(second,model,'ir',2)
    assert_same(oa,ob);assert a.statistics==b.statistics
