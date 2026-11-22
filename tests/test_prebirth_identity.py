import numpy as np
import torch

from yolox.tracker.association_calibrator import AssociationCalibrator
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.prebirth_identity_tracker import PrebirthStrongRGBTracker
from yolox.tracker.strong_rgb_tracker import StrongRGBForegroundTracker
from test_calibrated_tracker import data_with_descriptors
from test_foreground_tracker import with_transport,assert_same


def tracker(**config):
    return PrebirthStrongRGBTracker(config,model=AssociationCalibrator(base_dimensions=26),frozen_RGB=AssociationCalibrator(),cue='ir')


def observations_at(frames,total):
    boxes=[];offsets=[0]
    for frame in range(1,total+1):
        boxes.extend(frames.get(frame,[]));offsets.append(len(boxes))
    data=with_transport(boxes,[.95]*len(boxes),offsets,zero=True)
    for key in ['embedding','rgb_mask_embedding','ir_mask_embedding','rgb_hog','ir_hog','color']:
        data[key]=np.ones((len(boxes),3),np.float32)/np.sqrt(3)
    data['quality']=np.ones((len(boxes),2),np.float32)
    return data


def test_recovery_suppresses_only_a_never_emitted_new_ID_and_resets_stale_position():
    data=observations_at({1:[[20,20,50,40]],3:[[120,20,150,40]]},3)
    BaseTrack._count=0;owner=tracker(prebirth_recovery=True,identity_retention=25)
    owner.recovery_scores=lambda frame,context,data:(np.array([.99]),np.array([.5]))
    first=owner.update(1,data);saved=first[0][1].copy()
    assert owner.update(2,data)==[]
    result=owner.update(3,data)
    assert [row[2] for row in result]==[1]
    np.testing.assert_allclose(result[0][1],[120,20,30,20],rtol=0,atol=1e-6)
    np.testing.assert_array_equal(first[0][1],saved)
    assert owner.statistics['prebirth_recoveries']==1
    assert owner.statistics['suppressed_unemitted_IDs']==1
    assert all(t.track_id!=1 for t in owner.inner.removed_stracks)


def test_a_previously_emitted_current_identity_can_never_be_reassigned():
    data=observations_at({1:[[20,20,50,40],[120,20,150,40]],2:[[120,20,150,40]]},2)
    BaseTrack._count=0;owner=tracker(prebirth_recovery=True,identity_retention=128)
    owner.recovery_scores=lambda frame,context,data:(np.array([.99]),np.array([.5]))
    owner.update(1,data);result=owner.update(2,data)
    assert [row[2] for row in result]==[2]
    assert owner.statistics['prebirth_recoveries']==0


def test_a_newly_confirmed_identity_is_replaceable_only_before_its_first_returned_output():
    data=observations_at({1:[[20,20,50,40]],2:[[120,20,150,40]],3:[[120,20,150,40]]},3)
    BaseTrack._count=0;owner=tracker(prebirth_recovery=True,identity_retention=25)
    owner.recovery_scores=lambda frame,context,data:(np.array([0 if frame==2 else .99]),np.array([.5]))
    owner.update(1,data);assert owner.update(2,data)==[]
    result=owner.update(3,data)
    assert [row[2] for row in result]==[1]
    assert owner.emitted_identities=={1}


def test_archive_extension_is_explicit_and_ambiguous_old_identity_reuse_abstains():
    for retention in [25,128]:
        data=observations_at({1:[[20,20,50,40]],29:[[120,20,150,40]]},29)
        BaseTrack._count=0;owner=tracker(prebirth_recovery=True,identity_retention=retention)
        owner.recovery_scores=lambda frame,context,data:(np.array([.99]),np.array([.5]))
        for frame in range(1,30):result=owner.update(frame,data)
        assert bool(result)==(retention==128)
        assert owner.statistics['prebirth_long_recoveries']==int(retention==128)
    data=observations_at({1:[[20,20,50,40],[70,20,100,40]],3:[[120,20,150,40]]},3)
    BaseTrack._count=0;owner=tracker(prebirth_recovery=True,identity_retention=128)
    owner.recovery_scores=lambda frame,context,data:(np.array([.99]),np.array([.5]))
    for frame in range(1,4):result=owner.update(frame,data)
    assert result==[] and owner.statistics['prebirth_recoveries']==0


def test_disabled_recovery_keeps_parent_outputs_and_missing_motion_does_not_use_old_position():
    data=data_with_descriptors();RGB=AssociationCalibrator();residual=AssociationCalibrator(base_dimensions=26)
    BaseTrack._count=0;parent=StrongRGBForegroundTracker({},model=residual,frozen_RGB=RGB,cue='ir')
    expected=[parent.update(frame,data) for frame in range(1,4)]
    BaseTrack._count=0;owner=tracker(prebirth_recovery=False)
    assert_same(expected,[owner.update(frame,data) for frame in range(1,4)])
    torch.manual_seed(6)
    with torch.no_grad():owner.model.net[-1].weight.normal_(std=.2)
    before=owner.appearance_scores(3,dict(frame=1,index=0),data)
    data['boxes'][4:]+=np.array([1000,500,1000,500],np.float32)
    after=owner.appearance_scores(3,dict(frame=1,index=0),data)
    np.testing.assert_array_equal(before[0],after[0])
    np.testing.assert_array_equal(before[1],after[1])


def test_future_observations_cannot_change_the_prebirth_output_prefix():
    first=data_with_descriptors()
    second={k:(v.copy() if isinstance(v,np.ndarray) else {name:arr.copy() for name,arr in v.items()}) for k,v in first.items()}
    for key in ['boxes','embedding','rgb_mask_embedding','ir_mask_embedding','rgb_hog','ir_hog','color']:
        second[key][4:]=-10000
    outputs=[]
    for data in [first,second]:
        BaseTrack._count=0;owner=tracker(prebirth_recovery=True,identity_retention=128,clear_identity_anchor=True)
        outputs.append([owner.update(frame,data) for frame in range(1,3)])
    assert_same(*outputs)
