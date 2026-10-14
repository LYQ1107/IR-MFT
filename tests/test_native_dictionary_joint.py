from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from cfp.native_dictionary_joint import native_joint_inputs, install_dictionary_joint


def fixture():
    value=np.ones(384,np.float32)/np.sqrt(384)
    tracks=[SimpleNamespace(identity_history=[(2,value,None),(3,value,None),(4,value,None)],start_frame=1,end_frame=4,state=1,
                            is_activated=True,tlbr=np.array([i*20,0,i*20+10,10],float)) for i in range(2)]
    detections=[SimpleNamespace(current_rgb=value,score=.8,tlbr=np.array([0,0,10,10],float)),
                SimpleNamespace(current_rgb=None,score=.8,tlbr=np.array([20,0,30,10],float))]
    return tracks,detections


def test_native_inputs_reject_future_and_use_only_actual_observed_fields():
    tracks,detections=fixture();data=native_joint_inputs(tracks,detections,5,512,np.random.default_rng(0))
    assert data['valid'].tolist()==[[True,True,True,False]]*2
    assert data['detection_valid'].tolist()==[True,False]
    assert not ({'labels','physical_identity','teacher','target'} & data.keys())
    tracks[0].identity_history.append((5,np.ones(384,np.float32),None))
    with pytest.raises(ValueError,match='strictly-past'):native_joint_inputs(tracks,detections,5,512,np.random.default_rng(0))


def test_unextractable_current_crop_retains_original_geometry_decisions():
    class Base:
        args=SimpleNamespace(track_thresh=.58);frame_id=5
        def association_cost(self,tracks,detections):return np.array([[.3,.7],[.8,.4]])
    class Model(nn.Module):
        capacity=512
        def forward(self,**inputs):return dict(logits=torch.tensor([[10.,-10.,0.],[-torch.inf,-torch.inf,0.]]))
    tracker=install_dictionary_joint(Base(),Model());tracks,detections=fixture()
    costs=tracker.association_cost(tracks,detections)
    assert costs[0,0]==0 and costs[:,1].tolist()==[.7,.4]
