from types import SimpleNamespace
import numpy as np
import torch

from cfp.association import AppearanceTracker, IdentityTrack
from cfp.utility_association import UtilityAssociationTracker
from yolox.tracker.basetrack import BaseTrack

ARGS=SimpleNamespace(track_thresh=.58,track_buffer=45,match_thresh=.97,mot20=False)

class ConstantUtility(torch.nn.Module):
    def __init__(self,value):super().__init__();self.value=value
    def forward(self,evidence):return torch.full((len(evidence),),self.value)


def trace(tracker,missing=False,future=False):
    result=[]
    for frame in range(4):
        detections=np.array([[frame,0,30+frame,20,.9],[70-frame,0,100-frame,20,.8]],np.float32)
        rgb=np.array([[1.,0.],[0.,1.]],np.float32)
        ir=None if missing else rgb[::-1].copy()
        if future and frame==3:ir=rgb.copy()
        tracks=tracker.update(detections,(1,1),(1,1),rgb,ir)
        result.append([(t.track_id,*t.tlwh.tolist(),float(t.score)) for t in tracks])
    return result


def test_rejected_gate_and_missing_ir_preserve_rgb_full_tracker_trace_exactly():
    BaseTrack._count=0;reference=trace(AppearanceTracker(ARGS,mode='rgb'))
    for missing,value in [(False,-1.),(True,100.)]:
        BaseTrack._count=0
        tracker=UtilityAssociationTracker(ARGS,ConstantUtility(value))
        assert trace(tracker,missing)==reference
        assert tracker.utility_counts['accepted_rows']==0
        assert (tracker.utility_counts['missing_ir_rows'] if missing else tracker.utility_counts['rejected_rows'])>0


def test_future_branch_change_preserves_prefix_and_zero_weight_is_exact_rgb():
    outputs=[]
    for future in [False,True]:
        BaseTrack._count=0
        outputs.append(trace(UtilityAssociationTracker(ARGS,ConstantUtility(1.)),future=future))
    assert outputs[0][:3]==outputs[1][:3]
    BaseTrack._count=0;reference=trace(AppearanceTracker(ARGS,appearance_weight=0))
    BaseTrack._count=0
    assert trace(UtilityAssociationTracker(ARGS,ConstantUtility(1.),appearance_weight=0))==reference


def test_rgb_distance_boundary_is_fixed_before_ir_and_keeps_original_precision():
    x=np.nextafter(np.float32(.3),np.float32(0))
    y=np.nextafter(np.float32(np.sqrt(1-x*x)),np.float32(np.inf))
    candidate=np.array([x,y],np.float32)
    track=IdentityTrack([0,0,30,20],.9,np.array([1.,0.],np.float32),np.array([1.,0.],np.float32))
    track.memory_rgb=track.memory_ir=np.array([1.,0.],np.float32)
    track.last_identity_frame=1
    detection=IdentityTrack([0,0,30,20],.9,candidate,np.array([1.,0.],np.float32))
    assert (1-track.memory_rgb@detection.current_rgb)/2>.35
    reference=AppearanceTracker(ARGS);reference.frame_id=2
    tracker=UtilityAssociationTracker(ARGS,ConstantUtility(1.));tracker.frame_id=2
    assert np.array_equal(reference.association_cost([track],[detection]),tracker.association_cost([track],[detection]))
    assert tracker.utility_counts['evaluated_rows']==0


def test_partial_missing_ir_preserves_that_edge_exactly_in_an_accepted_row():
    rgb=np.array([.6,.8],np.float32)
    track=IdentityTrack([0,0,30,20],.91,rgb,np.array([1.,0.],np.float32))
    track.memory_rgb=rgb;track.memory_ir=np.array([1.,0.],np.float32);track.last_identity_frame=1
    detections=[IdentityTrack([1,0,30,20],.83,np.array([1.,0.],np.float32),None),
                IdentityTrack([2,0,30,20],.86,rgb,np.array([0.,1.],np.float32))]
    reference=AppearanceTracker(ARGS,appearance_weight=.3);reference.frame_id=2
    tracker=UtilityAssociationTracker(ARGS,ConstantUtility(1.),appearance_weight=.3);tracker.frame_id=2
    baseline=reference.association_cost([track],detections)
    mixed=tracker.association_cost([track],detections)
    assert mixed[0,0]==baseline[0,0]  # Available IR on another edge must not round this RGB cost.
    assert mixed[0,1]!=baseline[0,1]
    assert tracker.utility_counts['accepted_rows']==1


def test_uncertainty_control_rejects_equal_information_and_responds_to_entropy():
    track=IdentityTrack([0,0,30,20],.9,np.array([1.,0.],np.float32),np.array([1.,0.],np.float32))
    track.memory_rgb=track.memory_ir=np.array([1.,0.],np.float32);track.last_identity_frame=1;track.is_activated=True
    rgb=np.array([.6,.8],np.float32)
    detections=[IdentityTrack([0,0,30,20],.9,rgb,rgb),IdentityTrack([5,0,30,20],.9,rgb,rgb)]
    baseline=AppearanceTracker(ARGS);baseline.frame_id=2
    tracker=UtilityAssociationTracker(ARGS,kind='uncertainty');tracker.frame_id=2;tracker.candidate_context_version=2
    assert np.array_equal(tracker.association_cost([track],detections),baseline.association_cost([track],detections))
    assert tracker.utility_counts['rejected_rows']==1
    detections[0].current_ir=np.array([1.,0.],np.float32);detections[1].current_ir=np.array([0.,1.],np.float32)
    mixed=tracker.association_cost([track],detections)
    assert tracker.utility_counts['accepted_rows']==1 and tracker.utility_counts['weight_sum']>0
    assert not np.array_equal(mixed,baseline.association_cost([track],detections))
