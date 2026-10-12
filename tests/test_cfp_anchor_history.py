from types import SimpleNamespace

import numpy as np
import pytest

from cfp.association import IdentityTrack,AppearanceTracker
from cfp.history import anchor_recent_history
from cfp.predictive_association import PredictiveAssociationTracker
from yolox.tracker.kalman_filter import KalmanFilter
from yolox.tracker.basetrack import BaseTrack


def test_anchor_retains_original_visual_information_after_recent_buffer_is_poisoned():
    track=IdentityTrack([0,0,30,20],.9,np.array([1.,0.],np.float32),None,0)
    track.activate(KalmanFilter(),1)
    track.write_identity(1,.9)
    for frame in range(2,21):
        observation=IdentityTrack([0,0,30,20],.9,np.array([0.,1.],np.float32),None,frame-1)
        track.update(observation,frame);track.write_identity(frame,.9)
    assert len(track.identity_history)==8
    assert all(np.array_equal(rgb,np.array([0.,1.],np.float32)) for _,rgb,_ in track.identity_history)
    assert np.array_equal(track.initial_identity_rgb,np.array([1.,0.],np.float32))
    assert track.initial_identity_frame==1 and track.initial_proposal_index==0
    history=anchor_recent_history(track.identity_history,(1,track.initial_identity_rgb,None))
    assert [r[0] for r in history]==[1,14,15,16,17,18,19,20]
    assert len(history)==8 and all(r[0]<21 for r in history)


def test_anchor_budget_order_and_duplicate_timestamp_guards():
    assert anchor_recent_history([(1,0),(2,1)],(1,0))==[(1,0),(2,1)]
    assert anchor_recent_history([(10,9)],(1,0))==[(1,0),(10,9)]
    with pytest.raises(ValueError):anchor_recent_history([(1,0),(2,1)],(3,2))
    with pytest.raises(ValueError):anchor_recent_history([(2,0),(2,1)],(1,2))


def test_anchor_and_recent_control_use_the_same_candidate_graph():
    track=IdentityTrack([0,0,30,20],.9,np.array([1.,0.],np.float32),None,0)
    track.activate(KalmanFilter(),1)
    track.write_identity(1,.9)
    for frame in range(2,11):
        track.update(IdentityTrack([0,0,30,20],.9,np.array([0.,1.],np.float32),None,frame),frame);track.write_identity(frame,.9)
    detections=[IdentityTrack([0,0,30,20],.9,np.array([1.,0.],np.float32),None),IdentityTrack([1,0,30,20],.9,np.array([0.,1.],np.float32),None)]
    args=SimpleNamespace(track_thresh=.58,track_buffer=45,match_thresh=.97,mot20=False)
    last=PredictiveAssociationTracker(args,memory='last');last.frame_id=11
    anchor=PredictiveAssociationTracker(args,memory='anchor_only',history_view='anchor_recent');anchor.frame_id=11
    recent_costs=last.association_cost([track],detections);anchor_costs=anchor.association_cost([track],detections)
    assert recent_costs[0,0]>anchor_costs[0,0] and recent_costs[0,1]<anchor_costs[0,1]
    assert last.prediction_counts['modified_edges']==anchor.prediction_counts['modified_edges']==2


def test_initial_anchor_survives_native_removed_object_reactivation_window():
    BaseTrack._count=0
    args=SimpleNamespace(track_thresh=.58,track_buffer=45,match_thresh=.97,mot20=False)
    tracker=AppearanceTracker(args,mode='rgb')
    box=np.array([[0,0,30,20,.9]],np.float32)
    tracker.update(box.copy(),(1,1),(1,1),np.array([[1.,0.]],np.float32),proposal_indices=np.array([0]))
    original=tracker.tracked_stracks[0]
    for frame in range(2,40):tracker.update(np.empty((0,5),np.float32),(1,1),(1,1),np.empty((0,2),np.float32))
    assert original in tracker.lost_stracks and original in tracker.removed_stracks
    assert original.initial_identity_frame==1 and not original.identity_history
    returned=tracker.update(box.copy(),(1,1),(1,1),np.array([[0.,1.]],np.float32),proposal_indices=np.array([200]))
    assert returned[0].track_id==original.track_id
    assert original.initial_identity_frame==1 and original.initial_proposal_index==0
    assert np.array_equal(original.initial_identity_rgb,np.array([1.,0.],np.float32))
