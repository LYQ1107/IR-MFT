from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.fusion_memory_features import pool_p3
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.fusion_memory_tracker import (
    FusionMemoryConfig,FusionMemoryTrack,FusionMemoryTracker,FusionObservation,observations,
)


def packet(rgb=((1.,0.),),fused=None,quality=.8,available=True):
    rgb=np.array(rgb,np.float32)
    return dict(rgb=rgb,auxiliary=rgb.copy(),fused=rgb.copy() if fused is None else np.array(fused,np.float32),
        quality=np.full(len(rgb),quality),auxiliary_available=np.full(len(rgb),available,dtype=bool))


def detection(box=(0,0,40,20),score=.9):return np.array([[*box,score]],np.float32)


def snapshot(core):
    return [[(t.track_id,t.state,t.frame_id,t.start_frame,t.tracklet_len,bool(t.is_activated),
              float(t.score),t.mean.copy(),t.covariance.copy()) for t in group]
            for group in [core.tracked_stracks,core.lost_stracks,core.removed_stracks]]


def assert_states_equal(left,right):
    assert len(left)==len(right)
    for a,b in zip(left,right):
        assert len(a)==len(b)
        for x,y in zip(a,b):
            assert x[:7]==y[:7]
            np.testing.assert_array_equal(x[7],y[7]);np.testing.assert_array_equal(x[8],y[8])


class Unreadable:
    def __array__(self,*args,**kwargs):raise AssertionError('Unavailable feature was read')
    def __getitem__(self,key):raise AssertionError('Unavailable feature was read')


def test_zero_weight_preserves_all_byte_states_across_low_scores_loss_and_reactivation():
    args=SimpleNamespace(track_thresh=.6,track_buffer=30,match_thresh=.9,mot20=False)
    frames=[detection(),detection((2,0,42,20),.4),np.empty((0,5),np.float32),
            detection((5,0,45,20)),detection((200,100,240,120)),
            np.array([[8,0,48,20,.8],[202,100,242,120,.8]],np.float32)]
    BaseTrack._count=0;baseline=BYTETracker(args,frame_rate=25);expected=[]
    for data in frames:baseline.update(data.copy(),(1,1),(1,1));expected.append(snapshot(baseline))
    tracker=FusionMemoryTracker(FusionMemoryConfig(appearance_weight=0),args)
    for frame,(data,state) in enumerate(zip(frames,expected),1):
        before=data.copy();tracker.update('video',frame,data,Unreadable())
        np.testing.assert_array_equal(data,before);assert_states_equal(snapshot(tracker.core),state)
        assert all(t.rgb_memory is None for t in tracker.core.tracked_stracks+tracker.core.lost_stracks)
    assert tracker.statistics['primary_cost_calls']==0


def test_bad_quality_and_secondary_stage_observations_freeze_memory_while_age_advances():
    tracker=FusionMemoryTracker();tracker.update('v',1,detection(),packet())
    track=tracker.core.tracked_stracks[0];before=track.fused_memory.copy();last=track.memory_last_frame.copy()
    tracker.update('v',2,detection(),packet(fused=((0,1),),quality=0.))
    np.testing.assert_array_equal(track.fused_memory,before);assert track.memory_last_frame==last and track.memory_age==1
    tracker.update('v',3,detection(score=.4),packet(fused=((0,1),),quality=1.))
    np.testing.assert_array_equal(track.fused_memory,before);assert track.memory_age==2


def test_reliable_update_uses_normalized_reliability_weighted_memory_without_aliasing():
    cfg=FusionMemoryConfig();track=FusionMemoryTrack(np.array([0,0,40,20]),.9,None,cfg,[0],.6)
    first=FusionObservation(np.array([1,0],np.float32),None,np.array([1,0],np.float32),.8,True)
    second=FusionObservation(np.array([0,1],np.float32),None,np.array([0,1],np.float32),.5,True)
    track.accept(first,1);track.accept(second,2)
    expected=np.array([.9,.1],np.float32);expected/=np.linalg.norm(expected)
    np.testing.assert_allclose(track.fused_memory,expected,rtol=1e-6)
    second.fused[:]=7;np.testing.assert_allclose(track.fused_memory,expected,rtol=1e-6)
    assert track.memory_last_frame['fused']==2


def test_max_box_overlap_is_a_proxy_that_can_freeze_ambiguous_observations():
    data=np.repeat(detection(),2,axis=0);p=packet(rgb=((1,0),(0,1)),quality=1.)
    result=observations(data,p,'provided')
    assert all(o.reliability==0 for o in result)


def test_missing_ir_never_reads_stale_arrays_and_matches_rgb_memory_control():
    first=FusionMemoryTracker(FusionMemoryConfig(feature_source='fused'))
    second=FusionMemoryTracker(FusionMemoryConfig(feature_source='rgb'))
    for frame in range(1,5):
        p=dict(rgb=np.array([[1,frame/10]],np.float32),auxiliary_available=np.array([False]),
               auxiliary=Unreadable(),fused=Unreadable(),quality=Unreadable())
        first.update('v',frame,detection((frame,0,frame+40,20)),p)
        second.update('v',frame,detection((frame,0,frame+40,20)),p)
        assert_states_equal(snapshot(first.core),snapshot(second.core))
    assert first.core.tracked_stracks[0].aux_memory is None


def test_ordinary_rgb_control_does_not_read_auxiliary_presence_quality_or_embeddings():
    tracker=FusionMemoryTracker(FusionMemoryConfig(feature_source='rgb',quality_mode='uniform'))
    p=dict(rgb=np.array([[1,0]],np.float32),auxiliary_available=Unreadable(),quality=Unreadable(),
           auxiliary=Unreadable(),fused=Unreadable())
    tracker.update('v',1,detection(),p)
    assert tracker.core.tracked_stracks[0].rgb_memory is not None


def test_rgb_control_with_matched_fusion_quality_does_not_read_auxiliary_embeddings():
    tracker=FusionMemoryTracker(FusionMemoryConfig(feature_source='rgb'))
    p=dict(rgb=np.array([[1,0]],np.float32),auxiliary_available=np.array([True]),quality=np.array([.8]),
           auxiliary=Unreadable(),fused=Unreadable())
    tracker.update('v',1,detection(),p)
    assert tracker.core.tracked_stracks[0].rgb_memory is not None


def test_appearance_never_lowers_costs_opens_far_edges_or_changes_other_stages():
    tracker=FusionMemoryTracker();tracker.update('v',1,detection(),packet())
    past=tracker.core.tracked_stracks
    obs=FusionMemoryTrack(np.array([0,0,40,20]),.9,
        FusionObservation(np.array([0,1],np.float32),None,np.array([0,1],np.float32),.8,True),tracker.config,[0],.6)
    base=np.array([[.2]],np.float64);changed=tracker.cost(past,[obs],base,2,'primary')
    assert changed[0,0]>.2 and changed[0,0]<=.2+tracker.config.appearance_weight
    assert tracker.cost(past,[obs],base,2,'secondary') is base
    assert tracker.cost(past,[obs],base,2,'pending') is base
    far=FusionMemoryTrack(np.array([1000,1000,40,20]),.9,obs.observation,tracker.config,[0],.6)
    outside=np.array([[1.]],np.float64)
    np.testing.assert_array_equal(tracker.cost(past,[far],outside,2,'primary'),outside)


def test_sequence_change_clears_all_memory_and_rejects_future_or_repeated_frames():
    tracker=FusionMemoryTracker();tracker.update('one',1,detection(),packet())
    with pytest.raises(ValueError):tracker.update('one',3,detection(),packet())
    with pytest.raises(ValueError):tracker.update('two',2,detection(),packet())
    tracker.update('two',1,detection(),packet(rgb=((0,1),)))
    assert tracker.core.tracked_stracks[0].track_id==1
    np.testing.assert_array_equal(tracker.core.tracked_stracks[0].fused_memory,[0,1])
    with pytest.raises(ValueError):tracker.update('two',1,detection(),packet())


def test_roi_operator_uses_network_coordinates_and_actual_stride_and_detaches():
    x=(torch.arange(12,dtype=torch.float32)[None,:]+.5).expand(8,12)*8
    y=(torch.arange(8,dtype=torch.float32)[:,None]+.5).expand(8,12)*8
    features=torch.stack([x,y])[None].requires_grad_()
    boxes=np.array([[4,2,20,10]],np.float32);before=boxes.copy()
    pooled,valid=pool_p3(features,boxes,(32,48),(64,96),2.,normalize=False)
    torch.testing.assert_close(pooled,torch.tensor([[24.,12.]]));assert valid.tolist()==[True]
    assert not pooled.requires_grad;np.testing.assert_array_equal(boxes,before)
    with pytest.raises(ValueError):pool_p3(features,boxes,(32,48),(64,96),1.)
    with pytest.raises(ValueError):pool_p3(features[:,:,:4,:6],boxes,(32,48),(64,96),2.)


def test_padded_outside_rois_and_zero_embeddings_cannot_supply_memory():
    features=torch.ones(1,3,8,12)
    result,valid=pool_p3(features,np.array([[60,30,80,40]],np.float32),(32,48),(64,96),2.)
    assert not valid.any() and not result.any()
    result,valid=pool_p3(torch.zeros_like(features),np.array([[1,1,10,10]],np.float32),(32,48),(64,96),2.)
    assert not valid.any() and not result.any()
