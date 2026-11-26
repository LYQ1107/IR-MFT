"""Behavioral checks for protected geometry, cue fallback and causal no-op."""
from types import SimpleNamespace
import numpy as np
from tools.mft_ambiguity_reasoning import AmbiguityReasoning,policies
from tools.mft_tracklet_permanence_r2 import TrackletPermanence,cases
from tools.native_pyramid_stream_tracks import IndependentByteTrack,mot_line
from yolox.tracker import matching
from yolox.tracker.basetrack import TrackState

T=dict(RGB_cosine_threshold=.6,IR_cosine_threshold=.4,Train_unknown_speed_upper=.2,
       primary_RGB_cosine_threshold=.5,primary_IR_cosine_threshold=.5)


def objects(rgb,ir,frame=1):
    out=[]
    for a,b in zip(rgb,ir):
        out.append(SimpleNamespace(rgb=np.asarray(a,np.float32),ir=np.asarray(b,np.float32),accepted=3,
            memory_frame=frame,memory_quality=1.,frame_id=frame,state=TrackState.Tracked,
            tlbr=np.array([0,0,20,10],np.float32)))
    return out


def candidates(rgb,ir,available=True):
    return [SimpleNamespace(tlbr=np.array([0,0,20,10],np.float32),observation=dict(rgb=np.asarray(a,np.float32),
        ir=np.asarray(b,np.float32),quality=1.,auxiliary_available=available))for a,b in zip(rgb,ir)]


def test_zero_and_late_only_are_literal_references():
    zero=AmbiguityReasoning(policies()[0],T);late=AmbiguityReasoning(policies()[1],T)
    reference=IndependentByteTrack();old=TrackletPermanence(cases()[2],T)
    for frame in range(1,34):
        present=frame<=3 or frame>=31;d=np.array([[0,0,20,10,.95]],np.float32)if present else np.empty((0,5),np.float32)
        e=np.tile(np.array([[1,0]],np.float32),(len(d),1))
        assert ''.join(map(mot_line,zero.update('video',frame,d)))==''.join(map(mot_line,reference.update(d,frame)))
        assert ''.join(map(mot_line,late.update('video',frame,d,e,None)))==''.join(map(mot_line,old.update('video',frame,d,e,e)))


def test_appearance_reverses_only_ambiguous_pairs_and_protects_clear_pair():
    model=AmbiguityReasoning(policies()[2],T);model.reset('video')
    x=np.eye(3,dtype=np.float32);tracks=objects(x,x);detections=candidates(x[[0,2,1]],x[[0,2,1]])
    base=np.array([[.05,.95,.95],[.95,.45,.48],[.95,.48,.45]],np.float64)
    changed=model.cost(tracks,detections,base,2,'primary')
    pairs=set(map(tuple,matching.linear_assignment(changed,.9)[0]))
    assert pairs=={(0,0),(1,2),(2,1)}and np.all(changed>=base)
    assert changed[0,0]==base[0,0]and model.statistics['primary_assignment_changed_frames']==1


def test_ir_rescue_occurs_only_when_rgb_is_ambiguous():
    ambiguous=np.array([[1,0],[1,0]],np.float32);distinct=np.eye(2,dtype=np.float32)
    tracks=objects(ambiguous,distinct);d=candidates(ambiguous,distinct[[1,0]])
    base=np.array([[.45,.48],[.48,.45]],np.float64)
    rgb=AmbiguityReasoning(policies()[2],T);rgb.reset('v');np.testing.assert_array_equal(rgb.cost(tracks,d,base,2,'primary'),base)
    joint=AmbiguityReasoning(policies()[3],T);joint.reset('v');changed=joint.cost(tracks,d,base,2,'primary')
    assert set(map(tuple,matching.linear_assignment(changed,.9)[0]))=={(0,1),(1,0)}
    assert joint.statistics['IR_rescue_pairs']==2
    missing=AmbiguityReasoning(policies()[8],T);missing.reset('v')
    np.testing.assert_array_equal(missing.cost(tracks,candidates(ambiguous,distinct,False),base,2,'primary'),base)


def test_missing_ir_is_not_read_and_matches_rgb_only():
    class Unreadable:
        def __array__(self,*a,**k):raise AssertionError('Unavailable IR must never be read')
    rgb=AmbiguityReasoning(policies()[4],T);missing=AmbiguityReasoning(policies()[8],T)
    for frame in range(1,9):
        d=np.array([[frame,0,frame+20,10,.95]],np.float32);e=np.array([[1,0]],np.float32)
        assert ''.join(map(mot_line,rgb.update('v',frame,d,e,Unreadable())))==''.join(map(mot_line,missing.update('v',frame,d,e,Unreadable())))


def test_duplicate_rgb_auxiliary_does_not_read_actual_ir_or_count_it_as_ir():
    class Unreadable:
        def __array__(self,*a,**k):raise AssertionError('RGB auxiliary cannot read actual IR')
    model=AmbiguityReasoning(policies()[6],T)
    for frame in range(1,9):
        d=np.array([[frame,0,frame+20,10,.95]],np.float32);e=np.array([[1,0]],np.float32)
        model.update('v',frame,d,e,Unreadable())
    assert model.statistics['IR_rescue_pairs']==0
