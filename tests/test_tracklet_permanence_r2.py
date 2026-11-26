"""Causal identity-link invariants, using artificial proposals/descriptors only."""
import numpy as np
from tools.mft_tracklet_permanence_r2 import TrackletPermanence,cases
from tools.native_pyramid_stream_tracks import IndependentByteTrack,mot_line

THETA=dict(RGB_cosine_threshold=.6,IR_cosine_threshold=.4,Train_unknown_speed_upper=.2)


def inputs(frame,wrong_IR=False):
    present=frame<=3 or frame>=31
    packet=np.array([[0,0,20,10,.95]],np.float32)if present else np.empty((0,5),np.float32)
    rgb=np.array([[1,0]],np.float32)if present else np.empty((0,2),np.float32)
    ir=np.array([[0,1]if wrong_IR and frame>=31 else[1,0]],np.float32)if present else np.empty((0,2),np.float32)
    return packet,rgb,ir


def test_zero_literal_MOT_and_unread_descriptors():
    class Unreadable:
        def __array__(self,*args,**kwargs):raise AssertionError('Zero path cannot read identity features')
    model=TrackletPermanence(cases()[0],THETA);original=IndependentByteTrack()
    for frame in range(1,34):
        packet,_,_=inputs(frame);untouched=packet.copy()
        assert ''.join(map(mot_line,model.update('video',frame,packet,Unreadable(),Unreadable())))==''.join(map(mot_line,original.update(packet,frame)))
        np.testing.assert_array_equal(packet,untouched)


def test_removed_identity_reconnects_without_geometry_changes():
    model=TrackletPermanence(cases()[3],THETA);original=IndependentByteTrack()
    for frame in range(1,34):
        packet,rgb,ir=inputs(frame);rows=model.update('video',frame,packet,rgb,ir);base=original.update(packet,frame)
        assert len(rows)==len(base)
        for actual,expected in zip(rows,base):
            np.testing.assert_array_equal(actual[1],expected[1]);assert actual[0]==expected[0] and actual[3]==expected[3]
        if frame==32:assert rows[0][2]==1 and base[0][2]==2
    assert model.statistics['identity_links']==1 and model.statistics['primary_association_costs_changed']==0


def test_ir_disagreement_veto_and_published_id_never_reassigned():
    model=TrackletPermanence(cases()[3],THETA)
    for frame in range(1,33):rows=model.update('video',frame,*inputs(frame,wrong_IR=True))
    assert rows[0][2]==2 and model.statistics['identity_links']==0
    rows=model.update('video',33,*inputs(33,wrong_IR=False))
    assert rows[0][2]==2 and model.statistics['published_IDs_reassigned']==0


def test_sequence_reset_clears_old_identity_templates():
    model=TrackletPermanence(cases()[3],THETA)
    for frame in range(1,33):model.update('first',frame,*inputs(frame))
    assert model.statistics['identity_links']==1
    rows=model.update('second',1,*inputs(1))
    assert rows[0][2]==1 and not model.closed and model.statistics['identity_links']==0


def test_just_removed_track_cannot_lend_id_while_it_can_still_match():
    model=TrackletPermanence(cases()[3],THETA)
    original=IndependentByteTrack()
    for frame in range(1,31):
        boxes=([[0,0,20,10,.95]] if frame<=3 else
               [[40,0,60,10,.95]] if frame in [28,29] else
               [[0,0,20,10,.95],[40,0,60,10,.95]] if frame==30 else [])
        packet=np.asarray(boxes,dtype=np.float32).reshape(-1,5)
        evidence=np.tile(np.array([[1,0]],np.float32),(len(packet),1))
        rows=model.update('video',frame,packet,evidence,evidence)
        baseline=original.update(packet,frame)
        assert len(rows)==len(baseline)
        for actual,expected in zip(rows,baseline):
            np.testing.assert_array_equal(actual[1],expected[1])
        if frame==29:
            assert 1 in {t.track_id for t in model.core.lost_stracks}
            assert 1 not in model.closed
            assert rows[0][2]==2
        if frame==30:
            assert {row[2]for row in rows}=={1,2}
    assert model.statistics['identity_links']==0
    assert model.statistics['deferred_removed_still_in_matching_pool']>0


def test_below_threshold_competitor_still_prevents_ambiguous_link():
    from types import SimpleNamespace
    model=TrackletPermanence(cases()[3],dict(THETA,RGB_cosine_threshold=.5))
    model.reset('video')
    box=np.array([0,0,20,10],np.float32)
    for identity,score,sign in [(1,.51,1),(2,.49,-1)]:
        model.closed[identity]=dict(rgb=np.array([score,sign*np.sqrt(1-score**2)],np.float32),
            ir=np.array([1,0],np.float32),box=box,last_frame=1,internal_track=identity)
    new=SimpleNamespace(track_id=3,accepted=2,rgb=np.array([1,0],np.float32),
        ir=np.array([1,0],np.float32),last_box=box)
    assert model.link([new],40,set())=={}
    assert model.statistics['identity_links']==0
