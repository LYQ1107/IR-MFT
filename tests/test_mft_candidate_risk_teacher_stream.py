import numpy as np
import pytest
from tools.mft_candidate_risk_teacher import CandidateTeacher
from tools.mft_candidate_risk_training_stream import TrainEdgeCollector
from tools.native_pyramid_stream_tracks import IndependentByteTrack,mot_line


def packet(frame,empty=False):
    if empty:return np.empty((0,5),np.float32),np.empty((0,2),np.float16),np.empty((0,2),np.float16)
    shift=frame*.3
    d=np.asarray([[shift,0,shift+20,20,.95],[60-shift,0,80-shift,20,.85]],np.float32)
    return d,np.asarray([[1,0],[0,1]],np.float16),np.asarray([[.6,.8],[.8,-.6]],np.float16)


def gt(frame,swap=False):
    d=packet(frame)[0];result=[]
    for i,box in enumerate(d):
        result.append([frame,2-i if swap else i+1,box[0],box[1],box[2]-box[0],box[3]-box[1],1,1,1])
    return np.asarray(result,np.float64)


def test_teacher_observation_cannot_change_original_association_or_packets():
    callbacks=[]
    def observer(x,allowed,identities,indices,frame):
        callbacks.append((frame,allowed.copy(),identities,indices))
        x[:]=123;allowed[:]=False
        return np.full(allowed.shape,np.inf)  # Observer return must never enter LAP.
    teacher=CandidateTeacher(.13137420405549355,observer);original=IndependentByteTrack()
    for frame in range(1,48):
        d,r,a=packet(frame,empty=frame in [9,10,11,28,29,30,31,32,33,34])
        if frame in [6,7]and len(d):d[0,4]=.45  # Original low-confidence stage.
        before=[v.copy()for v in [d,r,a]]
        actual=teacher.update('synthetic',frame,d,r,a)
        expected=original.update(d,frame)
        assert ''.join(map(mot_line,actual))==''.join(map(mot_line,expected))
        for x,y in zip([d,r,a],before):np.testing.assert_array_equal(x,y)
        assignments=teacher.current_proposal_assignments(frame)
        assert len(set(assignments.values()))==len(assignments)
    assert any(allowed.any()for _,allowed,_,_ in callbacks)
    assert teacher.statistics['original_association_cost_entries_changed']==0


def test_training_stream_does_not_use_current_labels_as_past_history():
    batches=[];stream=TrainEdgeCollector(.13137420405549355,batches.append)
    for frame in range(1,7):
        stream.step('synthetic',frame,*packet(frame),gt(frame),split='train')
        if frame<=3:assert not batches
    assert batches and batches[0].frame==4
    assert all(b.x.shape[1]==26 and b.y.sum()==2 for b in batches)
    assert stream.statistics['positive_pairs']==6


def test_current_gt_changes_only_targets_never_features_or_trajectory():
    original=[];changed=[]
    a=TrainEdgeCollector(.13137420405549355,original.append)
    b=TrainEdgeCollector(.13137420405549355,changed.append)
    trajectories=[[],[]]
    for frame in range(1,7):
        for index,stream in enumerate([a,b]):
            rows=stream.step('synthetic',frame,*packet(frame),gt(frame,swap=index==1 and frame>=4),split='train')
            trajectories[index].extend(map(mot_line,rows))
    assert trajectories[0]==trajectories[1]
    assert len(original)==3 and len(changed)==1
    # The changed label lowers past purity after frame4; subsequent pairs are
    # unknown supervision, while every proposal still enters the same tracker.
    assert b.statistics['unknown_pairs']>a.statistics['unknown_pairs']
    np.testing.assert_array_equal(original[0].x,changed[0].x)
    assert not np.array_equal(original[0].y,changed[0].y)


def test_sequence_reset_and_prefix_replay_do_not_keep_identity_or_future_evidence():
    def run(length,suffix_change=False):
        batches=[];t=CandidateTeacher(.13137420405549355,
            lambda x,allowed,ids,indices,frame:batches.append((frame,x.copy(),allowed.copy())))
        outputs=[]
        for f in range(1,length+1):
            d,r,a=packet(f)
            if suffix_change and f>6:d[:,:4]+=777;r=-r;a=-a
            outputs.append(''.join(map(mot_line,t.update('one',f,d,r,a))))
        return t,outputs,batches
    teacher,a,features=run(10);_,b,other=run(10,True)
    assert a[:6]==b[:6]
    for x,y in zip(features[:6],other[:6]):
        assert x[0]==y[0];np.testing.assert_array_equal(x[1],y[1]);np.testing.assert_array_equal(x[2],y[2])
    assert ''.join(map(mot_line,teacher.update('two',1,*packet(1))))==a[0]
    batches=[];stream=TrainEdgeCollector(.13137420405549355,batches.append)
    for f in range(1,6):stream.step('one',f,*packet(f),gt(f),split='train')
    count=len(batches);stream.step('two',1,*packet(1),gt(1),split='train')
    assert len(batches)==count and stream.history.frame==1
    with pytest.raises(AssertionError):stream.step('two',2,*packet(2),gt(2),split='test')


def test_no_ir_teacher_keeps_actual_rgb_templates_and_feature_fallback():
    observed=[];teacher=CandidateTeacher(.13137420405549355,
        lambda x,allowed,*args:observed.append((x.copy(),allowed.copy())))
    for f in range(1,6):
        d,r,_=packet(f);teacher.update('synthetic',f,d,r,None)
    assert teacher.core.tracked_stracks[0].ir is None
    assert teacher.core.tracked_stracks[0].accepted==5
    assert observed[-1][1].any()
    assert np.isfinite(observed[-1][0]).all()
