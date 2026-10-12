from types import SimpleNamespace

import numpy as np

from cfp.predictive_association import PredictiveAssociationTracker
from cfp.reliable_writes import ReliableWritesMixin
from yolox.tracker.basetrack import BaseTrack

ARGS=SimpleNamespace(track_thresh=.58,track_buffer=45,match_thresh=.97,mot20=False)
POLICY=dict(minimum_confidence=.68,minimum_row_margin=.02,minimum_column_margin=.02)


class ReliablePredictiveTracker(ReliableWritesMixin,PredictiveAssociationTracker):
    pass


def run(policy,score=.62,future=False):
    BaseTrack._count=0;tracker=ReliablePredictiveTracker(ARGS,memory='last',write_policy=policy)
    outputs=[]
    for frame in range(1,4):
        box=np.array([[frame,0,30+frame,20,.9 if frame==1 else score]],np.float32)
        feature=np.array([[1.,0.]],np.float32) if frame==1 else np.array([[.8,.6]],np.float32)
        if future and frame==3:feature=feature[:,::-1].copy()
        targets=tracker.update(box,(1,1),(1,1),feature)
        outputs.append([(t.track_id,*t.tlwh.tolist()) for t in targets])
    return tracker,outputs


def test_refusal_preserves_identity_but_geometry_keeps_updating_and_disabled_is_exact():
    baseline,baseline_trace=run(None)
    safe,safe_trace=run(POLICY)
    assert baseline_trace==safe_trace  # This fixture's correct geometry assignment is unchanged.
    assert baseline.tracked_stracks[0].last_identity_frame==3
    assert safe.tracked_stracks[0].last_identity_frame==1
    assert len(safe.tracked_stracks[0].identity_history)==1
    assert np.array_equal(safe.tracked_stracks[0].memory_rgb,np.array([1.,0.],np.float32))
    assert safe.tracked_stracks[0].frame_id==3 and not safe.tracked_stracks[0].pending
    assert safe.write_counts['newborn_initializations']==1 and safe.write_counts['rejected_confidence']==2
    BaseTrack._count=0;ordinary=PredictiveAssociationTracker(ARGS,memory='last')
    outputs=[]
    for frame in range(1,4):
        targets=ordinary.update(np.array([[frame,0,30+frame,20,.9 if frame==1 else .62]],np.float32),(1,1),(1,1),np.array([[1.,0.] if frame==1 else [.8,.6]],np.float32))
        outputs.append([(t.track_id,*t.tlwh.tolist()) for t in targets])
    assert outputs==baseline_trace and np.array_equal(ordinary.tracked_stracks[0].memory_rgb,baseline.tracked_stracks[0].memory_rgb)


def test_safe_policy_keeps_future_prefix_and_high_confidence_unambiguous_writes():
    first,trace=run(POLICY,.9);second,future=run(POLICY,.9,True)
    assert trace[:2]==future[:2]
    assert first.write_counts['accepted_existing']==2 and first.write_counts['rejected_existing']==0
    assert first.tracked_stracks[0].last_identity_frame==3


def test_row_and_column_competition_refuses_poison_and_low_stage_keeps_geometry():
    BaseTrack._count=0;tracker=ReliablePredictiveTracker(ARGS,memory='last',write_policy=POLICY)
    boxes=np.array([[0,0,30,20,.9],[70,0,100,20,.9]],np.float32);features=np.eye(2,dtype=np.float32)
    tracker.update(boxes.copy(),(1,1),(1,1),features)
    tracks=tracker.tracked_stracks;tracker.frame_id=2;tracker._write_evidence={}
    for track in tracks:tracker._guard_track(track)
    track=tracks[0];track.pending=True;track.current_proposal_index=0
    tracker._write_evidence[(track.track_id,0)]=dict(frame=2,row_margin=-.03,column_margin=.1)
    before=track.memory_rgb.copy();track.current_rgb=np.array([0.,1.],np.float32)
    assert not track.write_identity(2,.9) and np.array_equal(track.memory_rgb,before)
    track.pending=True;tracker._write_evidence[(track.track_id,0)]['row_margin']=.1;tracker._write_evidence[(track.track_id,0)]['column_margin']=0.
    assert not track.write_identity(2,.9) and tracker.write_counts['rejected_margin']==2
    low,trace=run(POLICY,.4)
    assert len(trace[-1])==1 and low.tracked_stracks[0].frame_id==3
    assert low.tracked_stracks[0].last_identity_frame==1 and low.write_counts['rejected_low_or_unscored']==2
