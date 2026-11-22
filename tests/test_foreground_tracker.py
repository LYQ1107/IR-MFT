"""Identity, lifecycle and causality regressions for point association."""
from types import SimpleNamespace

import numpy as np

from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.foreground_byte_tracker import ForegroundByteTracker


def with_transport(boxes, scores, offsets, zero=False):
    offsets=np.asarray(offsets,int);counts=np.diff(offsets)
    data=dict(boxes=np.asarray(boxes,np.float32),scores=np.asarray(scores,np.float32),offsets=offsets)
    point=dict(quality=np.ones((len(boxes),2),np.float16))
    for lag in [1,2,4,8]:
        previous=np.r_[np.zeros(lag,int),counts[:-lag]] if len(counts)>lag else np.zeros_like(counts)
        point[f'vote_offsets_{lag}']=np.r_[0,np.cumsum(previous*counts)]
        point[f'count_offsets_{lag}']=np.r_[0,np.cumsum(previous)]
        matrices=[]
        for frame,count in enumerate(counts):
            matrix=np.zeros((previous[frame],count),np.float16)
            if not zero:np.fill_diagonal(matrix,1.)
            matrices.append(matrix.ravel())
        for mode in ['bbox','rgb','ir','joint']:
            point[f'good_{lag}_{mode}']=np.full(previous.sum(),20,np.uint8)
            point[f'total_{lag}_{mode}']=np.full(previous.sum(),20,np.uint8)
            for target in ['bbox','rgb','ir','joint']:
                point[f'vote_{lag}_{mode}_{target}']=np.concatenate(matrices)
    data['foreground_transport']=point
    return data


def run_original(data, frames):
    BaseTrack._count=0
    tracker=BYTETracker(SimpleNamespace(track_thresh=.6,track_buffer=30,match_thresh=.9,mot20=False),frame_rate=25)
    output=[]
    for f in range(1,frames+1):
        a,b=data['offsets'][f-1:f+1]
        det=np.column_stack([data['boxes'][a:b],data['scores'][a:b]])
        output.append([(f,t.tlwh.copy(),t.track_id,t.score) for t in tracker.update(det,(1,1),(1,1))])
    return output


def run_points(data, frames, **config):
    BaseTrack._count=0
    tracker=ForegroundByteTracker(config)
    output=[tracker.update(f,data) for f in range(1,frames+1)]
    return tracker,output


def assert_same(first,second):
    assert len(first)==len(second)
    for a,b in zip(first,second):
        assert len(a)==len(b)
        for x,y in zip(a,b):
            assert x[0]==y[0] and x[2:]==y[2:]
            np.testing.assert_array_equal(x[1],y[1])


def test_disabled_or_unsupported_points_keep_original_lifecycle_exactly():
    boxes=[];scores=[];offsets=[0]
    for frame in range(1,66):
        for identity in range(3):
            if identity==1 and 15<=frame<=46:continue
            if identity==2 and 20<=frame<=23:continue
            x=20+identity*60+.3*frame
            boxes.append([x,20,x+25,40]);scores.append(.4 if frame%7==0 else .95)
        offsets.append(len(boxes))
    data=with_transport(boxes,scores,offsets)
    expected=run_original(data,65)
    tracker,actual=run_points(data,65,point_weight=0,object_history=True,lags=[1,2,4,8])
    assert tracker.inner.max_time_lost==25
    assert_same(expected,actual)
    data=with_transport(boxes,scores,offsets,zero=True)
    tracker,actual=run_points(data,65,point_weight=1,object_history=True,lags=[1,2,4,8])
    assert_same(expected,actual)
    assert tracker.statistics['point_rows']==0


def crossing_data():
    centers=[(25,75),(75,25),(90,10)]
    boxes=[[x-15,20,x+15,40] for pair in centers for x in pair]
    return with_transport(boxes,[.95]*6,[0,2,4,6])


def test_corresponding_body_points_recover_crossing_with_zero_box_overlap():
    data=crossing_data()
    plain=run_original(data,3)
    tracker,point=run_points(data,3,point_weight=1,reciprocal_gate=True)
    reference={x[2]:x[1][0]+x[1][2]/2 for x in plain[1]}
    corrected={x[2]:x[1][0]+x[1][2]/2 for x in point[1]}
    assert reference[1]<reference[2]
    assert corrected[1]>corrected[2]
    assert tracker.statistics['point_opened_edges']>=2
    assert all(len({x[2] for x in out})==2 for out in point)


def test_future_point_evidence_does_not_change_emitted_prefix():
    first=crossing_data()
    second={k:(v.copy() if isinstance(v,np.ndarray) else {n:z.copy() for n,z in v.items()}) for k,v in first.items()}
    second['boxes'][4:]=-1000
    p=second['foreground_transport']
    for lag in [1,2,4,8]:
        for mode in ['bbox','rgb','ir','joint']:
            p[f'vote_{lag}_{mode}_{mode}'][p[f'vote_offsets_{lag}'][2]:]=1e4
    a,oa=run_points(first,2,point_weight=1,object_history=True,lags=[1,2,4,8])
    b,ob=run_points(second,2,point_weight=1,object_history=True,lags=[1,2,4,8])
    assert_same(oa,ob)
    assert a.statistics==b.statistics


def test_previously_co_visible_objects_survive_overlap_duplicate_cleanup():
    data=with_transport([[20,20,40,40],[60,20,80,40]]*2+[[40,20,60,40]],
                        [.95]*5,[0,2,4,5])
    for preserve in [False,True]:
        BaseTrack._count=0
        owner=ForegroundByteTracker(dict(point_weight=1,preserve_distinct_objects=preserve))
        owner.update(1,data);owner.update(2,data)
        if preserve:assert owner.co_visible[(1,2)]==2
        # Two previously separate fish now project onto the same image region.
        # The current observation belongs only to fish1; fish2 is occluded.
        for track in owner.inner.tracked_stracks:
            track.mean[0]=50. if track.track_id==1 else 50.1
            track.mean[4:]=0
        output=owner.update(3,data)
        assert [row[2] for row in output]==[1]
        assert bool(owner.inner.lost_stracks)==preserve
        if preserve:
            assert owner.inner.lost_stracks[0].track_id==2
            assert owner.statistics['duplicate_vetoes']==1
