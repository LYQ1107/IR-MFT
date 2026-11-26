import numpy as np
from tools.mft_pose_continuity import AspectKalman,PoseContinuity,policies
from tools.mft_ambiguity_reasoning import AmbiguityReasoning,policies as old
from tools.native_pyramid_stream_tracks import mot_line
from yolox.tracker.byte_tracker import STrack
from yolox.tracker.kalman_filter import KalmanFilter

SIGMAS=[.0524738868694014,.03998233376115184,.029661453082623204,.029335073269698916]
THRESHOLDS=dict(RGB_cosine_threshold=.5,IR_cosine_threshold=.30187624990940093,
    Train_unknown_speed_upper=.13137420405549355,primary_RGB_cosine_threshold=.3608310019969939,
    primary_IR_cosine_threshold=.42077010989189145)


def meta(rows):
    return [(f,i,s)for f,box,i,s in rows]


def test_aspect_prior_leaves_center_and_height_given_same_measurements():
    old_filter=KalmanFilter();new_filter=AspectKalman(SIGMAS)
    measurement=np.asarray([100.,60.,2.,20.])
    a,p=old_filter.initiate(measurement);b,q=new_filter.initiate(measurement)
    index=[0,1,3,4,5,7]
    changed=False
    for frame in range(1,41):
        a,p=old_filter.predict(a,p);b,q=new_filter.predict(b,q)
        observation=measurement+np.asarray([frame*.7,frame*.1,.3*np.sin(frame*.4),.5*np.cos(frame*.2)])
        a,p=old_filter.update(a,p,observation);b,q=new_filter.update(b,q,observation)
        np.testing.assert_allclose(a[index],b[index],rtol=0,atol=1e-10)
        np.testing.assert_allclose(p[np.ix_(index,index)],q[np.ix_(index,index)],rtol=0,atol=1e-10)
        assert np.linalg.eigvalsh((q+q.T)/2).min()>-1e-9
        changed|=abs(a[2]-b[2])>1e-5
    assert changed


def test_zero_late_literal_and_rendering_preserves_identity_score_filter_states():
    trackers=[PoseContinuity(c,THRESHOLDS,SIGMAS)for c in policies()]
    references=[AmbiguityReasoning(old()[i],THRESHOLDS)for i in [0,1]]
    cue=np.eye(2,dtype=np.float32);shared=STrack.shared_kalman
    for frame in range(1,36):
        d=np.asarray([[frame*.7,0,frame*.7+20+4*np.sin(frame),20,.9],[65,0,85,20,.9]],np.float32)
        if 5<=frame<=8:d=d[:1]
        original=[m.update('synthetic',frame,d,cue[:len(d)],cue[:len(d)])for m in references]
        actual=[m.update('synthetic',frame,d,cue[:len(d)],cue[:len(d)])for m in trackers]
        assert [[mot_line(r)for r in v]for v in actual[:2]]==[[mot_line(r)for r in v]for v in original]
        for index,reference in [(4,0),(5,1),(6,0),(7,1)]:
            assert meta(actual[index])==meta(original[reference])
            own={t.track_id:t for t in trackers[index].core.tracked_stracks}
            expected={t.track_id:t for t in references[reference].core.tracked_stracks}
            assert set(own)==set(expected)
            for identity in own:
                np.testing.assert_array_equal(own[identity].mean,expected[identity].mean)
                np.testing.assert_array_equal(own[identity].covariance,expected[identity].covariance)
            for _,box,identity,score in actual[index]:
                t=next(t for t in own.values()if trackers[index].published.get(t.track_id)==identity)
                assert t.measurement_frame==frame
                if index in[6,7]:np.testing.assert_array_equal(box,t.current_measurement)
        assert STrack.shared_kalman is shared


def test_shape_anchor_uses_filtered_center_and_current_shape():
    original=PoseContinuity(policies()[0],THRESHOLDS,SIGMAS)
    anchor=PoseContinuity(policies()[4],THRESHOLDS,SIGMAS)
    cue=np.asarray([[1.,0.]],np.float32)
    for frame in range(1,12):
        width=20+frame*.6;d=np.asarray([[frame,0,frame+width,20,.9]],np.float32)
        a=original.update('synthetic',frame,d,cue,cue);b=anchor.update('synthetic',frame,d,cue,cue)
        assert meta(a)==meta(b)
        for x,y in zip(a,b):
            np.testing.assert_allclose(x[1][:2]+x[1][2:]/2,y[1][:2]+y[1][2:]/2,atol=1e-12)
            np.testing.assert_array_equal(y[1][2:],d[0,2:4]-d[0,:2])


def test_sequence_reset_and_small_raw_area_keeps_original_rows():
    original=PoseContinuity(policies()[0],THRESHOLDS,SIGMAS)
    anchor=PoseContinuity(policies()[6],THRESHOLDS,SIGMAS)
    cue=np.asarray([[1.,0.]],np.float32)
    for frame in range(1,8):
        size=20. if frame<7 else 9.9
        d=np.asarray([[0,0,size,size,.9]],np.float32)
        a=original.update('first',frame,d,cue,cue);b=anchor.update('first',frame,d,cue,cue)
        assert meta(a)==meta(b)
        if frame==7:
            for x,y in zip(a,b):np.testing.assert_array_equal(x[1],y[1])
    fresh=PoseContinuity(policies()[6],THRESHOLDS,SIGMAS)
    d=np.asarray([[0,0,20,20,.9]],np.float32)
    assert ''.join(map(mot_line,anchor.update('second',1,d,cue,cue)))==''.join(map(mot_line,fresh.update('second',1,d,cue,cue)))
