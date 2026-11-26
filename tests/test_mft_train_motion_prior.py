import numpy as np
import pytest
from tools.mft_train_motion_prior import TrainMotionKalman,TrainMotionPrior,policies,prediction_filter
from tools.mft_ambiguity_reasoning import AmbiguityReasoning,policies as prior
from tools.native_pyramid_stream_tracks import mot_line
from yolox.tracker.byte_tracker import STrack
from yolox.tracker.kalman_filter import KalmanFilter

SIGMAS=[.0524738868694014,.03998233376115184,.029661453082623204,.029335073269698916]
THRESHOLDS=dict(RGB_cosine_threshold=.5,IR_cosine_threshold=.30187624990940093,
    Train_unknown_speed_upper=.13137420405549355,primary_RGB_cosine_threshold=.3608310019969939,
    primary_IR_cosine_threshold=.42077010989189145)


def test_original_measurement_and_initialization_retained_covariance_psd():
    original=KalmanFilter();new=TrainMotionKalman(SIGMAS,'full')
    measurement=np.asarray([100.,80.,1.5,30.])
    a,p=original.initiate(measurement);b,q=new.initiate(measurement)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(p,q)
    for mode in ['center','full']:
        model=TrainMotionKalman(SIGMAS,mode)
        for frame in range(1,41):
            a,p=model.predict(a,p)
            assert np.linalg.eigvalsh(p).min()>-1e-9
            np.testing.assert_array_equal(model.project(a,p)[0],original.project(a,p)[0])
            np.testing.assert_array_equal(model.project(a,p)[1],original.project(a,p)[1])
            if frame%5!=0:
                a,p=model.update(a,p,measurement+np.asarray([frame*.8,frame*.3,(-1)**frame*.1,0.]))
                assert np.isfinite(a).all()and np.linalg.eigvalsh((p+p.T)/2).min()>-1e-9


def test_center_only_leaves_shape_process_block_and_scalar_batch_agree():
    a,p=KalmanFilter().initiate(np.asarray([10.,20.,2.,40.]))
    model=TrainMotionKalman(SIGMAS,'center')
    _,oldp=KalmanFilter().predict(a,p);_,newp=model.predict(a,p)
    shape=[2,3,6,7]
    np.testing.assert_array_equal(oldp[np.ix_(shape,shape)],newp[np.ix_(shape,shape)])
    means,covs=model.multi_predict(np.stack([a,a]),np.stack([p,p]))
    x,y=model.predict(a,p)
    np.testing.assert_array_equal(means,np.stack([x,x]));np.testing.assert_array_equal(covs,np.stack([y,y]))


def test_shared_filter_restored_after_exception_and_interleaved_zero_paths():
    old=STrack.shared_kalman;model=TrainMotionKalman(SIGMAS,'full')
    with pytest.raises(RuntimeError):
        with prediction_filter(model):
            assert STrack.shared_kalman is model
            raise RuntimeError('synthetic deliberate failure')
    assert STrack.shared_kalman is old
    trackers=[TrainMotionPrior(policies()[i],THRESHOLDS,SIGMAS)for i in range(6)]
    references=[AmbiguityReasoning(prior()[i],THRESHOLDS)for i in [0,1]]
    cue=np.eye(2,dtype=np.float32)
    for frame in range(1,36):
        d=np.asarray([[frame,0,frame+20,20,.9],[60,0,80,20,.9]],np.float32)
        if 5<=frame<=8:d=d[:1]
        # Calibrated updates precede both original cases to detect shared-state leakage.
        for tracker in trackers[2:]:tracker.update('synthetic',frame,d,cue[:len(d)],cue[:len(d)])
        actual=[t.update('synthetic',frame,d,cue[:len(d)],cue[:len(d)])for t in trackers[:2]]
        expected=[t.update('synthetic',frame,d,cue[:len(d)],cue[:len(d)])for t in references]
        assert [[mot_line(r)for r in v]for v in actual]==[[mot_line(r)for r in v]for v in expected]
        assert STrack.shared_kalman is old


def test_sequence_reset_and_no_future_measurement_access():
    model=TrainMotionPrior(policies()[5],THRESHOLDS,SIGMAS)
    other=TrainMotionPrior(policies()[5],THRESHOLDS,SIGMAS)
    cue=np.asarray([[1.,0.]],np.float32);d=np.asarray([[0,0,20,20,.9]],np.float32)
    for frame in range(1,12):model.update('first',frame,d,cue,cue)
    a=model.update('second',1,d,cue,cue);b=other.update('second',1,d,cue,cue)
    assert ''.join(map(mot_line,a))==''.join(map(mot_line,b))
    assert model.statistics['Train_calibrated_motion_update_frames']==1
