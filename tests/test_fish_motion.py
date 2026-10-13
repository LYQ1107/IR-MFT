import numpy as np
import pytest
from cfp.fish_motion import NativeUnscentedFilter
from yolox.tracker.kalman_filter import KalmanFilter


def test_same_native_linear_predict_project_update_including_observation_gaps():
    rng=np.random.default_rng(6);kf=KalmanFilter();ukf=NativeUnscentedFilter();measurement=np.array([120.,80.,1.2,20.]);a=kf.initiate(measurement);b=ukf.initiate(measurement)
    for frame in range(160):
        a=kf.predict(*a);b=ukf.predict(*b)
        np.testing.assert_allclose(a[0],b[0],atol=2e-9,rtol=2e-11);np.testing.assert_allclose(a[1],b[1],atol=2e-9,rtol=2e-11)
        projected_a=kf.project(*a);projected_b=ukf.project(*b)
        for x,y in zip(projected_a,projected_b):np.testing.assert_allclose(x,y,atol=2e-9,rtol=2e-11)
        if frame%9<5:
            observation=np.array([120+frame*.2,80+frame*.1,1.2,20.])+rng.normal(0,[.1,.1,.002,.1])
            a=kf.update(*a,observation);b=ukf.update(*b,observation)
            np.testing.assert_allclose(a[0],b[0],atol=2e-9,rtol=2e-11);np.testing.assert_allclose(a[1],b[1],atol=2e-9,rtol=2e-11)
        assert np.linalg.eigvalsh((b[1]+b[1].T)/2).min()>0
    assert ukf.calls['predict']==160


def test_sigma_transform_reproduces_native_gaussian_without_clipping():
    rng=np.random.default_rng(7);u=NativeUnscentedFilter();mean=rng.normal(size=8);factor=rng.normal(size=(8,8));cov=factor@factor.T+np.eye(8)
    m,p=u.transform(u.sigma_points(mean,cov));np.testing.assert_allclose(m,mean,atol=1e-14);np.testing.assert_allclose(p,cov,atol=1e-13)
    with pytest.raises(np.linalg.LinAlgError):u.sigma_points(mean,-np.eye(8))


def test_batched_and_empty_motion_inputs_have_same_filter_result():
    u=NativeUnscentedFilter();k=KalmanFilter();states=[u.initiate(np.array([100.,100.,.5,20.])),u.initiate(np.array([50.,70.,1.,30.]))];means=np.array([a[0] for a in states]);cov=np.array([a[1] for a in states]);a=u.multi_predict(means,cov);b=k.multi_predict(means,cov)
    for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=1e-12)
    m,c=u.multi_predict(np.empty((0,8)),np.empty((0,8,8)));assert m.shape==(0,8) and c.shape==(0,8,8)
