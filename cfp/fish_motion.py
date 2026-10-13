"""Independent sigma-point control with native motion/noise/lifecycle held fixed.

This 8-D XYAH constant-velocity UKF uses exactly the native linear F/H and
state-dependent Q/R. It does not invent nonlinear fish-motion evidence. The
unscented transform must agree numerically with the same linear Kalman model.
No external SU-T filter implementation or changed 9-D observation is copied.
"""
import numpy as np
import scipy.linalg
from yolox.tracker.kalman_filter import KalmanFilter


class NativeUnscentedFilter(KalmanFilter):
    def __init__(self):
        super().__init__()
        self.mean_weights=np.full(17,1/16,dtype=np.float64)
        self.mean_weights[0]=0.
        self.cov_weights=self.mean_weights.copy();self.cov_weights[0]=2.
        self.calls=dict(predict=0,project=0,update=0)

    def sigma_points(self,mean,covariance):
        mean=np.asarray(mean,dtype=np.float64);covariance=np.asarray(covariance,dtype=np.float64)
        if mean.shape!=(8,) or covariance.shape!=(8,8) or not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            raise ValueError('Finite native8D Gaussian state required')
        # alpha=1,beta=2,kappa=0; no noise tuning, eigenvalue clipping or jitter.
        factor=np.linalg.cholesky((covariance+covariance.T)*4)
        return np.concatenate([mean[None],mean[None]+factor.T,mean[None]-factor.T],axis=0)

    def transform(self,points):
        # Relative accumulation avoids unnecessary cancellation of image-space
        # center coordinates; mean and covariance remain the actual UT result.
        center=points[0];mean=center+self.mean_weights@(points-center)
        difference=points-mean
        covariance=(difference.T*self.cov_weights)@difference
        return mean,covariance

    def predict(self,mean,covariance):
        self.calls['predict']+=1
        pos=[self._std_weight_position*mean[3],self._std_weight_position*mean[3],1e-2,self._std_weight_position*mean[3]]
        vel=[self._std_weight_velocity*mean[3],self._std_weight_velocity*mean[3],1e-5,self._std_weight_velocity*mean[3]]
        points=self.sigma_points(mean,covariance)@self._motion_mat.T
        predicted,cov=self.transform(points)
        return predicted,cov+np.diag(np.square(pos+vel))

    def project(self,mean,covariance):
        self.calls['project']+=1
        points=self.sigma_points(mean,covariance)@self._update_mat.T
        observed,cov=self.transform(points)
        noise=[self._std_weight_position*mean[3],self._std_weight_position*mean[3],1e-1,self._std_weight_position*mean[3]]
        return observed,cov+np.diag(np.square(noise))

    def update(self,mean,covariance,measurement):
        self.calls['update']+=1
        points=self.sigma_points(mean,covariance);observations=points@self._update_mat.T
        projected,innovation_covariance=self.project(mean,covariance)
        differences=points-mean;observed_difference=observations-projected
        cross=(differences.T*self.cov_weights)@observed_difference
        factor,lower=scipy.linalg.cho_factor(innovation_covariance,lower=True,check_finite=False)
        gain=scipy.linalg.cho_solve((factor,lower),cross.T,check_finite=False).T
        return mean+gain@(measurement-projected),covariance-gain@innovation_covariance@gain.T

    def multi_predict(self,mean,covariance):
        if not len(mean):return np.array(mean,copy=True),np.array(covariance,copy=True)
        values=[self.predict(m,c) for m,c in zip(mean,covariance)]
        return np.array([v[0] for v in values]),np.array([v[1] for v in values])


def install_native_motion_filter(tracker,kind):
    """Fresh single-tracker process only; no changed track update/birth policy."""
    from yolox.tracker.byte_tracker import STrack
    if kind not in ('kalman','unscented_linear') or tracker.frame_id!=0 or tracker.tracked_stracks or tracker.lost_stracks:
        raise ValueError('Declared filter before any native track/birth required')
    if kind=='unscented_linear':
        sigma=NativeUnscentedFilter()
        tracker.kalman_filter=sigma
        # Native static batch prediction reads this process-local shared filter;
        # every born track separately receives tracker.kalman_filter on activate.
        STrack.shared_kalman=sigma
    return tracker
