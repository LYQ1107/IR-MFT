"""Train-calibrated constant-acceleration covariance in the existing tracker.

Only covariance dynamics change. Original measurements, initialization,
association thresholds and lost lifetime are retained. A scoped shared-filter
binding covers ByteTrack's static batch prediction and always restores it.
"""
from contextlib import contextmanager
from threading import RLock
import numpy as np
from tools.mft_ambiguity_reasoning import AmbiguityReasoning, policies as previous_policies
from yolox.tracker.byte_tracker import STrack
from yolox.tracker.kalman_filter import KalmanFilter

_PREDICTION_LOCK=RLock()


def policies():
    return [dict(id='MK_native_zero',motion='original',late=False),
            dict(id='MK_late_only',motion='original',late=True),
            dict(id='MK_center_Q',motion='center',late=False),
            dict(id='MK_full_Q',motion='full',late=False),
            dict(id='MK_center_Q_late',motion='center',late=True),
            dict(id='MK_full_Q_late',motion='full',late=True)]


@contextmanager
def prediction_filter(filter_object):
    with _PREDICTION_LOCK:
        previous=STrack.shared_kalman
        STrack.shared_kalman=filter_object
        try:
            yield
        finally:
            STrack.shared_kalman=previous


class TrainMotionKalman(KalmanFilter):
    def __init__(self,sigmas,mode):
        super().__init__()
        self.sigmas=np.asarray(sigmas,np.float64).copy()
        assert self.sigmas.shape==(4,)and np.isfinite(self.sigmas).all()and(self.sigmas>=1e-5).all()
        assert mode in ['center','full']
        self.dimensions=[0,1]if mode=='center'else[0,1,2,3]

    def process_covariance(self,mean):
        height=max(float(mean[3]),1.)
        pos=[self._std_weight_position*height,self._std_weight_position*height,1e-2,self._std_weight_position*height]
        vel=[self._std_weight_velocity*height,self._std_weight_velocity*height,1e-5,self._std_weight_velocity*height]
        result=np.diag(np.square(np.r_[pos,vel]))
        for i in self.dimensions:
            scale=1. if i==2 else height
            variance=float(self.sigmas[i]*scale)**2
            result[i,i]=.25*variance
            result[i,i+4]=result[i+4,i]=.5*variance
            result[i+4,i+4]=variance
        return result

    def predict(self,mean,covariance):
        q=self.process_covariance(mean)
        predicted=mean@self._motion_mat.T
        covariance=self._motion_mat@covariance@self._motion_mat.T+q
        return predicted,(covariance+covariance.T)/2

    def multi_predict(self,means,covariances):
        if not len(means):
            return means.copy(),covariances.copy()
        values=[self.predict(m,p)for m,p in zip(means,covariances)]
        return np.stack([v[0]for v in values]),np.stack([v[1]for v in values])


class TrainMotionPrior(AmbiguityReasoning):
    def __init__(self,case,thresholds,sigmas):
        assert case in policies()
        self.motion_case=dict(case)
        self.motion_sigmas=np.asarray(sigmas,np.float64).copy()
        super().__init__(previous_policies()[1 if case['late']else 0],thresholds)

    def reset(self,sequence):
        super().reset(sequence)
        if self.motion_case['motion']!='original':
            self.core.kalman_filter=TrainMotionKalman(self.motion_sigmas,self.motion_case['motion'])
        self.statistics['Train_calibrated_motion_update_frames']=0

    def update(self,sequence,frame,detections,rgb=None,ir=None):
        if self.sequence!=sequence:
            assert frame==1
            self.reset(sequence)
        if self.motion_case['motion']=='original':
            return super().update(sequence,frame,detections,rgb,ir)
        with prediction_filter(self.core.kalman_filter):
            result=super().update(sequence,frame,detections,rgb,ir)
        self.statistics['Train_calibrated_motion_update_frames']+=1
        return result
