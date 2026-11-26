"""Separate identity continuity from current observed box shape.

Aspect process noise uses the unchanged Train estimate. Rendering anchors
never feed back into association or filter state and never revise past output.
"""
from contextlib import nullcontext
import numpy as np
from tools.mft_ambiguity_reasoning import AmbiguityReasoning,QualityTrack,policies as old_policies
from tools.mft_train_motion_prior import TrainMotionKalman,prediction_filter


def policies():
    return [dict(id='PC_native_zero',aspect_Q=False,render='filtered',late=False),
            dict(id='PC_late_only',aspect_Q=False,render='filtered',late=True),
            dict(id='PC_aspect_Q',aspect_Q=True,render='filtered',late=False),
            dict(id='PC_aspect_Q_late',aspect_Q=True,render='filtered',late=True),
            dict(id='PC_shape_anchor',aspect_Q=False,render='current_shape',late=False),
            dict(id='PC_shape_anchor_late',aspect_Q=False,render='current_shape',late=True),
            dict(id='PC_raw_box',aspect_Q=False,render='current_box',late=False),
            dict(id='PC_raw_box_late',aspect_Q=False,render='current_box',late=True)]


class AspectKalman(TrainMotionKalman):
    def __init__(self,sigmas):
        super().__init__(sigmas,'full')
        self.dimensions=[2]


class MeasurementTrack(QualityTrack):
    def __init__(self,*args):
        super().__init__(*args)
        self.current_measurement=self._tlwh.copy()
        self.measurement_frame=None

    def activate(self,kalman_filter,frame_id):
        super().activate(kalman_filter,frame_id)
        self.measurement_frame=frame_id

    def update(self,new_track,frame_id):
        super().update(new_track,frame_id)
        self.current_measurement=new_track._tlwh.copy()
        self.measurement_frame=frame_id

    def re_activate(self,new_track,frame_id,new_id=False):
        super().re_activate(new_track,frame_id,new_id)
        self.current_measurement=new_track._tlwh.copy()
        self.measurement_frame=frame_id


class PoseContinuity(AmbiguityReasoning):
    def __init__(self,case,thresholds,sigmas):
        assert case in policies()
        self.pose_case=dict(case)
        self.sigmas=np.asarray(sigmas,np.float64).copy()
        super().__init__(old_policies()[1 if case['late']else 0],thresholds)

    def reset(self,sequence):
        super().reset(sequence)
        if self.pose_case['aspect_Q']:
            self.core.kalman_filter=AspectKalman(self.sigmas)
        if self.pose_case['render']!='filtered':
            counter=[0]
            self.core.track_factory=lambda box,score,obs:MeasurementTrack(box,score,obs,counter)
        self.statistics.update(aspect_calibrated_update_frames=0,anchor_applied_rows=0,
            numeric_output_box_changed_rows=0,small_raw_area_original_box_fallbacks=0)

    def update(self,sequence,frame,detections,rgb=None,ir=None):
        if self.sequence!=sequence:
            assert frame==1
            self.reset(sequence)
        scope=prediction_filter(self.core.kalman_filter)if self.pose_case['aspect_Q']else nullcontext()
        with scope:
            rows=super().update(sequence,frame,detections,rgb,ir)
        self.statistics['aspect_calibrated_update_frames']+=int(self.pose_case['aspect_Q'])
        if self.pose_case['render']=='filtered':
            return rows
        live={self.published[t.track_id]:t for t in self.core.tracked_stracks
              if t.is_activated and t.frame_id==frame and t.track_id in self.published}
        result=[]
        for time,filtered,identity,score in rows:
            track=live[identity]
            assert track.measurement_frame==frame
            measured=track.current_measurement
            assert np.isfinite(measured).all()and(measured[2:]>0).all()
            if measured[2]*measured[3]<=100:
                self.statistics['small_raw_area_original_box_fallbacks']+=1
                box=filtered
            else:
                self.statistics['anchor_applied_rows']+=1
                box=measured.copy()
                if self.pose_case['render']=='current_shape':
                    box[:2]=filtered[:2]+filtered[2:]/2-box[2:]/2
                self.statistics['numeric_output_box_changed_rows']+=int(not np.array_equal(box,filtered))
            result.append((time,box,identity,score))
        return result
