"""Complete counterfactual association context, including geometry fallback edges."""
import numpy as np

from .utility import decision_evidence


def complete_candidate_context(track,detections,raw_geometry,geometry,frame_id,
                               teacher_weight=.3,match_threshold=.97,
                               proximity_iou=.1,appearance_max_distance=.35):
    if track.memory_rgb is None or not detections:return None
    if track.last_identity_frame>=frame_id:raise ValueError('Current observation entered the teacher history')
    rgb=np.array([float(np.clip((1-track.memory_rgb@d.current_rgb)/2,0,1)) if d.current_rgb is not None else 1. for d in detections],np.float64)
    valid=np.array([d.current_rgb is not None for d in detections],bool)
    modifiable=valid & (raw_geometry<=1-proximity_iou) & (rgb<=appearance_max_distance)
    ir=rgb.copy();available=np.zeros(len(rgb),bool)
    for j,detection in enumerate(detections):
        if modifiable[j] and track.memory_ir is not None and detection.current_ir is not None:
            ir[j]=float(np.clip((1-track.memory_ir@detection.current_ir)/2,0,1));available[j]=True
    if not available.any():return None
    null_cost=match_threshold if track.is_activated else .7
    # Include every geometry-accepted edge, even with weak/unavailable appearance,
    # and every modifiable edge that either intervention could change.
    pool=np.flatnonzero((geometry<=null_cost)|modifiable)
    selected=np.flatnonzero(modifiable)
    teacher_rgb=geometry.copy()
    teacher_rgb[modifiable]=(1-teacher_weight)*geometry[modifiable]+teacher_weight*rgb[modifiable]
    teacher_ir=teacher_rgb.copy()
    teacher_ir[available]=(1-teacher_weight)*geometry[available]+teacher_weight*ir[available]
    evidence=decision_evidence(teacher_rgb[pool],teacher_ir[pool],raw_geometry[pool],
              np.array([detections[j].score for j in pool]),available[pool],
              frame_id-track.start_frame+1,frame_id-track.last_identity_frame,
              [row[1] for row in track.identity_history],null_distance=null_cost)
    return dict(pool=pool,modifiable=selected,rgb=teacher_rgb[pool],ir=teacher_ir[pool],
                raw_rgb=rgb[selected],raw_ir=ir[selected],evidence=evidence,null_distance=null_cost)
