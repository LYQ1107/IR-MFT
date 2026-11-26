"""GT-free, current-candidate/past-template features for association risk."""
import numpy as np
from yolox.tracker import matching

FEATURE_NAMES=(
    'geometric_cost','IoU','dx_body','dy_body','center_error_body',
    'abs_log_width','abs_log_height','abs_log_aspect','log_gap','log_age',
    'detection_score','current_quality','past_template_quality','log_template_age',
    'RGB_cosine','IR_cosine','IR_available','RGB_row_margin','RGB_column_margin',
    'IR_row_margin','IR_column_margin','cue_disagreement','RGB_quality_support',
    'IR_quality_support','velocity_alignment','relative_speed')
IR_FEATURES=tuple(FEATURE_NAMES.index(n)for n in[
    'IR_cosine','IR_available','IR_row_margin','IR_column_margin',
    'cue_disagreement','IR_quality_support'])


def rgb_only_mask(values):
    result=np.asarray(values).copy()
    assert result.shape[-1]==len(FEATURE_NAMES)
    result[...,IR_FEATURES]=0
    return result


def _margins(scores,valid):
    """All eligible competitors count, including weak similarity candidates."""
    a=np.where(valid,scores,-np.inf)
    if not a.size:return np.zeros_like(a),np.zeros_like(a)
    def along_rows(values,mask):
        winner=np.argmax(values,axis=1);indices=np.arange(len(values))
        other=values.copy();other[indices,winner]=-np.inf
        competitors=np.broadcast_to(values.max(axis=1,keepdims=True),values.shape).copy()
        competitors[indices,winner]=other.max(axis=1)
        result=np.zeros_like(values)
        np.subtract(values,competitors,out=result,where=mask&np.isfinite(competitors))
        return result
    return along_rows(a,valid),along_rows(a.T,valid.T).T


def candidate_features(tracks,detections,base,frame,speed_upper):
    """No GT object or identity key is accepted by this function."""
    base=np.asarray(base,np.float64)
    assert base.shape==(len(tracks),len(detections))and type(frame)is int and frame>0
    assert np.isfinite(base).all()and 0<=speed_upper<=1
    n,m=base.shape
    x=np.zeros((n,m,len(FEATURE_NAMES)),np.float32)
    eligible=np.zeros((n,m),bool);rgb=np.zeros((n,m),np.float64);ir=np.zeros_like(rgb);ir_valid=np.zeros_like(eligible)
    if not base.size:return x,eligible
    ious=np.clip(1-matching.iou_distance(tracks,detections),0,1)
    for i,t in enumerate(tracks):
        if t.accepted<3 or t.rgb is None or t.last_box is None:continue
        assert t.frame_id<frame and t.memory_frame is not None and t.memory_frame<frame
        gap=frame-t.frame_id
        past=np.asarray(t.last_box,np.float64);size=np.maximum(past[2:]-past[:2],1.)
        past_center=(past[:2]+past[2:])/2;pred=np.asarray(t.tlbr,np.float64)
        predicted_center=(pred[:2]+pred[2:])/2;diagonal=float(np.linalg.norm(size))
        velocity=np.asarray(t.mean[4:6],np.float64)
        for j,d in enumerate(detections):
            obs=d.observation
            if obs is None:continue
            box=np.asarray(obs['box'],np.float64);current_size=np.maximum(box[2:]-box[:2],1.)
            center=(box[:2]+box[2:])/2;motion=center-past_center
            reach=np.linalg.norm(motion)/diagonal<=.5+speed_upper*gap
            scale=current_size/size
            supported=base[i,j]<=.9 or(reach and 1<=gap<=25 and(scale>=.5).all()and(scale<=2).all())
            if not supported:continue
            eligible[i,j]=True
            rgb[i,j]=float(np.clip(t.rgb@obs['rgb'],-1,1))
            available=bool(obs.get('ir_available',True))
            if available and t.ir is not None:
                ir[i,j]=float(np.clip(t.ir@obs['ir'],-1,1));ir_valid[i,j]=True
            error=(center-predicted_center)/diagonal
            vnorm=np.linalg.norm(velocity);mnorm=np.linalg.norm(motion)
            alignment=float(velocity@motion/(vnorm*mnorm))if vnorm>1e-9 and mnorm>1e-9 else 0.
            quality=float(np.clip(obs.get('quality',d.score),0,1));past_quality=float(np.clip(t.memory_quality,0,1))
            support=np.sqrt(quality*past_quality)
            values=[base[i,j],ious[i,j],*np.clip(error,-8,8),min(float(np.linalg.norm(error)),8),
                min(abs(float(np.log(scale[0]))),4),min(abs(float(np.log(scale[1]))),4),
                min(abs(float(np.log(scale[0]/scale[1]))),4),np.log1p(gap),np.log1p(frame-t.start_frame),
                d.score,quality,past_quality,np.log1p(frame-t.memory_frame),rgb[i,j],ir[i,j],float(ir_valid[i,j]),
                0.,0.,0.,0.,abs(rgb[i,j]-ir[i,j])if ir_valid[i,j]else 0.,
                rgb[i,j]*support,ir[i,j]*support if ir_valid[i,j]else 0.,np.clip(alignment,-1,1),
                min(float(mnorm/(gap*diagonal)),8)]
            assert len(values)==len(FEATURE_NAMES)
            x[i,j]=values
    rr,rc=_margins(rgb,eligible);ar,ac=_margins(ir,ir_valid)
    for name,values in [('RGB_row_margin',rr),('RGB_column_margin',rc),('IR_row_margin',ar),('IR_column_margin',ac)]:
        x[...,FEATURE_NAMES.index(name)]=values
    assert np.isfinite(x).all()
    return x,eligible
