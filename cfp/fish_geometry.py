"""Isolated source-informed geometry controls; no fish anatomy/occlusion labels.

FishIoU formula derives from SU-T (MIT, LI Weiran 2025), AGCv1 from TIDE
(MIT, Vran Lee 2026). Notices: third_party/notices/SU-T_LICENSE.txt and
TIDE_LICENSE.txt. We retain the FishIoU central-area bug in fish_source,
separate its disclosed repair, and disclose AGC finite clipping. These are
standalone geometry controls, not official complete-tracker reproduction.
"""
import numpy as np

KINDS=('native','continuous_iou','diou','fish_source','fish_central_repaired','agc_clipped','fish_affine_control','iou_offset005')


def box_terms(a,b):
    a=np.asarray(a,dtype=np.float64).reshape(-1,4);b=np.asarray(b,dtype=np.float64).reshape(-1,4)
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Finite native predicted/current boxes required')
    x=a[:,None];y=b[None];wa=np.maximum(x[...,2:]-x[...,:2],0);wb=np.maximum(y[...,2:]-y[...,:2],0)
    intersection=np.maximum(np.minimum(x[...,2:],y[...,2:])-np.maximum(x[...,:2],y[...,:2]),0).prod(-1)
    aa=wa.prod(-1);ab=wb.prod(-1);union=aa+ab-intersection
    iou=np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)
    ca=(x[...,:2]+x[...,2:])/2;cb=(y[...,:2]+y[...,2:])/2;distance=((ca-cb)**2).sum(-1)
    diagonal=((np.maximum(x[...,2:],y[...,2:])-np.minimum(x[...,:2],y[...,:2]))**2).sum(-1)
    penalty=np.divide(distance,diagonal,out=np.zeros_like(distance),where=diagonal>0)
    valid=(aa>0)&(ab>0)
    return x,y,wa,wb,aa,ab,intersection,union,iou,distance,penalty,valid


def geometry_similarity(a,b,kind):
    if kind not in KINDS or kind=='native':raise ValueError('Native mode directly returns original tracker cost; explicit alternative required')
    x,y,wa,wb,aa,ab,intersection,union,iou,distance,penalty,valid=box_terms(a,b)
    if kind=='continuous_iou':score=iou
    elif kind=='diou':score=np.clip(iou-penalty,0,1)
    elif kind=='fish_affine_control':score=np.clip(1.3*iou+.3,0,1)
    elif kind=='iou_offset005':score=np.clip(iou+.05,0,1)
    elif kind=='agc_clipped':
        # Source is singular at identical centers and can exceed 1. Use its
        # clipped limit 1 for distance=0, otherwise exact source +eps+.05/dis.
        bonus=np.divide(.05,np.sqrt(distance),out=np.full_like(distance,np.inf),where=distance>0)
        score=np.clip(iou+1e-6+bonus,0,1)
    else:
        eps=1e-6
        sx=x[...,:2]+wa*np.array([.15,.3]);ex=x[...,2:]-wa*np.array([.25,.3])
        sy=y[...,:2]+wb*np.array([.15,.3]);ey=y[...,2:]-wb*np.array([.25,.3])
        lo=np.maximum(sx,sy);hi=np.minimum(ex,ey);overlap=np.maximum(hi-lo,0).prod(-1)
        if kind=='fish_source':central_a=(hi-lo).prod(-1)+eps
        else:central_a=(ex-sx).prod(-1)+eps
        central_b=(ey-sy).prod(-1)+eps
        central=overlap/(central_a+central_b-overlap+eps)
        aspect_a=wa[...,0]/(wa[...,1]+eps);aspect_b=wb[...,0]/(wb[...,1]+eps)
        aspect=np.minimum(aspect_a,aspect_b)/(np.maximum(aspect_a,aspect_b)+eps)
        area=np.minimum(aa,ab)/(np.maximum(aa,ab)+eps)
        source_iou=intersection/(union+eps)
        # Preserve the exact resolution-dependent source factor to isolate only
        # central-area repair; no hidden rescaling or tuned matching threshold.
        center=distance/((((np.maximum(x[...,2:],y[...,2:])-np.minimum(x[...,:2],y[...,:2]))**2).sum(-1))+eps)
        scale=1-np.exp(-np.minimum(aa,ab)/1000.)
        score=np.clip(source_iou+.3*central+.1*aspect+.2*area-.4*scale*center,0,1)
    score=np.where(valid,score,0.)
    if not np.isfinite(score).all():raise ValueError('Nonfinite geometry score; no silent dropped track/proposal')
    return score


def install_geometry_control(tracker,kind):
    """Replace only association geometry, preserve identity and its IoU graph."""
    from yolox.tracker import matching
    from .history import anchor_recent_history
    if kind not in KINDS or tracker.frame_id!=0 or tracker.memory_mode!='predictor':raise ValueError('Registered geometry before any frame required')
    base=type(tracker)
    class GeometryTracker(base):
        def association_cost(self,tracks,detections):
            baseline=super().association_cost(tracks,detections)
            if kind=='native' or not baseline.size:return baseline
            raw=matching.iou_distance(tracks,detections)
            original=matching.fuse_score(raw,detections) if not self.args.mot20 else raw
            boxes=np.array([t.tlbr for t in tracks]);current=np.array([d.tlbr for d in detections])
            similarity=geometry_similarity(boxes,current,kind);alternate=1-similarity
            if not self.args.mot20:alternate=matching.fuse_score(alternate,detections)
            # The baseline predictor's available rows are exactly nonempty
            # strictly-past valid identity histories; its candidate graph uses
            # the original inclusive IoU, never the new geometry score.
            factor=np.ones_like(original)
            for i,t in enumerate(tracks):
                history=list(t.identity_history)
                if self.history_view=='anchor_recent':
                    anchor=None if t.initial_identity_rgb is None else (t.initial_identity_frame,t.initial_identity_rgb,t.initial_identity_ir)
                    history=anchor_recent_history(history,anchor,8)
                if any(frame>=self.frame_id for frame,_,_ in history):raise ValueError('Score geometry before current identity writes')
                if history and self.appearance_weight:
                    eligible=np.array([d.current_rgb is not None for d in detections])&(raw[i]<=1-self.proximity_iou)
                    factor[i,eligible]=1-self.appearance_weight
            result=baseline+factor*(alternate-original)
            self.geometry_counts['frames']+=1;self.geometry_counts['edges']+=result.size
            self.geometry_counts['native_threshold_crossings']+=int(((baseline<=self.args.match_thresh)!=(result<=self.args.match_thresh)).sum())
            return result
    tracker.__class__=GeometryTracker
    tracker.geometry_kind=kind;tracker.geometry_counts=dict(frames=0,edges=0,native_threshold_crossings=0)
    return tracker
