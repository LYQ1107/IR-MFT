"""IR silhouette anchors with RGB-first causal point transport.

The original RGB point packets are retained. The extra joint-body packet may
use corners from actual IR; native IR flow is an explicit removal/fallback
control, rather than replacing valid physical RGB correspondences.
"""
import cv2
import numpy as np

from yolox.tracker.foreground_transport import point_groups, track_points, vote_matrix


VARIANTS = ('semantic_rgb', 'semantic_safe', 'semantic_ir')


def semantic_groups(bgr, ir, boxes, maximum=20):
    regions = point_groups(bgr, ir, boxes, maximum=maximum)
    old = regions['groups']['joint']
    if float(ir.std()) < 1e-6:
        regions['semantic_joint'] = dict(old, native=np.zeros(len(old['points']), bool))
        return regions
    group = dict(points=[], owners=[], weights=[], native=[],
                 total=np.zeros(len(boxes),np.uint8), denominator=np.zeros(len(boxes),np.float32))
    for owner, box in enumerate(boxes):
        x1,y1=np.maximum(np.floor(box[:2]).astype(int),0)
        x2,y2=np.minimum(np.ceil(box[2:]).astype(int),[bgr.shape[1],bgr.shape[0]])
        if x2-x1<5 or y2-y1<5:
            continue
        crop=ir[y1:y2,x1:x2]
        response=cv2.cornerMinEigenVal(crop,3)
        native=regions['quality'][owner,1]>=.15 and np.isfinite(response).all() and response.max()>1e-8
        if native:
            mask=cv2.resize(regions['masks']['joint'][owner],(crop.shape[1],crop.shape[0]))
            maxima=response>=cv2.dilate(response,np.ones((3,3),np.uint8))
            yy,xx=np.nonzero(maxima&(mask>=.25)&(response>=response.max()*.001))
            chosen=[]
            for index in np.argsort(-response[yy,xx],kind='stable')[:200]:
                point=np.array([xx[index],yy[index]],np.float32)
                if chosen and np.min(np.linalg.norm(np.stack(chosen)-point,axis=1))<3:
                    continue
                chosen.append(point)
                if len(chosen)>=maximum:
                    break
            points=np.asarray(chosen,np.float32).reshape(-1,2)
            weights=np.array([mask[int(p[1]),int(p[0])] for p in points],np.float32)
            points=points+[x1,y1]
        else:
            select=old['owners']==owner
            points,weights=old['points'][select],old['weights'][select]
        group['points'].extend(points)
        group['weights'].extend(weights)
        group['owners'].extend([owner]*len(points))
        group['native'].extend([native]*len(points))
        group['total'][owner]=len(points)
        group['denominator'][owner]=sum(weights)
    for field,dtype in [('points',np.float32),('weights',np.float32),('owners',int),('native',bool)]:
        group[field]=np.asarray(group[field],dtype=dtype)
    group['points']=group['points'].reshape(-1,2)
    regions['semantic_joint']=group
    return regions


def semantic_votes(previous_rgb,current_rgb,previous_ir,current_ir,past,target):
    group=past['semantic_joint']
    points=group['points']
    rgb_positions,rgb_good,rgb_confidence=track_points(previous_rgb,current_rgb,points)
    if group['native'].any():
        ir_positions,ir_good,ir_confidence=track_points(previous_ir,current_ir,points)
        ir_good=ir_good&group['native']
    else:
        ir_positions,ir_good,ir_confidence=rgb_positions,np.zeros(len(points),bool),np.zeros(len(points),np.float32)
    result={}
    for variant in VARIANTS:
        positions=rgb_positions.copy()
        good=rgb_good.copy()
        confidence=rgb_confidence.copy()
        if variant=='semantic_safe':
            use=(~rgb_good)&ir_good
        elif variant=='semantic_ir':
            use=group['native']
        else:
            use=np.zeros(len(points),bool)
        positions[use]=ir_positions[use]
        good[use]=ir_good[use]
        confidence[use]=ir_confidence[use]
        movement=positions-points
        # Same per-body robust coherence check as the original point control.
        for owner in range(len(group['total'])):
            pick=np.flatnonzero(good&(group['owners']==owner))
            if len(pick)<3:
                continue
            median=np.median(movement[pick],axis=0)
            tolerance=max(2.5,.15*float(np.min(past['boxes'][owner,2:]-past['boxes'][owner,:2])))
            good[pick[np.linalg.norm(movement[pick]-median,axis=1)>tolerance]]=False
        matrix,counts=vote_matrix(group,positions,good,confidence,target,'rgb')
        result[variant]=dict(matrix=matrix,good=counts,total=group['total'],
            native_points=int(group['native'].sum()),IR_fallback_points=int(use.sum()),
            RGB_point_successes=int(rgb_good.sum()),IR_point_successes=int(ir_good.sum()))
    return result
