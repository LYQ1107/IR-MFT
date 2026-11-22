"""Role-separated foreground ownership and classical RGB point transport.

Inputs are only a past frame and the current frame. Generated IR contributes
geometry, avoiding a brightness-constancy assumption on generated textures.
Classical primitives are prior art; gains require the registered ablations.
"""
import cv2
import numpy as np

from yolox.tracker.perceptual_features import foreground

MODES=('bbox','rgb','ir','joint')


def region_masks(bgr,ir,box):
    h,w=bgr.shape[:2]
    x1,y1=np.maximum(np.floor(box[:2]).astype(int),0)
    x2,y2=np.minimum(np.ceil(box[2:]).astype(int),[w,h])
    x1,y1=min(x1,w-1),min(y1,h-1);x2,y2=max(x1+1,x2),max(y1+1,y2)
    rgb=cv2.resize(bgr[y1:y2,x1:x2],(64,32))
    lab=cv2.cvtColor(rgb,cv2.COLOR_BGR2LAB).astype(np.float32)
    border=np.concatenate([lab[:3].reshape(-1,3),lab[-3:].reshape(-1,3),lab[:,:3].reshape(-1,3),lab[:,-3:].reshape(-1,3)])
    distance=np.linalg.norm(lab-np.median(border,axis=0),axis=-1).clip(0,255).astype(np.uint8)
    rgb_mask,q_rgb=foreground(distance)
    ir_mask,q_ir=foreground(ir[y1:y2,x1:x2])
    if q_rgb<.15:rgb_mask=np.ones_like(rgb_mask)
    if q_ir<.15:ir_mask=rgb_mask.copy()
    joint=np.sqrt(np.maximum(rgb_mask*ir_mask,0))
    return dict(bbox=np.ones_like(rgb_mask),rgb=rgb_mask,ir=ir_mask,joint=joint),[q_rgb,q_ir]


def point_groups(bgr,ir,boxes,maximum=20):
    masks={mode:[] for mode in MODES};qualities=[]
    groups={mode:dict(points=[],owners=[],weights=[],total=np.zeros(len(boxes),np.uint8),denominator=np.zeros(len(boxes),np.float32)) for mode in MODES}
    for owner,box in enumerate(boxes):
        local,quality=region_masks(bgr,ir,box);qualities.append(quality)
        for mode in MODES:masks[mode].append(local[mode])
        x1,y1=np.maximum(np.floor(box[:2]).astype(int),0)
        x2,y2=np.minimum(np.ceil(box[2:]).astype(int),[bgr.shape[1],bgr.shape[0]])
        if x2-x1<5 or y2-y1<5:continue
        crop=bgr[y1:y2,x1:x2]
        response=np.maximum.reduce([cv2.cornerMinEigenVal(crop[:,:,c],3) for c in range(3)])
        if not np.isfinite(response).all() or response.max()<=1e-8:continue
        maxima=response>=cv2.dilate(response,np.ones((3,3),np.uint8))
        for mode in MODES:
            mask=cv2.resize(local[mode],(crop.shape[1],crop.shape[0]))
            valid=maxima&(mask>=.25)&(response>=response.max()*.001)
            yy,xx=np.nonzero(valid)
            order=np.argsort(-response[yy,xx],kind='stable')[:200]
            chosen=[]
            for index in order:
                point=np.array([xx[index],yy[index]],np.float32)
                if chosen and np.min(np.linalg.norm(np.stack(chosen)-point,axis=1))<3:continue
                chosen.append(point)
                if len(chosen)>=maximum:break
            group=groups[mode]
            for point in chosen:
                weight=float(mask[int(point[1]),int(point[0])])
                group['points'].append(point+[x1,y1]);group['owners'].append(owner);group['weights'].append(weight)
                group['denominator'][owner]+=weight
            group['total'][owner]=len(chosen)
    for group in groups.values():
        group['points']=np.asarray(group['points'],np.float32).reshape(-1,2)
        group['owners']=np.asarray(group['owners'],int)
        group['weights']=np.asarray(group['weights'],np.float32)
    return dict(boxes=np.asarray(boxes,np.float32),masks=masks,quality=np.asarray(qualities,np.float32).reshape(-1,2),groups=groups)


def track_points(previous,current,points):
    if not len(points):return np.empty((0,2),np.float32),np.empty(0,bool),np.empty(0,np.float32)
    p0=np.ascontiguousarray(points,np.float32).reshape(-1,1,2)
    options=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,.02))
    p1,ok,error=cv2.calcOpticalFlowPyrLK(previous,current,p0,None,**options)
    if p1 is None:return points.copy(),np.zeros(len(points),bool),np.zeros(len(points),np.float32)
    back,ok_back,_=cv2.calcOpticalFlowPyrLK(current,previous,p1,None,**options)
    if back is None:return p1.reshape(-1,2),np.zeros(len(points),bool),np.zeros(len(points),np.float32)
    fb=np.linalg.norm((back-p0).reshape(-1,2),axis=1)
    good=ok.ravel().astype(bool)&ok_back.ravel().astype(bool)&(fb<1.5)&(error.ravel()<40)
    confidence=np.exp(-(fb/1.5)**2)*np.clip(1-error.ravel()/40,0,1)
    return p1.reshape(-1,2),good,confidence.astype(np.float32)


def vote_matrix(group,positions,good,confidence,target,mode):
    nsource=len(group['total']);ntarget=len(target['boxes'])
    matrix=np.zeros((nsource,ntarget),np.float32)
    counts=np.bincount(group['owners'][good],minlength=nsource).astype(np.uint8)
    if not len(positions) or not ntarget or not good.any():return matrix,counts
    selected=np.flatnonzero(good);p=positions[selected]
    ownership=np.zeros((ntarget,len(p)),np.float32)
    for column,box in enumerate(target['boxes']):
        size=np.maximum(box[2:]-box[:2],1)
        uv=(p-box[:2])/size*[64,32]
        inside=(uv[:,0]>=0)&(uv[:,0]<64)&(uv[:,1]>=0)&(uv[:,1]<32)
        indices=np.flatnonzero(inside)
        if len(indices):
            xy=uv[indices].astype(int);weights=target['masks'][mode][column][xy[:,1],xy[:,0]]
            ownership[column,indices]=np.where(weights>=.25,weights,0)
    # A background alternative prevents an arbitrarily weak foreground pixel
    # from becoming certain solely because no other candidate covers it.
    ownership/=ownership.sum(axis=0,keepdims=True)+.25
    weights=group['weights'][selected]*confidence[selected]
    owners=group['owners'][selected]
    for column in range(ntarget):
        matrix[:,column]=np.bincount(owners,weights=weights*ownership[column],minlength=nsource)
    matrix/=np.maximum(group['denominator'][:,None],1e-8)
    return np.clip(matrix,0,1),counts


def frame_correspondences(previous,current,past_regions,current_regions):
    """Pool independent point queries to build one RGB pyramid per frame pair."""
    keys=[];points=[];offsets=[0]
    for arm,regions in past_regions.items():
        for mode in MODES:
            keys.append((arm,mode));points.append(regions['groups'][mode]['points']);offsets.append(offsets[-1]+len(points[-1]))
    all_points=np.concatenate(points,axis=0) if points else np.empty((0,2),np.float32)
    if len(all_points):
        unique,inverse=np.unique(all_points,axis=0,return_inverse=True)
        positions,good,confidence=track_points(previous,current,unique)
        positions,good,confidence=positions[inverse],good[inverse],confidence[inverse]
    else:
        positions,good,confidence=track_points(previous,current,all_points)
    result={}
    for i,(arm,source_mode) in enumerate(keys):
        a,b=offsets[i:i+2];group=past_regions[arm]['groups'][source_mode]
        valid=good[a:b].copy();movement=positions[a:b]-group['points']
        for owner in range(len(group['total'])):
            pick=np.flatnonzero(valid&(group['owners']==owner))
            if len(pick)<3:continue
            median=np.median(movement[pick],axis=0)
            tolerance=max(2.5,.15*float(np.min(past_regions[arm]['boxes'][owner,2:]-past_regions[arm]['boxes'][owner,:2])))
            valid[pick[np.linalg.norm(movement[pick]-median,axis=1)>tolerance]]=False
        for target_mode in MODES:
            result[(arm,source_mode,target_mode)]=vote_matrix(group,positions[a:b],valid,confidence[a:b],current_regions[arm],target_mode)
    return result
