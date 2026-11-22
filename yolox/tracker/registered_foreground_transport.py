"""Uncertainty-gated cross-modal ROI registration for body ownership."""
import cv2
import numpy as np

from yolox.tracker.foreground_transport import MODES,region_masks
from yolox.tracker.perceptual_features import foreground


def gradient_magnitude(image):
    if image.ndim==2:image=image[:,:,None]
    values=[]
    for channel in range(image.shape[2]):
        plane=image[:,:,channel].astype(np.float32)/255
        x=cv2.Scharr(plane,cv2.CV_32F,1,0);y=cv2.Scharr(plane,cv2.CV_32F,0,1)
        values.append(cv2.magnitude(x,y))
    value=np.maximum.reduce(values)
    return np.clip(value/max(float(np.percentile(value,95)),1e-5),0,2).astype(np.float32)


def registered_masks(bgr,ir,box,alignment):
    masks,quality=region_masks(bgr,ir,box);h,w=ir.shape;extent=np.maximum(box[2:]-box[:2],1)
    lower=np.maximum(np.floor(box[:2]-.15*extent),0).astype(int)
    upper=np.minimum(np.ceil(box[2:]+.15*extent),[w,h]).astype(int)
    x1,y1=lower;x2,y2=upper
    diagnostic=np.zeros(5,np.float32)
    if x2-x1<8 or y2-y1<8:return masks,quality,diagnostic
    first=gradient_magnitude(cv2.resize(bgr[y1:y2,x1:x2],(96,64)))
    second=gradient_magnitude(cv2.resize(ir[y1:y2,x1:x2],(96,64)))
    if first.std()<1e-6 or second.std()<1e-6:return masks,quality,diagnostic
    window=cv2.createHanningWindow((96,64),cv2.CV_32F)
    (dx,dy),response=cv2.phaseCorrelate(first.copy(),second.copy(),window)
    if not np.isfinite([dx,dy,response]).all():return masks,quality,diagnostic
    displacement=np.array([dx*(x2-x1)/96,dy*(y2-y1)/64],np.float32)
    relative=displacement/extent
    warped=cv2.warpAffine(second,np.float32([[1,0,-dx],[0,1,-dy]]),(96,64))
    def correlation(a,b):
        return float(np.corrcoef(a.ravel(),b.ravel())[0,1]) if a.std()>1e-6 and b.std()>1e-6 else 0.
    gain=correlation(first,warped)-correlation(first,second)
    diagnostic[:4]=[relative[0],relative[1],response,gain]
    if response<alignment['minimum_response'] or gain<alignment['minimum_correlation_gain']:return masks,quality,diagnostic
    if max(abs(relative))>alignment['maximum_box_shift'] or max(abs(relative))<alignment['minimum_box_shift']:return masks,quality,diagnostic
    x1,y1=np.maximum(np.floor(box[:2]),0).astype(int);x2,y2=np.minimum(np.ceil(box[2:]),[w,h]).astype(int)
    x1,y1=min(x1,w-1),min(y1,h-1);x2,y2=max(x1+1,x2),max(y1+1,y2)
    centre=(np.array([x1+x2-1,y1+y2-1],np.float32)/2+displacement)
    sample=cv2.getRectSubPix(ir,(x2-x1,y2-y1),(float(centre[0]),float(centre[1])))
    ir_mask,q_ir=foreground(sample)
    if q_ir<.15:ir_mask=masks['rgb'].copy()
    masks['ir']=ir_mask;masks['joint']=np.sqrt(np.maximum(masks['rgb']*ir_mask,0))
    quality[1]=q_ir;diagnostic[4]=1
    return masks,quality,diagnostic


def point_groups(bgr,ir,boxes,maximum=20,alignment=None):
    if alignment is None:raise ValueError('Explicit registered alignment policy required')
    masks={mode:[] for mode in MODES};qualities=[];diagnostics=[]
    groups={mode:dict(points=[],owners=[],weights=[],total=np.zeros(len(boxes),np.uint8),denominator=np.zeros(len(boxes),np.float32)) for mode in MODES}
    for owner,box in enumerate(boxes):
        local,quality,diagnostic=registered_masks(bgr,ir,box,alignment);qualities.append(quality);diagnostics.append(diagnostic)
        for mode in MODES:masks[mode].append(local[mode])
        x1,y1=np.maximum(np.floor(box[:2]).astype(int),0);x2,y2=np.minimum(np.ceil(box[2:]).astype(int),[bgr.shape[1],bgr.shape[0]])
        if x2-x1<5 or y2-y1<5:continue
        crop=bgr[y1:y2,x1:x2]
        response=np.maximum.reduce([cv2.cornerMinEigenVal(crop[:,:,c],3) for c in range(3)])
        if not np.isfinite(response).all() or response.max()<=1e-8:continue
        maxima=response>=cv2.dilate(response,np.ones((3,3),np.uint8))
        for mode in MODES:
            mask=cv2.resize(local[mode],(crop.shape[1],crop.shape[0]));valid=maxima&(mask>=.25)&(response>=response.max()*.001)
            yy,xx=np.nonzero(valid);order=np.argsort(-response[yy,xx],kind='stable')[:200];chosen=[]
            for index in order:
                point=np.array([xx[index],yy[index]],np.float32)
                if chosen and np.min(np.linalg.norm(np.stack(chosen)-point,axis=1))<3:continue
                chosen.append(point)
                if len(chosen)>=maximum:break
            group=groups[mode]
            for point in chosen:
                weight=float(mask[int(point[1]),int(point[0])]);group['points'].append(point+[x1,y1]);group['owners'].append(owner);group['weights'].append(weight);group['denominator'][owner]+=weight
            group['total'][owner]=len(chosen)
    for group in groups.values():
        group['points']=np.asarray(group['points'],np.float32).reshape(-1,2);group['owners']=np.asarray(group['owners'],int);group['weights']=np.asarray(group['weights'],np.float32)
    return dict(boxes=np.asarray(boxes,np.float32),masks=masks,quality=np.asarray(qualities,np.float32).reshape(-1,2),alignment=np.asarray(diagnostics,np.float32).reshape(-1,5),groups=groups)
