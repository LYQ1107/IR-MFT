"""Spatial working-region checks and exact half-precision detector input."""
import cv2
import numpy as np

from yolox.tracker.foveated_observations import attention_windows,uniform_windows,integrate_crops


def exact_half_preproc(image,input_size,mean,std):
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('Exact lookup preprocessing requires uint8 BGR input')
    ratio=min(input_size[0]/image.shape[0],input_size[1]/image.shape[1])
    h,w=int(image.shape[0]*ratio),int(image.shape[1]*ratio)
    resized=cv2.resize(image,(w,h),interpolation=cv2.INTER_LINEAR)
    padded=np.full((input_size[0],input_size[1],3),114,np.uint8);padded[:h,:w]=resized
    # The original float64 normalization is followed by float32 then half().
    # Every possible byte value has an exactly equivalent lookup entry.
    values=np.arange(256,dtype=np.float64)/255
    lookup=((values[None]-np.asarray(mean,np.float64)[:,None])/np.asarray(std,np.float64)[:,None]).astype(np.float32).astype(np.float16)
    result=np.empty((3,*input_size),np.float16)
    for channel in range(3):result[channel]=lookup[channel][padded[:,:,2-channel]]
    return result,ratio


def working_region(rgb,boxes,scores,map_shape=(180,320)):
    h,w=map_shape;height,width=rgb.shape[:2]
    small=cv2.cvtColor(cv2.resize(rgb,(w,h),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2LAB).astype(np.float32)
    median=np.median(small.reshape(-1,3),axis=0)
    deviation=np.maximum(np.median(abs(small-median).reshape(-1,3),axis=0),5.)
    distance=np.sqrt((((small-median)/deviation)**2).sum(-1))
    background=(distance<=3.).astype(np.uint8)
    background=cv2.morphologyEx(background,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
    contours,_=cv2.findContours(background,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    support=np.zeros((h,w),np.uint8)
    if contours:cv2.drawContours(support,[max(contours,key=cv2.contourArea)],-1,1,cv2.FILLED)
    if support.mean()<.25:return np.ones((h,w),np.float32)
    # Known current RGB observations remain eligible even outside the dominant
    # scene region. This region check cannot delete an existing RGB detection.
    for box,score in zip(boxes,scores):
        if score<.6:continue
        x1,y1,x2,y2=np.rint(np.asarray(box)*[w/width,h/height,w/width,h/height]).astype(int)
        x1,x2=np.clip([x1,x2],0,w);y1,y2=np.clip([y1,y2],0,h)
        support[y1:y2,x1:x2]=1
    return support.astype(np.float32)


def coarse_attention(image,boxes,scores,frame_shape,support=None,map_shape=(180,320)):
    height,width=frame_shape[:2];h,w=map_shape
    if support is None:support=np.ones((h,w),np.float32)
    small=cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA)
    if small.ndim==3:small=cv2.cvtColor(small,cv2.COLOR_BGR2LAB)
    else:small=small[...,None]
    value=small.astype(np.float32);valid=support>.5
    median=np.median(value[valid],axis=0)
    mad=np.maximum(np.median(abs(value[valid]-median),axis=0),5.)
    heat=cv2.GaussianBlur(np.sqrt((((value-median)/mad)**2).sum(-1)),(5,5),1.)
    heat=np.clip(heat/max(float(np.quantile(heat[valid],.95)),1e-6),0,1)*support
    prior=np.zeros((h,w),np.float32)
    for box,score in zip(boxes,scores):
        x1,y1,x2,y2=np.rint(np.asarray(box)*[w/width,h/height,w/width,h/height]).astype(int)
        x1,x2=np.clip([x1,x2],0,w);y1,y2=np.clip([y1,y2],0,h)
        if x2<=x1 or y2<=y1:continue
        if score>=.6:
            heat[y1:y2,x1:x2]*=.25
            diagonal=max(float(np.linalg.norm(np.asarray(box)[2:]-np.asarray(box)[:2])),1.)
            need=.25*min(1.,64/(diagonal*min(800/height,1440/width)))
        else:need=1-float(score)
        prior[y1:y2,x1:x2]+=need/((x2-x1)*(y2-y1))
    positive=prior[prior>0]
    if len(positive):prior=np.clip(prior/max(float(np.quantile(positive,.95)),1e-6),0,1)
    return (heat+prior).astype(np.float32)
