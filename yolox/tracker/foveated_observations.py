"""Current-image coarse attention and RGB crop observation integration."""
import cv2
import numpy as np

from yolox.tracker.matching import ious,linear_assignment


def coarse_attention(image,boxes,scores,frame_shape,map_shape=(180,320)):
    height,width=frame_shape[:2];h,w=map_shape
    small=cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA)
    if small.ndim==3:small=cv2.cvtColor(small,cv2.COLOR_BGR2LAB)
    else:small=small[...,None]
    value=small.astype(np.float32);median=np.median(value.reshape(-1,value.shape[-1]),axis=0)
    mad=np.maximum(np.median(abs(value-median).reshape(-1,value.shape[-1]),axis=0),5.)
    heat=np.sqrt((((value-median)/mad)**2).sum(-1))
    heat=cv2.GaussianBlur(heat,(5,5),1.)
    heat=np.clip(heat/max(float(np.quantile(heat,.95)),1e-6),0,1)
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


def attention_windows(heat,frame_shape,count=4,fraction=.5,stride=4):
    height,width=frame_shape[:2];h,w=heat.shape
    ch,cw=max(1,round(h*fraction)),max(1,round(w*fraction))
    residual=np.maximum(heat,0).astype(np.float32).copy();windows=[]
    ys=np.unique(np.r_[np.arange(0,h-ch+1,stride),h-ch]);xs=np.unique(np.r_[np.arange(0,w-cw+1,stride),w-cw])
    yy,xx=np.meshgrid(ys,xs,indexing='ij')
    for _ in range(count):
        integral=cv2.integral(residual)
        mass=integral[yy+ch,xx+cw]-integral[yy,xx+cw]-integral[yy+ch,xx]+integral[yy,xx]
        for previous in windows:
            px=round(previous[0]/width*w);py=round(previous[1]/height*h)
            mass[(xx==px)&(yy==py)]=-1
        if mass.max()<=0:
            fallback=next((box for box in uniform_windows(frame_shape) if box not in windows),None)
            if fallback is not None:
                windows.append(fallback)
                fx1,fy1,fx2,fy2=fallback
                residual[round(fy1/height*h):round(fy2/height*h),round(fx1/width*w):round(fx2/width*w)]=0
                continue
        # Equal-mass windows should put their evidence near the foveal center,
        # rather than selecting the first top-left coordinate on a plateau.
        moment_x=cv2.integral(residual*np.arange(w,dtype=np.float32)[None])
        moment_y=cv2.integral(residual*np.arange(h,dtype=np.float32)[:,None])
        def rectangle_sum(value):
            return value[yy+ch,xx+cw]-value[yy,xx+cw]-value[yy+ch,xx]+value[yy,xx]
        cx=rectangle_sum(moment_x)/np.maximum(mass,1e-8)
        cy=rectangle_sum(moment_y)/np.maximum(mass,1e-8)
        distance=((cx-(xx+(cw-1)/2))/cw)**2+((cy-(yy+(ch-1)/2))/ch)**2
        tied=mass>=mass.max()-max(1.,float(mass.max()))*1e-6
        best=np.unravel_index(int(np.where(tied,distance,np.inf).argmin()),mass.shape);y,x=int(yy[best]),int(xx[best])
        x1=round(x/w*width);y1=round(y/h*height)
        x2=min(width,x1+round(width*fraction));y2=min(height,y1+round(height*fraction))
        windows.append((x1,y1,x2,y2));residual[y:y+ch,x:x+cw]=0
    return windows


def uniform_windows(frame_shape):
    height,width=frame_shape[:2];hh,ww=height//2,width//2
    return [(0,0,ww,hh),(width-ww,0,width,hh),
            (0,height-hh,ww,height),(width-ww,height-hh,width,height)]


def nms_observations(detections,threshold=.7):
    detections=np.asarray(detections,np.float32).reshape(-1,5)
    order=np.argsort(-detections[:,4],kind='stable');keep=[]
    while len(order):
        i=int(order[0]);keep.append(i)
        if len(order)==1:break
        overlap=ious(detections[i:i+1,:4],detections[order[1:],:4])[0]
        order=order[1:][overlap<=threshold]
    return detections[keep]


def integrate_crops(base,views,refine=True):
    """One-to-one across views, never merging two predictions in one view."""
    base=np.asarray(base,np.float32).reshape(-1,5)
    if not views:return base.copy()
    if not refine:
        return nms_observations(np.concatenate([base]+[np.asarray(d,np.float32).reshape(-1,5) for d,_ in views]))
    sums=[d[:4].astype(np.float64)*float(d[4]) for d in base]
    weights=[float(d[4]) for d in base];scores=base[:,4].tolist();coordinates=[d[:4].copy() for d in base]
    for detections,resolution in views:
        detections=np.asarray(detections,np.float32).reshape(-1,5)
        if not len(detections):continue
        matrix=ious(np.asarray(coordinates,np.float32).reshape(-1,4),detections[:,:4])
        matches,_,unused=linear_assignment(1-matrix,.5)
        for i,j in matches:
            weight=float(detections[j,4])*float(resolution)**2
            sums[i]+=detections[j,:4]*weight;weights[i]+=weight
            coordinates[i]=(sums[i]/weights[i]).astype(np.float32);scores[i]=max(scores[i],float(detections[j,4]))
        for j in unused:
            weight=float(detections[j,4])*float(resolution)**2
            sums.append(detections[j,:4].astype(np.float64)*weight);weights.append(weight)
            coordinates.append(detections[j,:4].copy());scores.append(float(detections[j,4]))
    return nms_observations(np.column_stack([np.asarray(coordinates).reshape(-1,4),scores]).astype(np.float32))
