"""Fixed classical RGB motion on the detector's actual padded canvas."""
import cv2
import numpy as np


def padded_motion_view(bgr,input_hw=(800,1440),downsample=4):
    h,w=input_hw;ratio=min(h/bgr.shape[0],w/bgr.shape[1])
    resized=cv2.resize(bgr,(int(bgr.shape[1]*ratio),int(bgr.shape[0]*ratio)),interpolation=cv2.INTER_LINEAR)
    canvas=np.full((h,w,3),114,np.uint8);canvas[:resized.shape[0],:resized.shape[1]]=resized
    gray=cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray,(w//downsample,h//downsample),interpolation=cv2.INTER_AREA)


def backward_rgb_motion(previous,current,fb_tolerance=1.5):
    if previous.shape!=current.shape or previous.ndim!=2 or previous.dtype!=np.uint8 or current.dtype!=np.uint8:
        raise ValueError('Matched current/past uint8 grayscale canvases required')
    params=(.5,3,15,3,5,1.2,0)
    forward=cv2.calcOpticalFlowFarneback(previous,current,None,*params)
    backward=cv2.calcOpticalFlowFarneback(current,previous,None,*params)
    y,x=np.mgrid[:current.shape[0],:current.shape[1]]
    px=x.astype(np.float32)+backward[:,:,0];py=y.astype(np.float32)+backward[:,:,1]
    sampled=cv2.remap(forward,px,py,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    error=np.linalg.norm(backward+sampled,axis=-1)
    valid=(np.isfinite(backward).all(-1)&np.isfinite(error)&(error<fb_tolerance)
           &(px>=0)&(px<=current.shape[1]-1)&(py>=0)&(py<=current.shape[0]-1))
    return backward,valid
