"""Pool unchanged RGB features using independently measured spatial evidence."""
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from yolox.tracker.perceptual_features import foreground


def mask_axis(mask,patch_shape):
    height,width=patch_shape[:2];h,w=mask.shape
    yy,xx=np.mgrid[:h,:w];xy=np.stack([(xx+.5)*width/w,(yy+.5)*height/h],axis=-1)
    weight=np.maximum(np.asarray(mask,np.float64),0)
    if weight.sum()<1e-6:weight=np.ones_like(weight)
    center=(xy*weight[...,None]).sum((0,1))/weight.sum();relative=xy-center
    covariance=np.einsum('hwi,hwj,hw->ij',relative,relative,weight)/weight.sum()
    eigenvalue,eigenvector=np.linalg.eigh(covariance);axis=eigenvector[:,-1]
    if axis[0]<0 or (abs(axis[0])<1e-8 and axis[1]<0):axis=-axis
    coherence=float((eigenvalue[1]-eigenvalue[0])/max(float(eigenvalue.sum()),1e-8))
    scale=max(2*np.sqrt(max(float(eigenvalue[-1]),0)),1.)
    return center,axis,coherence,scale


def spatial_weights(bgr_patch,ir_patch=None,canvas_shape=(128,256),map_shape=(4,8)):
    """Return three pooling weights and nine quality values per available mode.

    RGB outputs never consume IR pixels. Geometry uses physical crop aspect,
    and the spatial weights share the exact existing RGB letterbox placement.
    """
    lab=cv2.cvtColor(cv2.resize(bgr_patch,(64,32)),cv2.COLOR_BGR2LAB).astype(np.float32)
    border=np.concatenate([lab[:3].reshape(-1,3),lab[-3:].reshape(-1,3),lab[:,:3].reshape(-1,3),lab[:,-3:].reshape(-1,3)])
    distance=np.linalg.norm(lab-np.median(border,axis=0),axis=-1).clip(0,255).astype(np.uint8)
    RGB_mask,q_RGB=foreground(distance)
    masks=dict(box=np.ones_like(RGB_mask),rgb=RGB_mask);quality=dict(box=1.,rgb=q_RGB)
    if ir_patch is not None:masks['ir'],quality['ir']=foreground(ir_patch)
    height,width=bgr_patch.shape[:2];H,W=canvas_shape;h,w=map_shape
    ratio=min(H/height,W/width);rh,rw=max(1,round(height*ratio)),max(1,round(width*ratio))
    dy,dx=(H-rh)//2,(W-rw)//2
    yy,xx=np.mgrid[:h,:w]
    positions=np.stack([((xx+.5)*W/w-dx)/ratio,((yy+.5)*H/h-dy)/ratio],axis=-1)
    valid=np.zeros((H,W),np.float32);valid[dy:dy+rh,dx:dx+rw]=1
    valid=cv2.resize(valid,(w,h),interpolation=cv2.INTER_AREA)
    RGB_center,RGB_axis,_,_=mask_axis(RGB_mask,bgr_patch.shape)
    result={}
    for mode,mask in masks.items():
        center,axis,coherence,scale=mask_axis(mask,bgr_patch.shape)
        canvas=np.zeros((H,W),np.float32)
        canvas[dy:dy+rh,dx:dx+rw]=cv2.resize(mask,(rw,rh))
        down=cv2.resize(canvas,(w,h),interpolation=cv2.INTER_AREA)
        support=.2*valid+.8*down
        projection=((positions-center)*axis).sum(-1)
        second=1/(1+np.exp(-np.clip(projection/max(.15*scale,1.),-20,20)))
        parts=np.stack([support*(1-second),support*second])
        global_weight=np.ones((h,w),np.float32) if mode=='box' else support
        weights=np.concatenate([global_weight[None],parts],axis=0).astype(np.float32)
        agreement=float(2*(mask*RGB_mask).sum()/max(float((mask*mask).sum()+(RGB_mask*RGB_mask).sum()),1e-8))
        mass=parts.sum((1,2))/max(float(support.sum()),1e-8)
        q=np.array([quality[mode],(mask>.35).mean(),coherence,center[0]/width,center[1]/height,
                    agreement,abs(float(axis@RGB_axis)),mass[0],mass[1]],np.float32)
        if weights.shape!=(3,h,w) or q.shape!=(9,) or not np.isfinite(q).all():raise ValueError('Invalid spatial tokens')
        result[mode]=dict(weights=weights,quality=q)
    return result


def pool_identity_tokens(feature_map,weights,projection):
    """Apply the same preserved RGB projection after spatial integration."""
    if weights.ndim!=4 or feature_map.ndim!=4 or feature_map.shape[0]!=weights.shape[0] or feature_map.shape[-2:]!=weights.shape[-2:]:
        raise ValueError('Feature map and ownership weights do not align')
    weights=weights.float();pooled=torch.einsum('nchw,nkhw->nkc',feature_map.float(),weights)
    pooled=pooled/weights.sum((-1,-2)).clamp_min(1e-8)[...,None]
    return F.normalize(F.linear(pooled,projection.weight.float()),dim=-1)
