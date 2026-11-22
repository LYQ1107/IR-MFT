"""Shared pixel preparation and losses for matched RGB/native-new-IR branches."""
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from yolox.tracker.perceptual_features import clipped_crop


def native_crop(image,box,size=(128,256)):
    patch=clipped_crop(image,box,.05)
    if patch.ndim==3:patch=cv2.cvtColor(patch,cv2.COLOR_BGR2RGB)
    if patch.ndim==2:patch=patch[:,:,None]
    height,width=size;ratio=min(height/patch.shape[0],width/patch.shape[1])
    rh,rw=max(1,round(patch.shape[0]*ratio)),max(1,round(patch.shape[1]*ratio))
    dy,dx=(height-rh)//2,(width-rw)//2
    resized=cv2.resize(patch,(rw,rh))
    if resized.ndim==2:resized=resized[:,:,None]
    canvas=np.zeros((height,width,patch.shape[2]),np.uint8);canvas[dy:dy+rh,dx:dx+rw]=resized
    return canvas,np.array([dy,dx,rh,rw],np.int16)


def native_input(pixels,rectangles,device):
    value=pixels.to(device=device,dtype=torch.float32).permute(0,3,1,2)/255
    if value.shape[1]==1:value=value.expand(-1,3,-1,-1)
    if value.shape[1:]!=(3,128,256):raise ValueError('Native cue crop shape')
    mean=value.new_tensor([.485,.456,.406])[None,:,None,None]
    std=value.new_tensor([.229,.224,.225])[None,:,None,None]
    rectangle=rectangles.to(device=device);dy,dx,rh,rw=[rectangle[:,i,None,None] for i in range(4)]
    yy=torch.arange(128,device=device)[None,:,None];xx=torch.arange(256,device=device)[None,None,:]
    valid=(yy>=dy)&(yy<dy+rh)&(xx>=dx)&(xx<dx+rw)
    return ((value-mean)/std)*valid[:,None]


def native_forward(model,value):
    raw=model.projection(model.backbone(value))
    identity=F.normalize(raw.float(),dim=-1)
    logits=model.classifier(model.neck(raw))
    return identity,logits


def same_video_triplet(identity,labels,video,frame):
    distance=(1-identity@identity.T).clamp(0,2)
    same=labels[:,None]==labels[None,:]
    positive=same&((frame[:,None]-frame[None,:]).abs()>=25)
    negative=(~same)&(video[:,None]==video[None,:])
    valid=positive.any(1)&negative.any(1)
    if not valid.any():return identity.sum()*0,0.
    hardest_positive=distance.masked_fill(~positive,-torch.inf).max(1).values
    hardest_negative=distance.masked_fill(~negative,torch.inf).min(1).values
    loss=F.relu(hardest_positive[valid]-hardest_negative[valid]+.3).mean()
    return loss,float(valid.float().mean())
