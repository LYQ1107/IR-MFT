"""Current-proposal DINOv2 descriptors; original detector boxes only."""
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from yolox.tracker.perceptual_features import clipped_crop

VIEWS=('RGB','IR','gray')
DIM=768


def square_crop(image,box,*,bgr,size=224):
    patch=clipped_crop(image,box,.05)
    if patch.ndim==2:patch=np.repeat(patch[:,:,None],3,axis=2)
    elif bgr:patch=cv2.cvtColor(patch,cv2.COLOR_BGR2RGB)
    assert patch.ndim==3 and patch.shape[2]==3 and patch.dtype==np.uint8
    ratio=min(size/patch.shape[0],size/patch.shape[1]);h,w=[max(1,round(x*ratio))for x in patch.shape[:2]]
    y,x=(size-h)//2,(size-w)//2
    out=np.zeros((size,size,3),np.uint8);out[y:y+h,x:x+w]=cv2.resize(patch,(w,h))
    return out,np.array([y,x,h,w],np.int16)


def input_tensors(pixels,rectangles,device):
    p=torch.as_tensor(np.stack(pixels),device=device,dtype=torch.float32).permute(0,3,1,2)/255
    r=torch.as_tensor(np.stack(rectangles),device=device)
    y,x,h,w=[r[:,i,None,None]for i in range(4)]
    yy=torch.arange(224,device=device)[None,:,None];xx=torch.arange(224,device=device)[None,None,:]
    valid=(yy>=y)&(yy<y+h)&(xx>=x)&(xx<x+w)
    mean=p.new_tensor([.485,.456,.406])[None,:,None,None];std=p.new_tensor([.229,.224,.225])[None,:,None,None]
    normalized=((p-mean)/std)*valid[:,None]
    weights=F.avg_pool2d(valid[:,None].float(),14,14).flatten(1)
    assert weights.shape[1]==256 and(weights.sum(1)>0).all()
    return normalized,weights


@torch.no_grad()
def encode(model,rgb,ir,boxes,batch_size=16):
    assert rgb.dtype==ir.dtype==np.uint8 and ir.ndim==2 and rgb.shape[:2]==ir.shape
    images={'RGB':rgb,'IR':ir,'gray':cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)};out={}
    for view,image in images.items():
        cls=[];mean=[]
        for lo in range(0,len(boxes),batch_size):
            crops=[square_crop(image,b,bgr=view=='RGB')for b in boxes[lo:lo+batch_size]]
            x,w=input_tensors([c[0]for c in crops],[c[1]for c in crops],'cuda')
            f=model.forward_features(x);tokens=f['x_norm_patchtokens'].float()
            assert tokens.shape==(len(crops),256,DIM)
            pooled=(tokens*w[:,:,None]).sum(1)/w.sum(1)[:,None]
            cls.append(F.normalize(f['x_norm_clstoken'].float(),dim=-1).cpu().numpy())
            mean.append(F.normalize(pooled,dim=-1).cpu().numpy())
        for kind,values in [('cls',cls),('mean',mean)]:
            a=np.concatenate(values).astype(np.float32)if values else np.empty((0,DIM),np.float32)
            assert a.shape==(len(boxes),DIM)and np.isfinite(a).all()
            out[view+'_'+kind]=a
    return out


def load_model(source,checkpoint):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model=torch.hub.load(str(source),'dinov2_vitb14',source='local',pretrained=False)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    return model.cuda().float().eval().requires_grad_(False)
