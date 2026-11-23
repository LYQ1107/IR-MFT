"""Causal physical transport of already bounded auxiliary corrections.

This is a candidate inference mechanism, not a claim of a new flow algorithm.
RGB flow is supplied by the caller; the module never reads an image or label.
"""
import torch
from torch.nn import functional as F


def warp_previous(value,backward_flow,validity):
    """Current-to-previous flow, in supplied grid pixels, rescaled per FPN level."""
    h,w=value.shape[-2:];fh,fw=backward_flow.shape[-2:]
    flow=F.interpolate(backward_flow.float(),size=(h,w),mode='bilinear',align_corners=False)
    flow=flow*flow.new_tensor([w/fw,h/fh]).reshape(1,2,1,1)
    y,x=torch.meshgrid(torch.arange(h,device=value.device),torch.arange(w,device=value.device),indexing='ij')
    px=x[None].float()+flow[:,0];py=y[None].float()+flow[:,1]
    grid=torch.stack([2*px/max(w-1,1)-1,2*py/max(h-1,1)-1],-1)
    inside=(px>=0)&(px<=w-1)&(py>=0)&(py<=h-1)
    ok=F.interpolate(validity.float(),size=(h,w),mode='nearest')>.5
    warped=F.grid_sample(value.float(),grid,mode='bilinear',padding_mode='zeros',align_corners=True)
    return warped,ok&inside[:,None]


class TemporalCorrection:
    def __init__(self,mode='flow',strength=.5,bound=.5,epsilon=1e-6):
        if mode not in {'none','identity','flow','without_current_evidence'}:raise ValueError('Unknown temporal policy')
        if not 0<=strength<=1 or bound<=0:raise ValueError('Bounded convex update required')
        self.mode=mode;self.strength=strength;self.bound=bound;self.epsilon=epsilon;self.previous=None

    def reset(self):self.previous=None

    def update(self,rgb,repaired,trust,backward_flow=None,validity=None,available=True):
        # Missing current evidence clears the state before inspecting a past
        # correction or a repaired/auxiliary tensor.
        if not available:
            self.reset();return tuple(rgb)
        if self.mode=='none'or self.strength==0:return tuple(repaired)
        if len(rgb)!=len(repaired)or len(rgb)!=len(trust):raise ValueError('Matched FPN levels required')
        if any(x.shape[0]!=1 for x in rgb):raise ValueError('Update independent frames sequentially, then batch the head')
        if self.previous is not None and (len(self.previous)!=len(rgb)or any(p['value'].shape!=x.shape or p['value'].device!=x.device for p,x in zip(self.previous,rgb))):
            self.reset()
        output=[];next_state=[]
        for index,(base,current,confidence)in enumerate(zip(rgb,repaired,trust)):
            value=base.float();confidence=confidence.detach().float().clamp(0,1)
            rms=value.square().mean(1,keepdim=True).add(1e-6).sqrt()
            normalized=(current.float()-value)/rms
            result=current
            if self.previous is not None:
                past=self.previous[index]
                if self.mode=='identity':
                    old=past['value'];old_trust=past['trust'];valid=past['valid']
                elif backward_flow is not None and validity is not None:
                    packed=torch.cat([past['value'],past['trust'],past['valid'].float()],1)
                    moved,valid=warp_previous(packed,backward_flow,validity)
                    channels=value.shape[1];old=moved[:,:channels];old_trust=moved[:,channels:channels+1]
                    valid=valid&(moved[:,channels+1:channels+2]>.999)
                else:old=None
                if old is not None:
                    energy=normalized.square().sum(1,keepdim=True)
                    cosine=(normalized*old).sum(1,keepdim=True)/torch.sqrt(energy*old.square().sum(1,keepdim=True)).clamp_min(self.epsilon)
                    if self.mode=='without_current_evidence':
                        # Preserve the actual small historical amplitude; do
                        # not multiply it by the inverse learned trust.
                        ratio=torch.ones_like(confidence);compatible=(cosine>0)|(energy<=self.epsilon**2)
                        maximum=torch.full_like(confidence,self.bound)
                    else:
                        ratio=(confidence/(old_trust+self.epsilon)).clamp(max=1)
                        compatible=(cosine>0)&(energy>self.epsilon**2)&(confidence>self.epsilon)
                        maximum=self.bound*confidence
                    use=valid&compatible
                    mixture=(1-self.strength)*normalized+self.strength*old*ratio
                    mixture=torch.maximum(torch.minimum(mixture,maximum),-maximum)
                    candidate=(value+rms*mixture).to(current.dtype)
                    result=torch.where(use,candidate,current)
            # Store the correction actually applied after output quantization,
            # with private detached buffers that cannot alias caller inputs.
            actual=(result.float()-value)/rms
            support=(confidence>self.epsilon)&(actual.abs().amax(1,keepdim=True)>self.epsilon)
            next_state.append(dict(value=actual.detach().clone(),trust=confidence.detach().clone(),valid=support.detach().clone()))
            output.append(result)
        self.previous=next_state
        return tuple(output)
