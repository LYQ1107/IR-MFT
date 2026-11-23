"""Small offset-tolerant auxiliary readout around a frozen RGB detector."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def local_readout(query,key,value,radius=2,temperature=.1):
    if query.shape!=key.shape or value.shape!=key.shape:
        raise ValueError('Matched current-frame feature-grid shapes required')
    batch,channels,height,width=query.shape
    count=(2*radius+1)**2
    q=F.normalize(query.float(),dim=1,eps=1e-6).flatten(2)
    k=F.unfold(F.normalize(key.float(),dim=1,eps=1e-6),2*radius+1,padding=radius)
    k=k.reshape(batch,channels,count,height*width)
    similarity=(q[:,:,None]*k).sum(1)
    valid=F.unfold(torch.ones_like(query[:,:1]),2*radius+1,padding=radius)>0
    logits=(similarity/temperature).masked_fill(~valid,-1e4)
    weight=logits.softmax(1)
    values=F.unfold(value.float(),2*radius+1,padding=radius).reshape(batch,channels,count,height*width)
    result=(weight[:,None]*values).sum(2).reshape(batch,channels,height,width)
    top=similarity.masked_fill(~valid,-2).topk(min(2,count),dim=1).values
    margin=top[:,0]-top[:,1]if count>1 else torch.zeros_like(top[:,0])
    entropy=-(weight*weight.clamp_min(1e-8).log()).sum(1)/max(math.log(count),1.)
    diagnostics=torch.stack([top[:,0],margin,1-entropy],dim=1).reshape(batch,3,height,width)
    return result,diagnostics


class CorrelationLevel(nn.Module):
    def __init__(self,channels,latent=32,radius=2,temperature=.1,risk_gate=True,bound=.5):
        super().__init__()
        self.radius=radius;self.temperature=temperature;self.risk_gate=risk_gate;self.bound=bound
        self.project=nn.Conv2d(channels,latent,1,bias=False)
        self.value=nn.Conv2d(channels,latent,1,bias=False)
        self.output=nn.Conv2d(latent,channels,1,bias=True)
        self.gate=nn.Sequential(nn.Conv2d(5,16,1),nn.SiLU(),nn.Conv2d(16,1,1))
        nn.init.orthogonal_(self.project.weight.flatten(1))
        self.value.weight.data.copy_(self.project.weight.data)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,-3.)

    def forward(self,rgb,auxiliary,teacher_score=None,availability=None):
        # An absent cue cannot trigger an auxiliary array read or alter RGB.
        if availability is not None:
            if isinstance(availability,bool):
                if not availability:return rgb
            elif not bool(availability.any()):return rgb
        if auxiliary is None:raise ValueError('Available auxiliary input is missing')
        if rgb.shape!=auxiliary.shape:raise ValueError('Current RGB/auxiliary grids differ')
        original_type=rgb.dtype
        rgb_value=rgb.float();auxiliary=auxiliary.float()
        query=self.project(rgb_value);key=self.project(auxiliary);value=self.value(auxiliary)
        aligned,agreement=local_readout(query,key,value,self.radius,self.temperature)
        residual=self.output(aligned-query)
        if self.risk_gate:
            rgb_rms=rgb_value.square().mean(1,keepdim=True).add(1e-6).sqrt()
            aux_rms=auxiliary.square().mean(1,keepdim=True).add(1e-6).sqrt()
            if teacher_score is None:teacher_score=torch.zeros_like(rgb_rms)
            teacher_score=teacher_score.detach().float().clamp(0,1)
            features=torch.cat([agreement,teacher_score,(aux_rms/rgb_rms).clamp_min(1e-6).log().clamp(-5,5)],1)
            trust=self.gate(features).sigmoid()*(1-teacher_score)*(teacher_score<.6)
            residual=self.bound*rgb_rms*residual.tanh()*trust
        if availability is not None and not isinstance(availability,bool):
            residual=residual*availability.to(residual).reshape(-1,1,1,1)
        return (rgb_value+residual).to(original_type)


class CorrelationRepair(nn.Module):
    def __init__(self,channels=(320,640,1280),latent=32,radius=2,temperature=.1,risk_gate=True,bound=.5):
        super().__init__()
        self.levels=nn.ModuleList([CorrelationLevel(c,latent,radius,temperature,risk_gate,bound)for c in channels])

    def forward(self,rgb,auxiliary,teacher_scores=None,availability=None):
        if len(rgb)!=len(self.levels):raise ValueError('Three original FPN levels required')
        scores=[None]*len(rgb)if teacher_scores is None else teacher_scores
        return tuple(layer(value,None if auxiliary is None else auxiliary[index],scores[index],availability)
                     for index,(layer,value)in enumerate(zip(self.levels,rgb)))
