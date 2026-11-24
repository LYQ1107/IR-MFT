"""Foreground and fixed-task-utility supervision candidate, not a novelty claim.

The inference interface has no labels. Training labels belong to the separate
objective. Kept outside the registered model directory and existing queues.
"""
import math

import torch
from torch import nn

from tools.spatial_native_fusion import channel_rms
from yolox.models.native_pyramid_fusion import NativePyramidFusion, quality_features


class TaskContext(nn.Module):
    def __init__(self, rgb_channels, auxiliary_channels, hidden):
        super().__init__()
        self.rgb_projection=nn.Conv2d(rgb_channels,hidden,1,bias=False)
        self.aux_projection=nn.Conv2d(auxiliary_channels,hidden,1,bias=False)
        self.context=nn.Sequential(nn.Conv2d(4*hidden+1,hidden,1),nn.SiLU(),
            nn.Conv2d(hidden,hidden,3,padding=1,groups=hidden),nn.SiLU())
        self.foreground=nn.Conv2d(hidden,1,1)
        self.utility=nn.Conv2d(hidden,1,1)
        for layer in [self.foreground,self.utility]:
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self,rgb,auxiliary,score):
        r=channel_rms(self.rgb_projection(rgb.detach().float()))
        a=channel_rms(self.aux_projection(auxiliary.float()))
        context=self.context(torch.cat((r,a,(r-a).abs(),r*a,score.detach().float()),1))
        return self.foreground(context),self.utility(context)


class TaskUtilityFusion(NativePyramidFusion):
    """Bounded feature correction and two explicitly uncalibrated task logits.

    All removals instantiate the same tensors. Zero task logits preserve a
    loaded quality-only parent exactly. A missing observation returns RGB and
    no label/proposal state. Optional detached proposals are training-only
    diagnostics, never inputs to inference or a permission to start a GPU.
    """
    def __init__(self,channels=(320,640,1280),hidden=32,use_foreground=True,use_utility=True,
                 spectral=True,use_quality=True,protect_rgb=True,bound=.5):
        if len(channels)!=3 or any(not isinstance(c,int) or c<1 for c in channels) or hidden<1 or not math.isfinite(bound) or bound<=0:
            raise ValueError('Positive channels, context width and finite residual bound required')
        super().__init__(channels=channels,mode='quality',spectral=spectral,use_quality=use_quality,
            protect_rgb=protect_rgb,use_memory=False,use_motion=False,bound=bound)
        self.channels=tuple(channels);self.use_foreground=use_foreground;self.use_utility=use_utility
        self.task=nn.ModuleList([TaskContext(c,a,hidden) for c,a in zip(channels,(64,96,128))])

    def forward(self,rgb,cue,scores,valid=None,available=True,state=None,sequence=None,frame=None,
                flow_to_previous=None,return_proposals=False):
        if state is not None or flow_to_previous is not None:
            raise ValueError('Single-frame task selector accepts no historical state or flow')
        if len(rgb)!=3 or len(scores)!=3:
            raise ValueError('All three matching RGB feature and teacher-score grids required')
        batch=len(rgb[0]);availability=torch.as_tensor(available,device=rgb[0].device,dtype=torch.bool)
        if availability.ndim==0:availability=availability.expand(batch)
        if availability.shape!=(batch,):raise ValueError('Availability is per observation')
        if not bool(availability.any()):
            return tuple(rgb),None,dict(missing_exact_RGB=True,state_cleared=True)
        if cue is None or cue.ndim!=4 or cue.shape[:2]!=(batch,3) or not bool(torch.isfinite(cue).all()):
            raise ValueError('Available inputs require finite normalized current cue pixels')
        if valid is None:valid=torch.ones_like(cue[:,:1],dtype=torch.bool)
        if valid.shape!=cue[:,:1].shape or valid.dtype!=torch.bool:
            raise ValueError('Boolean valid image rectangle required')
        latent,masks,image_stats=self.encoder(cue,valid)
        outputs,proposals,levels=[],[],[]
        for index,(base,current,score,mask,task) in enumerate(zip(rgb,latent,scores,masks,self.task)):
            if base.shape[:2]!=(batch,self.channels[index]) or base.shape[-2:]!=current.shape[-2:] or score.shape!=base[:,:1].shape:
                raise ValueError('Current RGB, cue and score geometry differs')
            statistics=quality_features(base,current,score,mask,image_stats)
            quality=self.quality[index](statistics) if self.use_quality else statistics.new_zeros((batch,1))
            foreground,utility=task(base,current,score)
            logit=quality[:,:,None,None].expand_as(foreground)
            if self.use_foreground:logit=logit+foreground
            if self.use_utility:logit=logit+utility
            gate=logit.sigmoid()
            probability=score.detach().float().clamp(0,1)
            trust=(1-probability)*(probability<.6) if self.protect_rgb else torch.ones_like(probability)
            scale=base.float().square().mean(1,keepdim=True).add(1e-6).sqrt()
            residual=self.bound*scale*self.output[index](current).tanh()*trust*mask
            active=availability[:,None,None,None]
            outputs.append(torch.where(active,(base.float()+residual*gate).to(base.dtype),base))
            if return_proposals:
                proposals.append(torch.where(active,(base.float()+.5*residual).to(base.dtype),base).detach())
            levels.append(dict(gate=gate,foreground_logits=foreground,utility_logits=utility,
                valid=mask & active,frame_quality=quality.sigmoid()))
        diagnostics=dict(levels=levels,missing_exact_RGB=False)
        if return_proposals:diagnostics['detached_proposal_features']=tuple(proposals)
        return tuple(outputs),None,diagnostics
