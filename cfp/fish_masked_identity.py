"""Genuine pre-attention image-token masking; independent stopped EMA target.

Current unmasked crop may serve supervised identity. Its activations never
enter the masked prediction branch, which replaces patch embeddings before
any backbone attention. This is an owned DINO/I-JEPA-style control, not novel
by mechanism and not official FishMAE source/weights reproduction.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .dinov2_fish_encoder import FineGrainedDinoFishEncoder
from .dinov2_identity import resized_observations


class MaskedPixelFishIdentity(nn.Module):
    def __init__(self,backbone,identities,count=4,decay=.99):
        super().__init__()
        if not 0<=decay<1:raise ValueError('Declared stopped EMA decay required')
        self.student=FineGrainedDinoFishEncoder(backbone,identities,'global',count,2)
        self.teacher=copy.deepcopy(self.student.backbone).requires_grad_(False).eval()
        self.predictor=nn.Sequential(nn.Linear(384,768),nn.GELU(),nn.Linear(768,384))
        self.decay=decay;self.register_buffer('teacher_updates',torch.zeros((),dtype=torch.int64))
        if any(a.data_ptr()==b.data_ptr() for a,b in zip(self.student.backbone.parameters(),self.teacher.parameters())):raise ValueError('Independent target storage required')
    def train(self,mode=True):
        super().train(mode);self.teacher.eval();return self
    def masked_prepared(self,images,mask):
        if images.shape[1:]!=(3,126,252) or mask.shape!=(len(images),162) or mask.dtype!=torch.bool:raise ValueError('Exact current prepared pixels and patch-grid mask required')
        observed=self.student.backbone.forward_features(images,masks=mask)
        return self.predictor(observed['x_norm_patchtokens'])
    def training_observations(self,images,content_masks,mask):
        prepared,fraction=resized_observations(images,content_masks)
        if (mask & ~(fraction.flatten(1)>0)).any():raise ValueError('Targetqueries must have observed image content, not padding-only')
        identity=self.student(images,content_masks)
        prediction=self.masked_prepared(prepared,mask)
        with torch.no_grad():target=self.teacher.forward_features(prepared)['x_norm_patchtokens'].detach()
        return dict(identity=identity,prediction=prediction,target=target,target_mask=mask,patch_fraction=fraction.flatten(1))
    def online(self,images,content_masks):
        return self.student(images,content_masks)
    @torch.no_grad()
    def update_target(self,successful=True):
        if not successful:return
        for target,source in zip(self.teacher.parameters(),self.student.backbone.parameters()):target.lerp_(source.detach(),1-self.decay)
        for target,source in zip(self.teacher.buffers(),self.student.backbone.buffers()):target.copy_(source)
        self.teacher_updates.add_(1)
        if any(p.requires_grad or p.grad is not None for p in self.teacher.parameters()):raise RuntimeError('Target must remain stopped')


def masked_patch_identity_loss(observations):
    selected=observations['target_mask']&(observations['patch_fraction']>0)
    if not selected.any():return observations['prediction'].sum()*0,0
    prediction=F.layer_norm(observations['prediction'].float(),(384,));target=F.layer_norm(observations['target'].detach().float(),(384,))
    return (prediction-target).abs()[selected].mean(),int(selected.sum())
