"""Current-crop fish identity learning with real local spatial evidence.

Geometric regions and observed padding/augmentation masks are not fish anatomy
or ground-truth occlusion. The capacity control reads CLS content only.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .dinov2_identity import resized_observations


class PatchIdentityHead(nn.Module):
    def __init__(self, identities, mode='patch_attention', count=4, dimension=384, heads=8):
        super().__init__()
        if mode not in ('global', 'simple', 'global_capacity', 'patch_attention') or count not in (4, 8):
            raise ValueError('Registered identity mode and K4/K8 required')
        if identities < 1 or dimension % heads:
            raise ValueError('Positive identity vocabulary and compatible attention dimensions required')
        self.mode, self.count, self.dimension, self.heads = mode, count, dimension, heads
        self.global_projection = nn.Linear(dimension, dimension, bias=False)
        self.local_projection = nn.Linear(dimension, dimension, bias=False)
        nn.init.eye_(self.global_projection.weight)
        nn.init.eye_(self.local_projection.weight)
        self.queries = nn.Parameter(torch.randn(count, dimension)*.02)
        self.position = nn.Parameter(torch.randn(162, dimension)*.02)
        self.attention = nn.MultiheadAttention(dimension, heads, batch_first=True, dropout=0.)
        self.local_norm = nn.LayerNorm(dimension)
        self.local_ffn = nn.Sequential(nn.Linear(dimension, dimension*2), nn.GELU(), nn.Linear(dimension*2, dimension))
        self.output_norm = nn.LayerNorm(dimension)
        self.reliability = nn.Sequential(nn.Linear(dimension+1, 64), nn.GELU(), nn.Linear(64, 1))
        self.fusion_logit = nn.Parameter(torch.tensor(math.log(.1/.9)))
        self.classifier = nn.Linear(dimension, identities, bias=False)
        region = torch.zeros(count, 9, 18, dtype=torch.bool)
        for k in range(count):
            region[k, :, math.floor(k*18/count):math.ceil((k+1)*18/count)] = True
        self.register_buffer('region_membership', region.flatten(1), persistent=False)
        # Report actual trainable capacity rather than unused instantiated heads.
        if mode == 'global':
            for name, parameter in self.named_parameters():
                if not name.startswith(('global_projection.', 'classifier.')):
                    parameter.requires_grad_(False)
        elif mode == 'simple':
            for name, parameter in self.named_parameters():
                if not name.startswith(('global_projection.', 'local_projection.', 'classifier.')) and name != 'fusion_logit':
                    parameter.requires_grad_(False)

    def forward(self, cls, patches, patch_visibility):
        b = len(cls)
        if (cls.shape != (b, self.dimension) or patches.shape != (b, 162, self.dimension)
                or patch_visibility.shape != (b, 1, 9, 18)
                or not all(torch.isfinite(x).all() for x in (cls, patches, patch_visibility))
                or (patch_visibility < 0).any() or (patch_visibility > 1).any()):
            raise ValueError('Aligned finite current CLS/patches/observable content fractions required')
        fractions = patch_visibility.flatten(1).float()
        member = self.region_membership.float()
        visible = (fractions @ member.T)/member.sum(1)[None]
        available = visible > 0
        crop_available = fractions.sum(1) > 0
        global_feature = self.global_projection(cls)
        local = cls.new_zeros(b, self.count, self.dimension)
        weights = fractions.new_zeros(b, self.count)
        if self.mode == 'simple':
            projected = self.local_projection(patches)
            mass = fractions @ member.T
            local = torch.einsum('kn,bnd->bkd', member, projected.float()*fractions[..., None])/mass[..., None].clamp_min(1e-6)
            local = local*available[..., None]
            weights = available.float()/available.sum(1, keepdim=True).clamp_min(1)
        elif self.mode in ('patch_attention', 'global_capacity'):
            # Repeated CLS + the same static positions/region masks gives equal
            # architecture and compute without access to spatial image content.
            raw = cls[:, None].expand(-1, 162, -1) if self.mode == 'global_capacity' else patches
            keys = self.local_projection(raw)+self.position[None]
            allowed = self.region_membership[None] & (fractions[:, None] > 0)
            bias = fractions[:, None].clamp_min(1e-6).log().expand(-1, self.count, -1).clone()
            bias = bias.masked_fill(~allowed, -torch.inf)
            empty = ~allowed.any(-1)
            bias[..., 0] = torch.where(empty, torch.zeros_like(bias[..., 0]), bias[..., 0])
            bias = bias[:, None].expand(-1, self.heads, -1, -1).reshape(b*self.heads, self.count, 162).to(keys.dtype)
            queries = self.queries[None].expand(b, -1, -1)
            local = self.attention(queries, keys, keys, attn_mask=bias, need_weights=False)[0]
            local = self.local_norm(local+queries)
            local = self.output_norm(local+self.local_ffn(local))*available[..., None]
            score = self.reliability(torch.cat([F.normalize(local.float(), dim=-1), visible[..., None]], -1)).squeeze(-1)
            score = score+visible.clamp_min(1e-6).log()
            safe = available.clone()
            safe[~available.any(1), 0] = True
            weights = F.softmax(score.masked_fill(~safe, -torch.inf), -1)*crop_available[:, None]
        feature = global_feature
        if self.mode != 'global':
            feature = feature+self.fusion_logit.sigmoid()*(local.float()*weights[..., None]).sum(1)
        feature = feature.float()*crop_available[:, None]
        return dict(embedding=F.normalize(feature, dim=-1), logits=self.classifier(feature),
            global_feature=F.normalize(global_feature.float(), dim=-1)*crop_available[:, None],
            raw_global=F.normalize(cls.float(), dim=-1)*crop_available[:, None],
            local_tokens=F.normalize(local.float(), dim=-1), local_weights=weights,
            local_valid=available if self.mode != 'global' else torch.zeros_like(available),
            content_fraction=visible, crop_valid=crop_available)


class FineGrainedDinoFishEncoder(nn.Module):
    def __init__(self, backbone, identities, mode='patch_attention', count=4, train_blocks=2):
        super().__init__()
        if len(backbone.blocks) != 12 or not 0 <= train_blocks <= 12:
            raise ValueError('Exact small twelve-block backbone and declared trainable suffix required')
        self.backbone = backbone.requires_grad_(False)
        if train_blocks:
            for block in self.backbone.blocks[-train_blocks:]:
                block.requires_grad_(True)
            self.backbone.norm.requires_grad_(True)
        self.head = PatchIdentityHead(identities, mode, count)

    def forward(self, images, content_masks):
        images, visibility = resized_observations(images, content_masks)
        observed = self.backbone.forward_features(images)
        return self.head(observed['x_norm_clstoken'], observed['x_norm_patchtokens'], visibility)
