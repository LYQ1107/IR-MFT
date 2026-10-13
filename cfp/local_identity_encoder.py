"""Current-crop feature stages and simple local spatial controls.

Spatial strips are geometric regions, not annotated fish anatomy. Padding
visibility measures real image content, not occlusion or correct fish identity.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def resnet_spatial(backbone, images):
    x = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(images))))
    for layer in (backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4):
        x = layer(x)
    return x


def strip_features(spatial, count):
    if spatial.ndim != 4 or count not in (4, 8) or spatial.shape[-1] < count:
        raise ValueError('Finite current spatial map and declared K4/8 strips required')
    if not torch.isfinite(spatial).all():
        raise ValueError('Nonfinite spatial identity observation')
    return F.adaptive_avg_pool2d(spatial, (1, count)).squeeze(2).transpose(1, 2)


def letterbox_visibility(image_shape, xywh, size=(128, 256), margin=.05):
    """Exact crop_to_tensor rectangle bookkeeping; no image or GT content test."""
    x, y, w, h = map(float, xywh)
    if not np.isfinite([x, y, w, h]).all() or min(w, h) <= 0:
        raise ValueError('Finite positive crop box required')
    x1, y1 = max(0, int(math.floor(x - margin * w))), max(0, int(math.floor(y - margin * h)))
    x2, y2 = min(image_shape[1], int(math.ceil(x + (1 + margin) * w))), min(image_shape[0], int(math.ceil(y + (1 + margin) * h)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Crop lies outside image')
    oh, ow = size
    ratio = min(oh / (y2 - y1), ow / (x2 - x1))
    rh, rw = max(1, round((y2 - y1) * ratio)), max(1, round((x2 - x1) * ratio))
    dy, dx = (oh - rh) // 2, (ow - rw) // 2
    mask = torch.zeros(1, oh, ow)
    mask[:, dy:dy + rh, dx:dx + rw] = 1
    return mask


def frozen_fish_stages(model, spatial):
    """The inherited fish projection/BN is a diagnostic, exposed reference."""
    raw = F.adaptive_avg_pool2d(spatial, 1).flatten(1)
    pre = model.projection(raw)
    post = model.neck(pre)
    result = dict(raw=F.normalize(raw.float(), dim=-1), pre_neck=F.normalize(pre.float(), dim=-1),
                  post_neck=F.normalize(post.float(), dim=-1))
    for count in (4, 8):
        tokens = strip_features(spatial, count)
        projected = model.projection(tokens)
        projected = model.neck(projected.flatten(0, 1)).reshape_as(projected)
        result['local' + str(count)] = F.normalize(projected.float(), dim=-1)
    return result


class LocalIdentityAdapter(nn.Module):
    """Supervised global/local head on stopped ImageNet spatial observations.

    The separately registered global, simple-local and learned-local controls
    consume the same current crop. Learned local weights are identity-trained
    pooling, not ground-truth fish visibility or a JEPA mechanism.
    """
    def __init__(self, identities, dimension=256, local_count=8, mode='learned'):
        super().__init__()
        if mode not in ('global', 'dual_global', 'simple', 'learned') or local_count not in (4, 8):
            raise ValueError('Registered global/simple/learned K4/8 adapter required')
        self.mode, self.local_count, self.dimension = mode, local_count, dimension
        self.global_projection = nn.Linear(2048, dimension, bias=False)
        self.local_projection = nn.Linear(2048, dimension, bias=False)
        self.local_weight = nn.Sequential(nn.Linear(dimension + 1, 64), nn.GELU(), nn.Linear(64, 1))
        self.neck = nn.LayerNorm(dimension)
        self.classifier = nn.Linear(dimension, identities, bias=False)

    def forward(self, global_raw, local_raw, visibility):
        b = len(global_raw)
        if (global_raw.shape != (b, 2048) or local_raw.shape != (b, self.local_count, 2048)
                or visibility.shape != (b, self.local_count)
                or not all(torch.isfinite(x).all() for x in (global_raw, local_raw, visibility))
                or (visibility < 0).any() or (visibility > 1).any()):
            raise ValueError('Finite aligned current crop features/letterbox visibility required')
        global_feature = self.global_projection(global_raw.detach().float())
        weights = torch.zeros_like(visibility)
        feature = global_feature
        if self.mode == 'dual_global':
            # Same two2048->D projections as simple local aggregation, but
            # neither can read spatial strips: capacity/optimizer control.
            feature = global_feature + self.local_projection(global_raw.detach().float())
        elif self.mode != 'global':
            local = self.local_projection(local_raw.detach().float())
            available = visibility > 0
            safe = available.clone()
            safe[~available.any(1), 0] = True
            if self.mode == 'learned':
                logits = self.local_weight(torch.cat([F.normalize(local, dim=-1), visibility[..., None]], dim=-1)).squeeze(-1)
                logits = logits + torch.log(visibility.clamp_min(1e-6))
                weights = F.softmax(logits.masked_fill(~safe, -torch.inf), dim=-1) * available.any(1)[:, None]
            else:
                weights = available.float() / available.sum(1, keepdim=True).clamp_min(1)
            feature = global_feature + (weights[..., None] * local).sum(1)
        neck = self.neck(feature)
        return dict(embedding=F.normalize(neck.float(), dim=-1), logits=self.classifier(neck), weights=weights)
