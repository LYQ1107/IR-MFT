"""ImageNet-initialized current-crop visual/local identity fitting.

Separate from stopped-feature adapters: image features retain backbone gradient.
Only current crops and observable letterbox masks are model inputs.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

from .local_identity_encoder import LocalIdentityAdapter, resnet_spatial, strip_features


class TrainableLocalFishReID(nn.Module):
    def __init__(self, identities, dimension=256, mode='global', count=8):
        super().__init__()
        self.backbone = resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        self.head = LocalIdentityAdapter(identities, dimension, count, mode)
        self.count = count

    def initialize_imagenet(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        original = resnet50(weights=None)
        original.load_state_dict(state, strict=True)
        original.fc = nn.Identity()
        self.backbone.load_state_dict(original.state_dict(), strict=True)

    def forward_spatial(self, spatial, visibility):
        raw = F.adaptive_avg_pool2d(spatial, 1).flatten(1)
        local_raw = strip_features(spatial, self.count)
        head = self.head
        global_feature = head.global_projection(raw.float())
        weights = torch.zeros_like(visibility)
        feature = global_feature
        if head.mode == 'dual_global':
            feature = feature + head.local_projection(raw.float())
        elif head.mode != 'global':
            local = head.local_projection(local_raw.float())
            available = visibility > 0
            safe = available.clone()
            safe[~available.any(1), 0] = True
            if head.mode == 'learned':
                score = head.local_weight(torch.cat([F.normalize(local.float(), dim=-1), visibility[..., None]], dim=-1)).squeeze(-1)
                score = score + torch.log(visibility.clamp_min(1e-6))
                weights = F.softmax(score.masked_fill(~safe, -torch.inf), dim=-1) * available.any(1)[:, None]
            else:
                weights = available.float() / available.sum(1, keepdim=True).clamp_min(1)
            feature = feature + (weights[..., None] * local).sum(1)
        feature = head.neck(feature)
        return dict(embedding=F.normalize(feature.float(), dim=-1), logits=head.classifier(feature), weights=weights)

    def forward(self, images, visibility):
        if images.ndim != 4 or visibility.shape != (len(images), self.count) or not torch.isfinite(images).all():
            raise ValueError('Finite current image crops and aligned letterbox visibility required')
        return self.forward_spatial(resnet_spatial(self.backbone, images), visibility)
