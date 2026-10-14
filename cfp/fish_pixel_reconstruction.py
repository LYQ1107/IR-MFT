"""Owned masked autoencoding control: reconstruct observable current RGB pixels.

Uses licensed DINO's full token encoder with pre-attention replacement, not
official MAE's visible-token-only encoder/decoder or official FishMAE weights.
There is no EMA teacher. Same-frame identity supervision is a separate branch.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .dinov2_fish_encoder import FineGrainedDinoFishEncoder
from .dinov2_identity import resized_observations


def patchify(images):
    if images.ndim != 4 or images.shape[1:] != (3, 126, 252):
        raise ValueError('Prepared current RGB126x252 patches required')
    return images.reshape(-1, 3, 9, 14, 18, 14).permute(0, 2, 4, 3, 5, 1).reshape(-1, 162, 588)


class PixelReconstructionFishIdentity(nn.Module):
    def __init__(self, backbone, identities, count=4):
        super().__init__()
        self.student = FineGrainedDinoFishEncoder(backbone, identities, 'global', count, 2)
        self.predictor = nn.Sequential(nn.Linear(384, 768), nn.GELU(), nn.Linear(768, 588))
        self.register_buffer('successful_updates', torch.zeros((), dtype=torch.int64))

    def masked_prepared(self, prepared, queries):
        return self.predictor(self.student.backbone.forward_features(prepared, masks=queries)['x_norm_patchtokens'])

    def training_observations(self, images, content, queries):
        prepared, fraction = resized_observations(images, content)
        if queries.shape != (len(images), 162) or queries.dtype != torch.bool or (queries & ~(fraction.flatten(1) > 0)).any():
            raise ValueError('Only actual observable patch pixels may be reconstruction targets')
        pixel_content = F.interpolate(content.float(), size=(126, 252), mode='bilinear', align_corners=False).clamp(0, 1)
        weights = patchify(pixel_content.expand(-1, 3, -1, -1)).detach()
        target = patchify(prepared).detach()
        mass = weights.sum(-1, keepdim=True).clamp_min(1e-6)
        mean = (target*weights).sum(-1, keepdim=True)/mass
        variance = ((target-mean).square()*weights).sum(-1, keepdim=True)/mass
        target = (target-mean)/(variance+1e-6).sqrt()
        return dict(identity=self.student(images, content), prediction=self.masked_prepared(prepared, queries),
                    target=target, pixel_weights=weights, target_mask=queries, patch_fraction=fraction.flatten(1))

    def online(self, images, content):
        return self.student(images, content)

    @torch.no_grad()
    def record_successful_update(self, successful=True):
        self.successful_updates.add_(int(successful))


def pixel_reconstruction_loss(observations):
    selected = observations['target_mask'] & (observations['patch_fraction'] > 0)
    weight = observations['pixel_weights']*selected[:, :, None]
    error = (observations['prediction'].float()-observations['target'].float()).square()
    return (error*weight).sum()/weight.sum().clamp_min(1e-6), int(selected.sum())


def frozen_reconstruction_base(model):
    names = {n for n, p in model.student.backbone.named_parameters() if not p.requires_grad}
    state = model.state_dict()
    return {'student.backbone.'+n: state['student.backbone.'+n].detach().cpu().clone() for n in sorted(names)}
