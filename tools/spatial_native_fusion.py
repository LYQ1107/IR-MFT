"""Current-image spatial selection for an unvalidated November-IR candidate.

Kept outside the live, source-pinned model directory. This reuses the frozen
native cue encoder and RGB protection, not historical causal-cache weights.
No training/evaluation launch, GT access, checkpoint selection or memory.
"""
import math

import torch
from torch import nn

from yolox.models.native_pyramid_fusion import NativePyramidFusion, quality_features


def channel_rms(value):
    return value / value.square().mean(1, keepdim=True).add(1e-6).sqrt()


class SpatialComparison(nn.Module):
    def __init__(self, rgb_channels, auxiliary_channels, hidden=32):
        super().__init__()
        self.rgb_projection = nn.Conv2d(rgb_channels, hidden, 1, bias=False)
        self.aux_projection = nn.Conv2d(auxiliary_channels, hidden, 1, bias=False)
        self.spatial = nn.Sequential(nn.Conv2d(4*hidden, hidden, 1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.SiLU(),
            nn.Conv2d(hidden, 1, 1))
        self.redundancy = nn.Sequential(nn.Conv2d(2*hidden, hidden//2, 1), nn.SiLU(),
            nn.Conv2d(hidden//2, 1, 1))
        nn.init.zeros_(self.spatial[-1].weight)
        nn.init.zeros_(self.spatial[-1].bias)
        nn.init.zeros_(self.redundancy[-1].weight)
        nn.init.constant_(self.redundancy[-1].bias, -6.)

    def forward(self, rgb, auxiliary, spatial=True, redundancy=False):
        r = channel_rms(self.rgb_projection(rgb.detach().float()))
        a = channel_rms(self.aux_projection(auxiliary.float()))
        difference, product = (r-a).abs(), r*a
        delta = self.spatial(torch.cat((r, a, difference, product), 1)) if spatial else torch.zeros_like(r[:, :1])
        rho = self.redundancy(torch.cat((difference, product), 1)).sigmoid() if redundancy else torch.zeros_like(delta)
        return delta, rho


class SpatialNativeFusion(NativePyramidFusion):
    """Same-capacity spatial/quality/redundancy controls over a native encoder.

    The final residual projection starts at zero, preserving complete RGB head
    outputs. A zero spatial delta also preserves a loaded quality-only parent
    when redundancy is disabled. Redundancy starts near, not exactly, disabled.
    """
    def __init__(self, channels=(320, 640, 1280), hidden=32, spectral=True,
                 use_quality=True, use_spatial=True, use_redundancy=False,
                 protect_rgb=True, bound=.5):
        if (len(channels) != 3 or any(not isinstance(c, int) or c < 1 for c in channels)
                or not isinstance(hidden, int) or hidden < 2 or hidden % 2
                or not math.isfinite(bound) or bound <= 0):
            raise ValueError('Three positive channel counts, even context width and positive finite bound required')
        super().__init__(channels=channels, mode='quality', spectral=spectral,
            use_quality=use_quality, protect_rgb=protect_rgb, use_memory=False,
            use_motion=False, bound=bound)
        self.channels, self.use_spatial, self.use_redundancy = tuple(channels), use_spatial, use_redundancy
        self.comparison = nn.ModuleList([SpatialComparison(c, a, hidden)
            for c, a in zip(channels, (64, 96, 128))])

    def forward(self, rgb, cue, scores, valid=None, available=True, state=None,
                sequence=None, frame=None, flow_to_previous=None):
        if state is not None or flow_to_previous is not None:
            raise ValueError('This single-frame spatial candidate accepts no historical state or flow')
        if len(rgb) != 3 or len(scores) != 3:
            raise ValueError('All three current RGB levels and confidence maps required')
        batch = len(rgb[0])
        availability = torch.as_tensor(available, device=rgb[0].device, dtype=torch.bool)
        if availability.ndim == 0:
            availability = availability.expand(batch)
        if availability.shape != (batch,):
            raise ValueError('Explicit per-observation availability required')
        if not bool(availability.any()):
            return tuple(rgb), None, dict(missing_exact_RGB=True, state_cleared=True)
        if cue is None or cue.ndim != 4 or cue.shape[:2] != (batch, 3):
            raise ValueError('Available observations require current normalized three-channel cue pixels')
        if valid is None:
            valid = torch.ones_like(cue[:, :1], dtype=torch.bool)
        if valid.shape != cue[:, :1].shape or valid.dtype != torch.bool:
            raise ValueError('Explicit valid image rectangle must be boolean Bx1xHxW')
        latent, masks, image_stats = self.encoder(cue, valid)
        outputs, diagnostics = [], []
        for index, (base, current, score, mask, comparison) in enumerate(zip(rgb, latent, scores, masks, self.comparison)):
            if (base.ndim != 4 or base.shape[:2] != (batch, self.channels[index])
                    or base.shape[-2:] != current.shape[-2:] or score.shape != base[:, :1].shape
                    or not bool(torch.isfinite(base).all()) or not bool(torch.isfinite(score).all())):
                raise ValueError('Finite matching current RGB/cue grids and teacher confidence required')
            statistics = quality_features(base, current, score, mask, image_stats)
            quality_logit = self.quality[index](statistics) if self.use_quality else statistics.new_zeros((batch, 1))
            if self.use_spatial or self.use_redundancy:
                delta, rho = comparison(base, current, self.use_spatial, self.use_redundancy)
            else:
                delta, rho = torch.zeros_like(current[:, :1]), torch.zeros_like(current[:, :1])
            gate = (quality_logit[:, :, None, None] + delta).sigmoid() * (1-rho)
            projection = self.output[index](current)
            probability = score.detach().float().clamp(0, 1)
            trust = (1-probability)*(probability < .6) if self.protect_rgb else torch.ones_like(probability)
            scale = base.float().square().mean(1, keepdim=True).add(1e-6).sqrt()
            correction = self.bound*scale*projection.tanh()*trust*gate*mask
            value = (base.float()+correction).to(base.dtype)
            outputs.append(torch.where(availability[:, None, None, None], value, base))
            diagnostics.append(dict(gate=gate, frame_quality=quality_logit.sigmoid(),
                spatial_delta=delta, redundancy=rho, valid=mask))
        return tuple(outputs), None, dict(levels=diagnostics, missing_exact_RGB=False)
