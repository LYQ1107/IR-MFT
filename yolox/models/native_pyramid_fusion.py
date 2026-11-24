"""Native cue learning with quality gating and motion-compensated confidence.

Center-surround contrast, residual gates, flow warping and scalar recursive
confidence updates are established primitives. These experimental combinations
are not claims of originality, calibrated sensor uncertainty or tracking gains.
The RGB encoder/head stay outside this module. Missing cue returns RGB exactly.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def contrast_planes(image, valid, enabled=True):
    """Nine channels; all controls retain the same encoder input dimensions."""
    if image.ndim != 4 or image.shape[1] != 3 or valid.shape != image[:, :1].shape:
        raise ValueError('Require normalized three-channel cue and its valid image rectangle')
    mask = valid.to(image)
    count = F.avg_pool2d(mask, 9, 1, 4, count_include_pad=False).clamp_min(1e-6)
    mean = F.avg_pool2d(image * mask, 9, 1, 4, count_include_pad=False) / count
    second = F.avg_pool2d(image.square() * mask, 9, 1, 4, count_include_pad=False) / count
    scale = (second - mean.square()).clamp_min(1e-4).sqrt()
    high = ((image - mean) / scale.clamp_min(.05)).clamp(-5, 5) * mask
    count = (3 * mask.sum((1, 2, 3))).clamp_min(1)
    stats = torch.stack(((image * mask).sum((1, 2, 3)) / count,
        ((image.square() * mask).sum((1, 2, 3)) / count).sqrt().log1p(),
        (high.abs().sum((1, 2, 3)) / count).log1p()), 1)
    if not enabled:
        high, scale = torch.zeros_like(high), torch.ones_like(scale)
    return torch.cat((image * mask, high, scale * mask), 1), stats


def block(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
                         nn.GroupNorm(8, out_channels), nn.SiLU())


class NativeCuePyramid(nn.Module):
    def __init__(self, spectral=True):
        super().__init__()
        self.spectral = spectral
        widths = [32, 48, 64, 96, 128]
        self.blocks = nn.ModuleList([block(c, d) for c, d in zip([9] + widths[:-1], widths)])

    def forward(self, image, valid):
        value, statistics = contrast_planes(image.float(), valid, self.spectral)
        levels, masks = [], []
        for index, layer in enumerate(self.blocks):
            value = layer(value)
            if index >= 2:
                mask = F.adaptive_avg_pool2d(valid.float(), value.shape[-2:]) >= .5
                levels.append(value * mask)
                masks.append(mask)
        return tuple(levels), tuple(masks), statistics


def warp_previous(value, flow_to_previous):
    """Current->previous pixel displacement; outside pixels carry no support."""
    if flow_to_previous.ndim != 4 or flow_to_previous.shape[:2] != (len(value), 2):
        raise ValueError('Require [B,2,H,W] current-to-previous flow')
    if not torch.isfinite(flow_to_previous).all():
        raise ValueError('Flow must be finite')
    height, width = value.shape[-2:]
    source_h, source_w = flow_to_previous.shape[-2:]
    flow = F.interpolate(flow_to_previous.to(device=value.device, dtype=torch.float32),
                         (height, width), mode='bilinear', align_corners=False)
    dx = flow[:, 0] * width / source_w
    dy = flow[:, 1] * height / source_h
    yy, xx = torch.meshgrid(torch.arange(height, device=value.device),
                            torch.arange(width, device=value.device), indexing='ij')
    grid = torch.stack((2 * (xx[None] + dx + .5) / width - 1,
                        2 * (yy[None] + dy + .5) / height - 1), -1)
    warped = F.grid_sample(value.float(), grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    support = F.grid_sample(torch.ones_like(value[:, :1]).float(), grid,
                           mode='bilinear', padding_mode='zeros', align_corners=False) > .999
    return warped, support


def quality_features(rgb, native, score, valid, image_stats):
    weight = score.detach().float().clamp(0, 1) * valid
    denominator = weight.sum((1, 2, 3)).clamp_min(1e-6)
    def moments(value):
        mean = (value.float().mean(1, keepdim=True) * weight).sum((1, 2, 3)) / denominator
        rms = ((value.float().square().mean(1, keepdim=True) * weight).sum((1, 2, 3)) /
               denominator).clamp_min(1e-8).sqrt()
        return mean / rms, rms.log1p()
    r_mean, r_rms = moments(rgb)
    a_mean, a_rms = moments(native)
    confidence = (score.detach().float() * weight).sum((1, 2, 3)) / denominator
    return torch.cat((torch.stack((r_mean, r_rms, a_mean, a_rms, confidence), 1), image_stats), 1).detach()


def quality_mlp():
    layer = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 1))
    nn.init.zeros_(layer[-1].weight)
    nn.init.zeros_(layer[-1].bias)
    return layer


@dataclass
class NativeTemporalState:
    sequence: str
    frame: int
    latent: tuple
    covariance: tuple
    valid: tuple


class NativePyramidFusion(nn.Module):
    """Small learned cue pyramid; mode is quality or causal confidence filtering.

    Temporal state is caller-owned and must be reset across sequences/gaps.
    A missing cue clears state and emits exact RGB; it cannot reuse stale IR.
    Feature protection is not a guarantee that neighboring head outputs stay
    unchanged: detector-space preservation must be measured separately.
    """
    def __init__(self, channels=(320, 640, 1280), mode='quality', spectral=True,
                 use_quality=True, protect_rgb=True, use_memory=True, use_motion=True, bound=.5):
        super().__init__()
        if mode not in ['quality', 'temporal'] or bound <= 0:
            raise ValueError('Registered mode and positive residual bound required')
        self.mode, self.bound = mode, bound
        self.use_quality, self.protect_rgb = use_quality, protect_rgb
        self.use_memory, self.use_motion = use_memory, use_motion
        self.encoder = NativeCuePyramid(spectral)
        self.output = nn.ModuleList([nn.Conv2d(c, d, 1) for c, d in zip((64, 96, 128), channels)])
        self.quality = nn.ModuleList([quality_mlp() for _ in channels])
        self.noise = nn.ModuleList([quality_mlp() for _ in channels]) if mode == 'temporal' else None
        self.process_noise = nn.Parameter(torch.full((3,), -4.)) if mode == 'temporal' else None
        for layer in self.output:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, rgb, cue, scores, valid=None, available=True, state=None,
                sequence=None, frame=None, flow_to_previous=None):
        if len(rgb) != 3 or len(scores) != 3:
            raise ValueError('All original RGB pyramid levels and confidence maps required')
        if self.mode == 'quality' and state is not None:
            raise ValueError('Quality mode has no temporal state')
        batch = len(rgb[0])
        availability = torch.as_tensor(available, device=rgb[0].device, dtype=torch.bool)
        if availability.ndim == 0:
            availability = availability.expand(batch)
        if availability.shape != (batch,):
            raise ValueError('Availability is per observation')
        if self.mode == 'temporal':
            if batch != 1 or not sequence or not isinstance(frame, int) or frame < 1:
                raise ValueError('Temporal mode requires one ordered frame with sequence identity')
            if state is not None and (state.sequence != sequence or state.frame + 1 != frame):
                raise ValueError('Reset temporal state across a sequence boundary or frame gap')
        if not bool(availability.any()):
            return tuple(rgb), None, dict(missing_exact_RGB=True, state_cleared=True)
        if cue is None or cue.shape[0] != batch:
            raise ValueError('Available observations require current cue pixels')
        if valid is None:
            valid = torch.ones_like(cue[:, :1], dtype=torch.bool)
        latent, masks, image_stats = self.encoder(cue, valid)
        outputs, filtered, covariances, diagnostics = [], [], [], []
        for index, (base, current, score, mask) in enumerate(zip(rgb, latent, scores, masks)):
            if base.shape[-2:] != current.shape[-2:] or score.shape != base[:, :1].shape:
                raise ValueError('Cue and RGB grids must have corresponding image geometry')
            features = quality_features(base, current, score, mask, image_stats)
            gate = self.quality[index](features).sigmoid() if self.use_quality else features.new_full((batch, 1), .5)
            variance = None
            gain = torch.ones_like(current[:, :1])
            if self.mode == 'temporal' and self.use_memory:
                variance = F.softplus(self.noise[index](features)).reshape(batch, 1, 1, 1) + .001
                if state is not None:
                    if self.use_motion and flow_to_previous is None:
                        raise ValueError('Past state requires current-to-previous RGB flow')
                    flow = flow_to_previous if self.use_motion else current.new_zeros((batch, 2, *cue.shape[-2:]))
                    previous, support = warp_previous(state.latent[index], flow)
                    predicted_variance, _ = warp_previous(state.covariance[index], flow)
                    old_valid, _ = warp_previous(state.valid[index].float(), flow)
                    support &= old_valid > .999
                    predicted_variance = predicted_variance + F.softplus(self.process_noise[index]) + 1e-5
                    gain = predicted_variance / (predicted_variance + variance)
                    gain = torch.where(support & mask, gain, torch.ones_like(gain))
                    current = (1 - gain) * previous + gain * current
                    variance = (1 - gain) * predicted_variance
                    variance = torch.where(support & mask, variance,
                        (F.softplus(self.noise[index](features)).reshape(batch, 1, 1, 1) + .001).expand_as(variance))
                variance = variance.expand_as(current[:, :1])
            filtered.append(current)
            covariances.append(variance)
            projection = self.output[index](current)
            probability = score.detach().float().clamp(0, 1)
            trust = (1 - probability) * (probability < .6) if self.protect_rgb else torch.ones_like(probability)
            scale = base.float().square().mean(1, keepdim=True).add(1e-6).sqrt()
            correction = self.bound * scale * projection.tanh() * trust * gate[:, :, None, None] * mask
            value = (base.float() + correction).to(base.dtype)
            outputs.append(torch.where(availability[:, None, None, None], value, base))
            diagnostics.append(dict(gate=gate, gain=gain, covariance=variance))
        next_state = None
        if self.mode == 'temporal' and self.use_memory:
            next_state = NativeTemporalState(sequence, frame, tuple(filtered), tuple(covariances), masks)
        return tuple(outputs), next_state, dict(levels=diagnostics, missing_exact_RGB=False)
