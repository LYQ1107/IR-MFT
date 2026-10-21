"""Fixed raw RGB crop descriptor used as an independent identity cue.

This module deliberately contains no learned weights.  It turns the same
current proposal crop used by the registered DINO producer into a normalized
384-dimensional descriptor: 8x8 pooled R/G/B values, luminance, and two
first-order spatial derivatives.  Letterbox padding is filled with the
visible crop mean and derivatives outside the visible rectangle are removed,
so the descriptor does not treat mean-valued padding as fish texture.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def raw_rgb_descriptor(crops: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
    """Return one unit-norm descriptor per normalized RGB crop.

    ``crops`` follows :func:`cfp.identity_data.crop_to_tensor` (B,3,128,256)
    and ``visibility`` is its geometric letterbox mask (B,1,128,256).  The
    operation is deterministic on CPU/GPU and has no identity, GT, or future
    inputs.  A zero descriptor is returned for an invalid crop.
    """
    if (crops.ndim != 4 or crops.shape[1:] != (3, 128, 256)
            or visibility.shape != (len(crops), 1, 128, 256)
            or crops.dtype not in (torch.float16, torch.float32, torch.float64)
            or visibility.dtype not in (torch.float16, torch.float32, torch.float64)):
        raise ValueError("raw descriptor requires normalized 128x256 RGB crops and masks")
    if not torch.isfinite(crops).all() or not torch.isfinite(visibility).all():
        raise ValueError("raw descriptor inputs must be finite")
    mask = visibility.float().clamp(0, 1)
    valid = mask.sum((1, 2, 3)) > 0
    # Undo ImageNet normalization solely to make the handcrafted cue readable.
    mean = crops.new_tensor([.485, .456, .406])[None, :, None, None]
    std = crops.new_tensor([.229, .224, .225])[None, :, None, None]
    rgb = crops.float() * std + mean
    area = mask.sum((2, 3)).clamp_min(1.)
    visible_mean = (rgb * mask).sum((2, 3)) / area
    rgb = rgb * mask + visible_mean[:, :, None, None] * (1 - mask)
    gray = (rgb[:, 0:1] * .299 + rgb[:, 1:2] * .587 + rgb[:, 2:3] * .114)
    # Forward differences retain orientation/shape information while staying
    # independent of a pretrained feature extractor.
    dx = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0)) * mask
    dy = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1)) * mask
    values = torch.cat([rgb, gray, dx, dy], dim=1)
    pooled = F.adaptive_avg_pool2d(values, (8, 8)).flatten(1)
    result = F.normalize(pooled, dim=1, eps=1e-8)
    result[~valid] = 0
    if not torch.isfinite(result).all():
        raise ValueError("raw descriptor produced nonfinite values")
    return result
