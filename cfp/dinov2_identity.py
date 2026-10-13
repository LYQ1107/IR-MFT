"""Pinned, licensed DINOv2 current-crop observations; no history or labels."""
import importlib
from pathlib import Path
import subprocess
import sys

import torch
from torch.nn import functional as F

from .provenance import sha256


SOURCE_COMMIT = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'


def load_pinned_small(reference, checkpoint, imported_proofs):
    """Load audited Apache source and a locally verified safe tensor state."""
    reference = Path(reference).resolve()
    head = subprocess.check_output(['git', '-C', str(reference), 'rev-parse', 'HEAD'], text=True).strip()
    if head != SOURCE_COMMIT or '- **License:** Apache License 2.0' not in (reference / 'MODEL_CARD.md').read_text():
        raise ValueError('Pinned source and explicit model license required')
    for relative, proof in imported_proofs.items():
        if sha256(reference / relative) != proof['sha256']:
            raise ValueError('Audited executable source changed: ' + relative)
    if 'dinov2' in sys.modules:
        if not Path(sys.modules['dinov2'].__file__).resolve().is_relative_to(reference):
            raise ValueError('A different dinov2 package is already imported')
    sys.path.insert(0, str(reference))
    try:
        hub = importlib.import_module('dinov2.hub.backbones')
        model = hub.dinov2_vits14(pretrained=False)
    finally:
        sys.path.remove(str(reference))
    for name, module in list(sys.modules.items()):
        if name == 'dinov2' or name.startswith('dinov2.'):
            filename = getattr(module, '__file__', None)
            if filename:
                path = Path(filename).resolve()
                if not path.is_relative_to(reference) or str(path.relative_to(reference)) not in imported_proofs:
                    raise ValueError('Unrecorded DINOv2 executable dependency: ' + name)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(value).all() for value in state.values() if isinstance(value, torch.Tensor)):
        raise ValueError('Nonfinite pretrained state')
    return model.eval().requires_grad_(False)


def resized_observations(images, masks, size=(126, 252)):
    """Resize existing normalized crops and their observable content masks."""
    if (images.ndim != 4 or images.shape[1] != 3
            or masks.shape != (len(images), 1, *images.shape[-2:])
            or not torch.isfinite(images).all() or not torch.isfinite(masks).all()
            or (masks < 0).any() or (masks > 1).any()
            or len(size) != 2 or any(n <= 0 or n % 14 for n in size)):
        raise ValueError('Aligned finite crops/content masks and patch14-multiple size required')
    images = F.interpolate(images.float(), size=size, mode='bilinear', align_corners=False, antialias=True)
    masks = F.interpolate(masks.float(), size=size, mode='area')
    patch_visibility = F.avg_pool2d(masks, kernel_size=14, stride=14)
    return images, patch_visibility


def spatial_observations(tokens, patch_visibility):
    """Real patch tokens and K4/8 geometric regions, excluding padding mass.

    Adaptive pooling can overlap column bins. Regions are not fish anatomy;
    content fraction is not an occlusion label or identity reliability score.
    """
    if (tokens.ndim != 3 or patch_visibility.ndim != 4 or patch_visibility.shape[1] != 1
            or tokens.shape[:2] != (len(patch_visibility), patch_visibility.shape[2] * patch_visibility.shape[3])
            or not torch.isfinite(tokens).all() or not torch.isfinite(patch_visibility).all()
            or (patch_visibility < 0).any() or (patch_visibility > 1).any()):
        raise ValueError('Aligned finite dense tokens and observable fractions required')
    h, w = patch_visibility.shape[-2:]
    spatial = tokens.float().transpose(1, 2).reshape(len(tokens), tokens.shape[-1], h, w)
    numerator = spatial * patch_visibility
    result = {'patch_mean': F.normalize(numerator.sum((2, 3)) / patch_visibility.sum((2, 3)).clamp_min(1e-6), dim=-1)}
    for count in (4, 8):
        if w < count:
            raise ValueError('Patch grid too narrow for declared local regions')
        visible = F.adaptive_avg_pool2d(patch_visibility, (1, count)).flatten(1)
        pooled = F.adaptive_avg_pool2d(numerator, (1, count)).squeeze(2).transpose(1, 2)
        pooled = pooled / visible[..., None].clamp_min(1e-6)
        result['local' + str(count)] = pooled
        result['visibility' + str(count)] = visible
    return result
