"""Classical structure-verified values and RGB-anchored relative semantics.

MIND-style self-similarity, inverse consistency and response differencing have
prior art. This is an experimental task-specific combination, not a novelty claim.
"""
import torch
from torch import nn
from torch.nn import functional as F


def structural_weights(query, key, query_valid, key_valid, radius=2,
                       temperature=.1, cycle=True, use_structure=True, tolerance=1.):
    """Nonlearned local weights; inverse consistency is measured in grid cells."""
    if query.shape != key.shape or query_valid.shape != key_valid.shape:
        raise ValueError('Paired structural grids required')
    batch, channels, height, width = query.shape
    count = (2*radius+1)**2; length = height*width
    q = query.float().flatten(2)
    patches = F.unfold(key.float(), 2*radius+1, padding=radius).reshape(batch, channels, count, length)
    inside = F.unfold(torch.ones_like(key[:, :1]), 2*radius+1, padding=radius) > 0
    if not use_structure:
        weights = inside.float()/inside.sum(1, keepdim=True).clamp_min(1)
        return weights, torch.ones_like(query[:, :1], dtype=torch.float32)
    valid = (F.unfold(key_valid.float(), 2*radius+1, padding=radius) > .5) & query_valid.flatten(2)
    distance = (q[:, :, None]-patches).abs().mean(1)
    masked = distance.masked_fill(~valid, 1e4)
    best_cost, best = masked.min(1)
    weights = (-masked/temperature).softmax(1)*valid
    weights = weights/weights.sum(1, keepdim=True).clamp_min(1e-12)
    trusted = valid.any(1)
    if cycle:
        reverse_patches = F.unfold(query.float(), 2*radius+1, padding=radius).reshape(batch, channels, count, length)
        reverse_valid = (F.unfold(query_valid.float(), 2*radius+1, padding=radius) > .5) & key_valid.flatten(2)
        reverse = (key.float().flatten(2)[:, :, None]-reverse_patches).abs().mean(1).masked_fill(~reverse_valid, 1e4).argmin(1)
        offsets = torch.stack(torch.meshgrid(torch.arange(-radius, radius+1, device=query.device),
                                            torch.arange(-radius, radius+1, device=query.device), indexing='ij'), -1).reshape(-1, 2)
        displacement = offsets[best]
        yy, xx = torch.meshgrid(torch.arange(height, device=query.device), torch.arange(width, device=query.device), indexing='ij')
        ty = (yy.flatten()[None]+displacement[:, :, 0]).clamp(0, height-1)
        tx = (xx.flatten()[None]+displacement[:, :, 1]).clamp(0, width-1)
        backward = offsets[reverse.gather(1, ty*width+tx)]
        trusted &= ((displacement+backward).float().square().sum(-1) <= tolerance**2)
    confidence = torch.exp(-best_cost/temperature)*trusted
    return weights, confidence.reshape(batch, 1, height, width)


class StructuralLevel(nn.Module):
    def __init__(self, channels, latent=32, radius=2, temperature=.1, bound=.5,
                 risk_threshold=.6, use_structure=True, cycle=True, tolerance=1.):
        super().__init__()
        self.radius = radius; self.temperature = temperature; self.bound = bound
        self.risk_threshold = risk_threshold; self.use_structure = use_structure
        self.cycle = cycle; self.tolerance = tolerance
        self.value = nn.Conv2d(channels, latent, 1, bias=False)
        self.output = nn.Conv2d(latent, channels, 1, bias=True)
        nn.init.orthogonal_(self.value.weight.flatten(1))
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, rgb, auxiliary, teacher_score, query, key, query_valid, key_valid,
                availability=True, null=False):
        if isinstance(availability, bool):
            if not availability:
                return rgb
        elif not bool(availability.any()):
            return rgb
        if auxiliary is None or query is None or key is None:
            raise ValueError('Available views require features and structural descriptors')
        if rgb.shape != auxiliary.shape:
            raise ValueError('Corresponding frozen feature shapes required')
        height, width = rgb.shape[-2:]
        q = F.adaptive_avg_pool2d(query.float(), (height, width))
        k = F.adaptive_avg_pool2d(key.float(), (height, width))
        qv = F.adaptive_avg_pool2d(query_valid.float(), (height, width)) >= .5
        kv = F.adaptive_avg_pool2d(key_valid.float(), (height, width)) >= .5
        if null:
            shift = (height//4, width//4)
            auxiliary = torch.roll(auxiliary, shift, (-2, -1))
            k = torch.roll(k, shift, (-2, -1)); kv = torch.roll(kv, shift, (-2, -1))
        weights, confidence = structural_weights(q, k, qv, kv, self.radius,
            self.temperature, self.cycle, self.use_structure, self.tolerance)
        source = rgb.float(); values = self.value(auxiliary.float())
        patches = F.unfold(values, 2*self.radius+1, padding=self.radius)
        patches = patches.reshape(len(rgb), values.shape[1], weights.shape[1], height*width)
        aligned = (patches*weights[:, None]).sum(2).reshape_as(values)
        residual = self.output(aligned-self.value(source))
        scale = source.square().mean(1, keepdim=True).add(1e-6).sqrt()
        probability = teacher_score.detach().float().clamp(0, 1)
        trust = confidence*(1-probability)*(probability < self.risk_threshold)
        result = (source+self.bound*scale*residual.tanh()*trust).to(rgb.dtype)
        if not isinstance(availability, bool):
            result = torch.where(availability.to(device=rgb.device, dtype=torch.bool).reshape(-1, 1, 1, 1), result, rgb)
        return result


class StructuralRepair(nn.Module):
    def __init__(self, channels=(320, 640, 1280), latent=32, radius=2, temperature=.1,
                 bound=.5, risk_threshold=.6, variant='full', tolerance=1.):
        super().__init__()
        if variant not in ['full', 'no_structure', 'no_cycle', 'no_centering']:
            raise ValueError('Unregistered structural variant')
        self.variant = variant
        self.levels = nn.ModuleList([StructuralLevel(c, latent, radius, temperature, bound,
            risk_threshold, variant != 'no_structure', variant not in ['no_structure', 'no_cycle'], tolerance)
            for c in channels])

    def forward(self, rgb, auxiliary, teacher_scores, query, key, query_valid, key_valid,
                availability=True, null=False):
        if len(rgb) != len(self.levels):
            raise ValueError('All original feature-pyramid levels required')
        return tuple(layer(value, None if auxiliary is None else auxiliary[index], teacher_scores[index],
                           query, key, query_valid, key_valid, availability, null)
                     for index, (layer, value) in enumerate(zip(self.levels, rgb)))


def relative_semantics(current, null, teacher, centered=True, bound=2.):
    """Raw YOLOX logits with frozen RGB geometry and bounded semantic influence."""
    if current.shape != null.shape or teacher.shape != current.shape:
        raise ValueError('Use corresponding dense original anchors')
    if bound <= 0:
        raise ValueError('Positive semantic bound required')
    reference = null if centered else teacher.detach()
    result = teacher.detach().clone()
    result[:, :, 4:] = teacher[:, :, 4:].detach()+bound*torch.tanh((current[:, :, 4:]-reference[:, :, 4:])/bound)
    return result
