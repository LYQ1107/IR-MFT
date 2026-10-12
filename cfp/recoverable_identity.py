"""Learned soft reliability of bounded past identity tokens, with no GT inputs."""
import torch
from torch import nn
from torch.nn import functional as F

from .predictive import checked_history


def reliability_evidence(features, frames, valid, query_frame):
    source, time, lengths, available, last = checked_history(features, frames, valid, query_frame)
    n, width, _ = source.shape
    similarity = source @ source.transpose(1, 2)
    pairs = valid[:, :, None] & valid[:, None, :]
    others = pairs & ~torch.eye(width, dtype=torch.bool, device=source.device)[None]
    count = others.sum(-1).clamp_min(1)
    average = (similarity * others).sum(-1) / count
    minimum = similarity.masked_fill(~others, 1.).min(-1).values
    maximum = similarity.masked_fill(~others, -1.).max(-1).values
    mean = F.normalize(source.sum(1), dim=-1)
    to_mean = (source * mean[:, None]).sum(-1)
    to_last = (source * last[:, None]).sum(-1)
    to_first = (source * source[:, :1]).sum(-1)
    position = torch.arange(width, device=source.device, dtype=source.dtype)[None].expand(n, -1) / max(width - 1, 1)
    previous = torch.zeros_like(to_mean)
    if width > 1:
        previous[:, 1:] = (source[:, 1:] * source[:, :-1]).sum(-1)
    evidence = torch.stack([average, minimum, maximum, to_mean, to_last, to_first, time[..., 0], time[..., 1], position, previous], -1)
    return source, evidence * valid[..., None], available


class RecoverableIdentityState(nn.Module):
    """Soft evidence-based restoration; the first token has no privileged truth."""
    def __init__(self, hidden=64):
        super().__init__()
        self.token_head = nn.Sequential(nn.Linear(10, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.confidence_head = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.token_head[-1].weight)
        nn.init.zeros_(self.token_head[-1].bias)

    def details(self, features, frames, valid, query_frame):
        source, evidence, available = reliability_evidence(features, frames, valid, query_frame)
        token_logits = self.token_head(evidence).squeeze(-1)
        reliability = torch.sigmoid(token_logits) * valid
        weights = reliability / reliability.sum(1, keepdim=True).clamp_min(1e-8)
        prototype = F.normalize((source * weights[..., None]).sum(1), dim=-1)
        prototype = prototype * available[:, None]
        mean = reliability.sum(1) / valid.sum(1).clamp_min(1)
        variance = ((reliability - mean[:, None]).square() * valid).sum(1) / valid.sum(1).clamp_min(1)
        contrast = evidence[..., 3].masked_fill(~valid, 1.).min(1).values
        concentration = weights.square().sum(1)
        confidence = self.confidence_head(torch.stack([mean, variance, contrast, concentration], -1)).squeeze(-1)
        if not torch.isfinite(prototype).all() or not torch.isfinite(token_logits).all():
            raise ValueError('Nonfinite recoverable identity state')
        return prototype, available, token_logits, confidence

    def forward(self, features, frames, valid, query_frame):
        return self.details(features, frames, valid, query_frame)[:2]


def reliability_supervision_loss(logits, targets, known):
    """Balance correct/wrong classes without treating unknown labels as wrong."""
    valid = known.bool()
    losses = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
    groups = []
    for positive in (False, True):
        selected = valid & (targets.bool() == positive)
        if selected.any():
            groups.append(losses[selected].mean())
    return torch.stack(groups).mean() if groups else logits.sum() * 0.
