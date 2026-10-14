"""Causal native wrapper for the independently trained FDTA-style adapter."""
import math

import torch
from torch import nn


class NativeFDTAWrapper(nn.Module):
    """Expose a current/history adapter as the native-local identity protocol.

    The wrapper consumes only actual accepted past DINO proposals and current
    DINO proposal crops.  It never sees physical labels, GT boxes, or future
    rows.  A track representation is obtained from its latest accepted past
    observation; candidate representations are conditioned on that same
    strictly-past history and compared by cosine similarity.
    """
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter
        self.logit_scale = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def online(self, history, history_fraction, history_global, times, history_valid,
               candidates, candidate_fraction, candidate_global, frame):
        # ``candidates`` is the local K4 tensor retained by the shared native
        # producer; this adapter deliberately consumes the matched CLS384
        # ``candidate_global`` column so its coordinate is explicit.
        candidates = candidate_global
        if history_global.ndim != 3 or candidates.ndim != 2 or history_global.shape[2] != 384 or candidates.shape[1] != 384:
            raise ValueError("Native FDTA requires aligned DINO384 global coordinates")
        n, length, _ = history_global.shape; m = len(candidates)
        valid = history_valid.bool()
        available = valid.any(1)
        safe = valid.clone(); safe[~available, 0] = True
        lengths = valid.long().sum(1).clamp_min(1)
        last = history_global[torch.arange(n), lengths - 1]
        last_fraction = history_fraction[torch.arange(n), lengths - 1]
        track = self.adapter(last, history_global, safe, last_fraction)["embedding"]
        if not m:
            return dict(scores=candidates.new_empty((n, 0)), history_available=available,
                        current=dict(valid=torch.zeros(0, dtype=torch.bool), embedding=candidates))
        flat_history = history_global[:, None].expand(n, m, length, 384).reshape(n*m, length, 384)
        flat_valid = safe[:, None].expand(n, m, length).reshape(n*m, length)
        flat_current = candidates[None].expand(n, m, 384).reshape(n*m, 384)
        flat_fraction = candidate_fraction[None].expand(n, m, candidate_fraction.shape[1]).reshape(n*m, candidate_fraction.shape[1])
        encoded = self.adapter(flat_current, flat_history, flat_valid, flat_fraction)["embedding"].reshape(n, m, 384)
        scores = (track[:, None, :] * encoded).sum(-1).clamp(-1, 1) * self.logit_scale.exp()
        current_valid = candidate_fraction > 0
        return dict(scores=scores, history_available=available,
                    current=dict(valid=current_valid, embedding=encoded))
