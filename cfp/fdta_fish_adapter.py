"""Independent FDTA-style fish identity adapters on frozen DINO observations.

This is a same-input adapter baseline, not an official FDTA port.  It keeps
the source paper's useful separation (current identity refinement, optional
temporal context and quality gating) while using the audited Apache DINOv2
384-dimensional crop coordinate already used by Phase VII.  Physical labels
and same-identity past rows are training/evaluation diagnostics only; this
module is not an online memory writer.
"""
import torch
from torch import nn
from torch.nn import functional as F


class FDTAFishAdapter(nn.Module):
    MODES = ("mlp", "temporal", "quality")

    def __init__(self, identities, dimension=384, hidden=128, mode="mlp"):
        super().__init__()
        if mode not in self.MODES or min(identities, dimension, hidden) <= 0:
            raise ValueError("Registered FDTA adapter mode and positive dimensions required")
        self.mode, self.dimension = mode, dimension
        self.input_norm = nn.LayerNorm(dimension)
        self.refine = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, dimension))
        self.temporal = nn.MultiheadAttention(dimension, 4, dropout=0.0, batch_first=True)
        self.temporal_norm = nn.LayerNorm(dimension)
        self.quality = nn.Sequential(nn.Linear(5, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.classifier = nn.Linear(dimension, identities, bias=False)

    def forward(self, current, history=None, history_valid=None, visibility=None):
        if current.ndim != 2 or current.shape[1] != self.dimension or not torch.isfinite(current).all():
            raise ValueError("Finite [B,384] current DINO features required")
        batch = len(current)
        if history is None:
            history = current.new_zeros(batch, 1, self.dimension)
            history_valid = torch.zeros(batch, 1, dtype=torch.bool, device=current.device)
        if history.ndim != 3 or history.shape[0] != batch or history.shape[2] != self.dimension:
            raise ValueError("Aligned strictly-past history features required")
        if history_valid is None or history_valid.shape != history.shape[:2] or history_valid.dtype != torch.bool:
            raise ValueError("Boolean history-valid mask required")
        if not torch.isfinite(history).all():
            raise ValueError("Finite history features required")
        if visibility is None:
            visibility = current.new_ones(batch, 4)
        if visibility.shape != (batch, 4) or not torch.isfinite(visibility).all() or (visibility < 0).any() or (visibility > 1).any():
            raise ValueError("Four finite observable visibility fractions required")
        base = self.input_norm(F.normalize(current.float(), dim=-1))
        residual = self.refine(base)
        context = torch.zeros_like(base)
        if self.mode in ("temporal", "quality"):
            safe = history_valid.clone()
            safe[~safe.any(dim=1), 0] = True
            attended, _ = self.temporal(base[:, None], history.float(), history.float(),
                                        key_padding_mask=~safe, need_weights=False)
            context = self.temporal_norm(attended[:, 0])
        if self.mode == "quality":
            stats = torch.cat((visibility, visibility.mean(1, keepdim=True)), dim=1)
            gate = torch.sigmoid(self.quality(stats))
        else:
            gate = torch.ones(batch, 1, device=current.device, dtype=current.dtype)
        embedding = F.normalize(base + gate * residual + (context if self.mode in ("temporal", "quality") else 0), dim=-1)
        logits = self.classifier(embedding)
        return {"embedding": embedding, "logits": logits, "quality_gate": gate, "context": context}
