"""Ordinary supervised Transformer over the same past/current candidate inputs."""
import torch
from torch import nn
from torch.nn import functional as F

from .predictive import checked_history


class CandidateTransformer(nn.Module):
    def __init__(self, dimension=512, latent=64):
        super().__init__()
        self.latent = latent
        self.visual = nn.Linear(dimension, latent)
        self.time = nn.Linear(2, latent)
        self.geometry = nn.Linear(10, latent)
        self.kind = nn.Embedding(3, latent)
        self.null_token = nn.Parameter(torch.zeros(latent))
        layer = nn.TransformerEncoderLayer(latent, 4, 2 * latent, dropout=0., batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.score = nn.Sequential(nn.Linear(latent, latent), nn.GELU(), nn.Linear(latent, 1))

    def forward(self, history, times, valid, frame, candidates, candidate_valid, evidence):
        source, time, _, available, _ = checked_history(history, times, valid, frame, 'log1p_15000')
        b, m, _ = candidates.shape
        if candidates.shape[:2] != candidate_valid.shape or evidence.shape[:2] != candidate_valid.shape:
            raise ValueError('Aligned actual candidate graph required')
        if not torch.isfinite(candidates).all() or not torch.isfinite(evidence).all():
            raise ValueError('Finite actual current observations required')
        past = self.visual(source) + self.time(time) + self.kind.weight[0]
        current = self.visual(candidates.float()) + self.geometry(evidence[..., 9:19]) + self.kind.weight[1]
        null = (self.null_token + self.kind.weight[2])[None, None].expand(b, 1, -1)
        tokens = torch.cat([past, current, null], dim=1)
        visible = torch.cat([valid, candidate_valid, torch.ones(b, 1, device=history.device, dtype=torch.bool)], dim=1)
        if b == 0:
            return dict(scores=history.new_zeros(b, m + 1), predicted_latent=history.new_zeros(b, m, self.latent))
        decoded = self.encoder(tokens, src_key_padding_mask=~visible)[:, history.shape[1]:]
        scores = self.score(decoded).squeeze(-1)
        scores[:, :m] = scores[:, :m].masked_fill(~candidate_valid, -torch.inf)
        return dict(scores=scores, predicted_latent=F.normalize(decoded[:, :m], dim=-1), history_available=available)
