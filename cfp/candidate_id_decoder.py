"""MOTIP-inspired in-context ID decoding with an explicit temporary ID dictionary.

This small frozen-feature control is not the official detector/training system.
Current detections decode active track ID codes from their actual past tokens.
"""
import torch
from torch import nn

from .predictive import checked_history


class CandidateIDDecoder(nn.Module):
    def __init__(self, dimension=512, hidden=64, capacity=128):
        super().__init__()
        self.dimension, self.hidden, self.capacity = dimension, hidden, capacity
        self.visual = nn.Linear(dimension, hidden)
        self.time = nn.Linear(3, hidden)
        self.dictionary = nn.Embedding(capacity + 1, hidden)
        self.memory = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(),
                                    nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden))
        self.decoder = nn.TransformerDecoderLayer(hidden, 4, 2 * hidden, dropout=0., batch_first=True)
        self.output = nn.LayerNorm(hidden)

    def forward(self, history, times, valid, frame, candidates, codes=None):
        # One actual frame graph: N tracks, M detections, at most8 past tokens.
        source, time, _, available, _ = checked_history(history, times, valid, frame, 'log1p_15000')
        n, length, _ = history.shape
        m = len(candidates)
        if candidates.shape != (m, self.dimension) or n > self.capacity or length == 0:
            raise ValueError('Aligned bounded whole-frame identity decoder inputs required')
        if not torch.isfinite(candidates).all():
            raise ValueError('Finite current observations required')
        if codes is None:
            codes = torch.arange(n, device=history.device)
        if codes.shape != (n,) or codes.dtype != torch.int64 or len(codes.unique()) != n:
            raise ValueError('Distinct temporary in-context ID codes required')
        if n and ((codes < 0).any() or (codes >= self.capacity).any()):
            raise ValueError('Temporary codes must exclude reserved newborn class')
        active_codes = torch.cat([codes, torch.tensor([self.capacity], device=history.device)])
        if m == 0:
            return history.new_zeros(0, n + 1)
        current = self.visual(candidates.float())
        if valid.any():
            early = (torch.arange(length, device=history.device)[None] < 3) & valid
            observed_time = torch.cat([time, early[..., None].float()], dim=-1)
            memory = self.visual(source) + self.time(observed_time) + self.dictionary(codes)[:, None]
            memory = memory + self.memory(memory)
            visible = memory[valid][None]
            decoded = self.decoder(current[None], visible).squeeze(0)
        else:
            decoded = current
        logits = self.output(decoded) @ self.dictionary(active_codes).T / self.hidden ** .5
        # Track IDs without an observable past cannot acquire an identity from GT.
        if n:
            logits[:, :n] = logits[:, :n].masked_fill(~available[None], -torch.inf)
        return logits
