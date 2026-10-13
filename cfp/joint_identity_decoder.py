"""Full-frame causal ID competition on observable, frozen visual features.

This independently implemented head is not official MOTIP. Columns refer to
the supplied track order, never physical fish IDs. The last column means no
match to any supplied old track; it does not by itself authorize a birth.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .predictive import checked_history


class JointFishIDDecoder(nn.Module):
    def __init__(self, dimension=512, hidden=64, heads=4, capacity=128,
                 track_competition=True, detection_competition=True,
                 motion_bias=True, temporal_memory=True, architecture='joint'):
        super().__init__()
        if hidden % heads or min(dimension, hidden, heads, capacity) <= 0:
            raise ValueError('Positive compatible decoder dimensions required')
        self.dimension, self.hidden, self.capacity = dimension, hidden, capacity
        self.track_competition = track_competition
        self.detection_competition = detection_competition
        self.motion_bias, self.temporal_memory = motion_bias, temporal_memory
        if architecture not in ('joint', 'pair'):
            raise ValueError('Registered joint or same-input independent pair architecture required')
        self.architecture = architecture
        self.visual = nn.Linear(dimension, hidden)
        self.time = nn.Linear(2, hidden)
        # age, time since last observation, active/lost flag: all observable.
        self.track_state = nn.Linear(3, hidden)
        self.confidence = nn.Linear(1, hidden)
        self.dictionary = nn.Embedding(capacity, hidden)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, hidden))
        self.null_track = nn.Parameter(torch.zeros(1, 1, hidden))
        self.null_detection = nn.Parameter(torch.zeros(1, 1, hidden))
        self.pool = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.track_attention = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.detection_attention = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.track_to_detection = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.detection_to_track = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)
        self.track_norm = nn.LayerNorm(hidden)
        self.detection_norm = nn.LayerNorm(hidden)
        self.track_key = nn.Linear(hidden, hidden)
        self.detection_query = nn.Linear(hidden, hidden)
        self.null_score = nn.Linear(hidden, 1)
        self.geometry_scale = nn.Parameter(torch.tensor(1.))
        self.appearance_scale = nn.Parameter(torch.tensor(1.))
        self.residual = nn.Sequential(nn.Linear(2 * hidden + 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.pair_score = nn.Sequential(nn.Linear(2 * hidden + 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        # Neutral initial residual, including when a full-ID head is untrained.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, history, times, valid, frame, detections, detection_valid,
                geometry, appearance, track_state, confidence, codes=None, pair_valid=None):
        source, age, lengths, available, last = checked_history(
            history, times, valid, frame, 'log1p_15000')
        n, length, dimension = history.shape
        m = len(detections)
        if (dimension != self.dimension or detections.shape != (m, dimension)
                or detection_valid.shape != (m,) or detection_valid.dtype != torch.bool
                or geometry.shape != (n, m) or appearance.shape != (n, m)
                or track_state.shape != (n, 3) or confidence.shape != (m,)
                or n > self.capacity or valid.dtype != torch.bool):
            raise ValueError('Aligned whole-frame observable inputs required')
        if not all(torch.isfinite(x).all() for x in (detections, geometry, appearance, track_state, confidence)):
            raise ValueError('Finite whole-frame inputs required')
        if (confidence < 0).any() or (confidence > 1).any():
            raise ValueError('Detection confidence must lie in [0,1]')
        if codes is None:
            codes = torch.arange(n, device=history.device)
        if (codes.shape != (n,) or codes.dtype != torch.int64 or len(codes.unique()) != n
                or (codes < 0).any() or (codes >= self.capacity).any()):
            raise ValueError('Distinct temporary dictionary codes required')
        if pair_valid is None:
            pair_valid = torch.ones(n, m, dtype=torch.bool, device=history.device)
        if pair_valid.shape != (n, m) or pair_valid.dtype != torch.bool:
            raise ValueError('Explicit observable pair eligibility required')
        legal = pair_valid & available[:, None] & detection_valid[None]
        if m == 0:
            return dict(logits=history.new_empty(0, n + 1), residual=history.new_empty(n, 0), legal=legal)
        current = self.visual(F.normalize(detections.detach().float(), dim=-1))
        current = current + self.confidence(confidence[:, None])
        if n:
            memory = self.visual(source) + self.time(age)
            safe = valid.clone()
            safe[~available, 0] = True  # padded empty row cannot cause softmax NaNs.
            if self.temporal_memory:
                track = self.pool(self.pool_query.expand(n, -1, -1), memory, memory,
                                  key_padding_mask=~safe, need_weights=False)[0][:, 0]
            else:
                track = self.visual(last)
            track = track + self.track_state(track_state) + self.dictionary(codes)
            if self.architecture == 'joint' and self.track_competition and available.any():
                track = track + self.track_attention(track[None], track[None], track[None],
                    key_padding_mask=~available[None], need_weights=False)[0][0]
            # Add a valid unmatched token to every attention pool. Entirely
            # occluded/gated frames remain finite without inventing observations.
            det_memory = torch.cat([current[None], self.null_detection], dim=1)
            bias = -(F.softplus(self.geometry_scale) * geometry
                     + F.softplus(self.appearance_scale) * appearance) if self.motion_bias else torch.zeros_like(geometry)
            track_bias = torch.cat([bias.masked_fill(~legal, -torch.inf), bias.new_zeros(n, 1)], dim=1)
            if self.architecture == 'joint':
                track = track + self.track_to_detection(track[None], det_memory, det_memory,
                    attn_mask=track_bias, need_weights=False)[0][0]
            track = self.track_norm(track)
        else:
            track = current.new_empty(0, self.hidden)
            bias = geometry
        if self.architecture == 'joint' and self.detection_competition and detection_valid.any():
            current = current + self.detection_attention(current[None], current[None], current[None],
                key_padding_mask=~detection_valid[None], need_weights=False)[0][0]
        track_memory = torch.cat([track[None], self.null_track], dim=1)
        det_bias = torch.cat([bias.T.masked_fill(~legal.T, -torch.inf), current.new_zeros(m, 1)], dim=1)
        if self.architecture == 'joint':
            current = current + self.detection_to_track(current[None], track_memory, track_memory,
                attn_mask=det_bias, need_weights=False)[0][0]
        current = self.detection_norm(current)
        edge = torch.cat([track[:, None].expand(n, m, -1), current[None].expand(n, m, -1),
                          geometry[..., None], appearance[..., None]], dim=-1)
        id_logits = self.detection_query(current) @ self.track_key(track).T / math.sqrt(self.hidden)
        if self.architecture == 'pair':
            id_logits = id_logits + self.pair_score(edge).squeeze(-1).T
        if self.motion_bias:
            id_logits = id_logits + bias.T
        id_logits = id_logits.masked_fill(~legal.T, -torch.inf)
        null = self.null_score(current)
        null = torch.where(detection_valid[:, None], null, torch.zeros_like(null))
        logits = torch.cat([id_logits, null], dim=1)
        residual = self.residual(edge).squeeze(-1).masked_fill(~legal, 0.)
        return dict(logits=logits, residual=residual, legal=legal)


def bounded_joint_cost(baseline, residual, legal, maximum_delta=.03):
    """Route 1: preserve baseline edges/lifecycle, bound the joint edit per edge."""
    if (baseline.shape != residual.shape or legal.shape != baseline.shape or legal.dtype != torch.bool
            or not math.isfinite(maximum_delta) or not 0 <= maximum_delta <= .1
            or not torch.isfinite(baseline).all() or not torch.isfinite(residual).all()):
        raise ValueError('Finite aligned baseline and declared bounded residual required')
    return baseline + maximum_delta * torch.tanh(residual) * legal


def decode_joint_assignment(logits, detection_valid):
    """Route 2: globally one-to-one old IDs plus independent unmatched choices.

    Returns one old-track index or -1 per detection. Birth confidence, duplicate
    suppression, confirmation and reactivation remain explicit tracker duties.
    """
    from scipy.optimize import linear_sum_assignment

    if logits.ndim != 2 or logits.shape[1] < 1 or detection_valid.shape != (len(logits),):
        raise ValueError('Detection-by-(track+NULL) logits required')
    if detection_valid.dtype != torch.bool or not torch.isfinite(logits[:, -1]).all() or torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError('Finite NULL option and boolean validity required')
    m, width = logits.shape
    assigned = torch.full((m,), -1, dtype=torch.int64, device=logits.device)
    indices = detection_valid.nonzero(as_tuple=True)[0]
    if not len(indices):
        return assigned
    costs = -F.log_softmax(logits[indices].detach().float(), dim=-1)
    k = len(indices)
    expanded = torch.full((k, width - 1 + k), torch.inf, device=costs.device)
    expanded[:, :width - 1] = costs[:, :-1]
    expanded[torch.arange(k, device=costs.device), width - 1 + torch.arange(k, device=costs.device)] = costs[:, -1]
    rows, columns = linear_sum_assignment(expanded.cpu().numpy())
    for row, column in zip(rows, columns):
        if column < width - 1:
            assigned[indices[row]] = int(column)
    return assigned
