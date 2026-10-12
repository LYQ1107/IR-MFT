"""Candidate-conditioned predictive identity model and training-only EMA target.

Each current candidate queries past identity tokens; scores include explicit NULL.
Teacher/future/labels have no place in the student's forward signature. Ordinary
same-architecture scoring uses exactly the same student without predictive loss.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from .predictive import checked_history
from .recoverability import FEATURE_NAMES


class CandidateJEPAStudent(nn.Module):
    def __init__(self, dimension=512, latent=64, architecture='cross_attention', candidate_context=True):
        super().__init__()
        if architecture not in ('cross_attention', 'gru', 'mlp') or latent % 4:
            raise ValueError('Declared architecture and four-head latent size required')
        self.dimension, self.latent, self.architecture = dimension, latent, architecture
        self.candidate_context = candidate_context
        self.encoder = nn.Sequential(nn.Linear(dimension, latent), nn.LayerNorm(latent), nn.GELU())
        self.time = nn.Linear(2, latent)
        self.edge = nn.Linear(len(FEATURE_NAMES), latent)
        if architecture == 'gru':
            self.temporal = nn.GRU(latent, latent, batch_first=True)
        elif architecture == 'cross_attention':
            self.cross_attention = nn.MultiheadAttention(latent, 4, dropout=0., batch_first=True)
        self.set_attention = nn.MultiheadAttention(latent, 4, dropout=0., batch_first=True) if candidate_context else None
        self.predictor = nn.Sequential(nn.Linear(3 * latent, 2 * latent), nn.GELU(), nn.Linear(2 * latent, latent))
        self.score = nn.Linear(latent, 1)
        self.null = nn.Sequential(nn.Linear(2 * latent, latent), nn.GELU(), nn.Linear(latent, 1))

    def forward(self, history, history_frames, history_valid, frame,
                candidates, candidate_valid, edge_features):
        if history.shape[-1] != self.dimension or candidates.ndim != 3 or candidates.shape[0] != len(history) or candidates.shape[-1] != self.dimension:
            raise ValueError('Frozen identity coordinates differ')
        if candidate_valid.shape != candidates.shape[:2] or edge_features.shape != candidates.shape[:2] + (len(FEATURE_NAMES),):
            raise ValueError('Candidate mask/evidence shape differs')
        if not torch.isfinite(candidates).all() or not torch.isfinite(edge_features).all():
            raise ValueError('Finite current candidate evidence required')
        if history_valid.dtype != torch.bool or candidate_valid.dtype != torch.bool:
            raise ValueError('History/candidate masks must be explicit booleans')
        source, time, lengths, available, _ = checked_history(history, history_frames, history_valid, frame)
        b, m, _ = candidates.shape
        if b == 0:
            return dict(scores=candidates.new_empty(0, m + 1),
                predicted_latent=candidates.new_empty(0, m, self.latent),
                candidate_latent=candidates.new_empty(0, m, self.latent), history_available=available)
        past = self.encoder(source) + self.time(time)
        past = past * history_valid[..., None]
        pooled = past.sum(1) / lengths.clamp_min(1)[:, None]
        current = self.encoder(F.normalize(candidates.detach().float(), dim=-1))
        candidate_latent = F.normalize(current, dim=-1)
        if m == 0:
            empty = current.new_zeros(b, 0, self.latent)
            null = self.null(torch.cat([pooled, torch.zeros_like(pooled)], dim=-1))
            return dict(scores=null, predicted_latent=empty, candidate_latent=empty, history_available=available)
        if self.architecture == 'gru':
            packed = nn.utils.rnn.pack_padded_sequence(past, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False)
            _, state = self.temporal(packed)
            context = state[-1][:, None].expand(b, m, self.latent) * available[:, None, None]
        elif self.architecture == 'cross_attention':
            safe = history_valid.clone()
            safe[~available, 0] = True
            context, _ = self.cross_attention(current, past, past, key_padding_mask=~safe, need_weights=False)
            context = context * available[:, None, None]
        else:
            context = pooled[:, None].expand(b, m, self.latent)
        if self.set_attention is not None:
            safe_candidates = candidate_valid.clone()
            safe_candidates[~candidate_valid.any(1), 0] = True
            interaction, _ = self.set_attention(current, current, current, key_padding_mask=~safe_candidates, need_weights=False)
            current = current + interaction * candidate_valid[..., None]
        # Cached scalar history features were computed before any training mask
        # or corruption. They must not reveal hidden raw tokens to the student.
        # This initial controlled model uses motion/confidence/current crowding
        # evidence only; raw RGB history is consumed through the checked branch.
        causal_evidence = edge_features.clone()
        causal_evidence[..., :9] = 0
        causal_evidence[..., 19:] = 0
        evidence = self.edge(causal_evidence)
        conditioned = self.predictor(torch.cat([current, context, evidence], dim=-1))
        predicted = F.normalize(conditioned, dim=-1)
        # Independent supervised score head remains identical in JEPA/ordinary
        # controls; latent accuracy cannot itself be called ranking improvement.
        scores = (self.score(conditioned).squeeze(-1) +
                  (predicted * candidate_latent).sum(-1) / .1).masked_fill(~candidate_valid, -torch.inf)
        current_pool = (current * candidate_valid[..., None]).sum(1) / candidate_valid.sum(1).clamp_min(1)[:, None]
        null = self.null(torch.cat([pooled, current_pool], dim=-1))
        output = torch.cat([scores, null], dim=1)
        if not torch.isfinite(output[:, -1]).all() or not torch.isfinite(output[:, :-1][candidate_valid]).all():
            raise ValueError('Candidate prediction produced nonfinite legal scores')
        return dict(scores=output, predicted_latent=predicted,
                    candidate_latent=candidate_latent,
                    history_available=available)


class CandidateEMATarget(nn.Module):
    """Training-only encoder, stopped gradients and successful-step updates."""
    def __init__(self, student, decay=.99):
        super().__init__()
        if not 0 <= decay < 1:
            raise ValueError('EMA decay must be in [0,1)')
        self.encoder = copy.deepcopy(student.encoder).requires_grad_(False).eval()
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def forward(self, clean_identity):
        return F.normalize(self.encoder(F.normalize(clean_identity.detach().float(), dim=-1)), dim=-1).detach()

    @torch.no_grad()
    def update(self, student, optimizer_step_succeeded):
        if not optimizer_step_succeeded:
            return
        for target, source in zip(self.encoder.parameters(), student.encoder.parameters()):
            target.lerp_(source.detach(), 1 - self.decay)
        for target, source in zip(self.encoder.buffers(), student.encoder.buffers()):
            target.copy_(source)
        self.updates += 1
        if any(p.grad is not None for p in self.parameters()):
            raise RuntimeError('Teacher received a gradient')

