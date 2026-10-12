"""Predict candidate-relative identity evidence in frozen raw RGB coordinates.

The student consumes past/current similarity relations, observed early-group
membership, time and current geometry. A training-only EMA encoder represents a
known clean current/future identity's relation to the same current candidates.
No physical labels, target crop, teacher or future appear in student forward.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .predictive import checked_history


def candidate_relations(history, candidates, candidate_valid):
    """[B,M,T,2]: raw cosine and margin over other legal current candidates."""
    cosine = torch.einsum('btd,bmd->bmt', F.normalize(history, dim=-1), F.normalize(candidates, dim=-1))
    b, m, t = cosine.shape
    if m > 1:
        masked = cosine.masked_fill(~candidate_valid[..., None], -1e4)
        top = masked.topk(2, dim=1)
        other = top.values[:, :1].expand(b, m, t).clone()
        other.scatter_(1, top.indices[:, :1], top.values[:, 1:2])
        margin = (cosine - other).clamp(-.2, .2) * 10
        margin = torch.where((candidate_valid.sum(1) > 1)[:, None, None], margin, torch.zeros_like(margin))
    else:
        margin = torch.zeros_like(cosine)
    return torch.stack([cosine, margin], dim=-1)


class CandidateRelationStudent(nn.Module):
    def __init__(self, latent=16):
        super().__init__()
        self.latent = latent
        self.pair_encoder = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, latent), nn.LayerNorm(latent))
        self.time_encoder = nn.Linear(3, latent)
        self.reliability = nn.Sequential(nn.Linear(2 * latent + 2, 32), nn.GELU(), nn.Linear(32, 1))
        self.predictor = nn.Sequential(nn.Linear(2 * latent + 10, 32), nn.GELU(), nn.Linear(32, latent))
        self.correction = nn.Linear(latent, 1)
        self.fusion = nn.Sequential(nn.Linear(latent + 10, 16), nn.GELU(), nn.Linear(16, 1))
        self.null = nn.Sequential(nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.reliability[-1].weight); nn.init.zeros_(self.reliability[-1].bias)
        nn.init.zeros_(self.correction.weight); nn.init.zeros_(self.correction.bias)
        nn.init.zeros_(self.fusion[-1].weight); nn.init.constant_(self.fusion[-1].bias, math.log(.3 / .7))
        nn.init.zeros_(self.null[-1].weight); nn.init.constant_(self.null[-1].bias, math.log(.97 / .03))

    def forward(self, history, times, valid, frame, candidates, candidate_valid, evidence, early=None):
        source, _, length, available, _ = checked_history(history, times, valid, frame, 'log1p_15000')
        if candidate_valid.dtype != torch.bool or valid.dtype != torch.bool or candidates.shape[:2] != candidate_valid.shape:
            raise ValueError('Explicit aligned masks required')
        if not torch.isfinite(candidates).all() or not torch.isfinite(evidence).all():
            raise ValueError('Finite current candidate evidence required')
        b, m, _ = candidates.shape
        if m == 0 or b == 0:
            return dict(scores=history.new_zeros(b, m + 1), predicted_latent=history.new_zeros(b, m, self.latent),
                attention=history.new_zeros(b, m, history.shape[1]), history_available=available,
                attention_logits=history.new_zeros(b, m, history.shape[1]), fusion_weight=history.new_zeros(b, m))
        if early is None:
            early = valid & (torch.arange(history.shape[1], device=history.device)[None] < 3)
        if early.shape != valid.shape or early.dtype != torch.bool or (early & ~valid).any():
            raise ValueError('Observed early-role mask must refer to visible past tokens')
        relations = candidate_relations(source, candidates.detach().float(), candidate_valid)
        encoded = self.pair_encoder(relations)
        age = torch.log1p((frame[:, None] - times).clamp_min(0).float()) / math.log(15001)
        delta = torch.zeros_like(age)
        if history.shape[1] > 1:
            delta[:, 1:] = torch.log1p((times[:, 1:] - times[:, :-1]).clamp_min(0).float()) / math.log(15001)
        temporal = self.time_encoder(torch.stack([age, delta, early.float()], dim=-1)) * valid[..., None]
        mean = F.normalize((source * valid[..., None]).sum(1), dim=-1)
        reference = F.normalize((source * early[..., None]).sum(1), dim=-1)
        agreement = torch.stack([(source * mean[:, None]).sum(-1), (source * reference[:, None]).sum(-1)], dim=-1)
        observable = torch.cat([encoded, temporal[:, None].expand_as(encoded), agreement[:, None].expand(b, m, history.shape[1], 2)], dim=-1)
        attention_logits = self.reliability(observable).squeeze(-1)
        safe = valid.clone(); safe[~available, 0] = True
        weights = attention_logits.masked_fill(~safe[:, None], -torch.inf).softmax(-1) * valid[:, None]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        support = (weights * relations[..., 0]).sum(-1)
        context = (weights[..., None] * (encoded + temporal[:, None])).sum(-2)
        legal = candidate_valid[..., None]
        pool = (context * legal).sum(1) / candidate_valid.sum(1).clamp_min(1)[:, None]
        # Only cached current geometry/confidence channels; old full-history
        # scalar similarities/prototype costs cannot leak a masked token.
        current = evidence[..., 9:19]
        predicted = self.predictor(torch.cat([context, pool[:, None].expand_as(context), current], dim=-1))
        alpha = self.fusion(torch.cat([predicted, current], dim=-1)).squeeze(-1).sigmoid()
        geometry = 1 - evidence[..., 9] * evidence[..., 14]
        appearance = (1 - support).clamp(0, 2) / 2
        cost = (1 - alpha) * geometry + alpha * appearance - .05 * self.correction(predicted).squeeze(-1).tanh()
        cost = torch.where(available[:, None], cost, geometry)
        scores = (-cost / .1).masked_fill(~candidate_valid, -torch.inf)
        masked_support = support.masked_fill(~candidate_valid, -1e4)
        best = masked_support.max(1).values
        second = masked_support.topk(min(2, m), dim=1).values[:, -1]
        second = torch.where(candidate_valid.sum(1) > 1, second, best)
        best = torch.where(candidate_valid.any(1), best, torch.zeros_like(best))
        second = torch.where(candidate_valid.any(1), second, torch.zeros_like(second))
        safe_scores = torch.where(candidate_valid.any(1)[:, None], scores, torch.zeros_like(scores))
        probabilities = safe_scores.softmax(1) * candidate_valid
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1) / candidate_valid.sum(1).clamp_min(2).float().log()
        conflict = 1 - (mean * reference).sum(1)
        geometry_min = geometry.masked_fill(~candidate_valid, 1.).min(1).values
        gap = evidence[:, 0, 16]
        null_features = torch.stack([best, best - second, entropy, geometry_min, conflict, gap], dim=-1)
        null_cost = self.null(null_features).squeeze(-1).sigmoid()
        all_scores = torch.cat([scores, (-null_cost / .1)[:, None]], dim=1)
        if not torch.isfinite(all_scores[:, -1]).all() or not torch.isfinite(scores[candidate_valid]).all():
            raise ValueError('Nonfinite legal relation score')
        return dict(scores=all_scores, predicted_latent=F.normalize(predicted, dim=-1),
            attention=weights, attention_logits=attention_logits, history_available=available, fusion_weight=alpha)


class CandidateRelationTarget(nn.Module):
    def __init__(self, student, decay=.99):
        super().__init__()
        if not 0 <= decay < 1:
            raise ValueError('Bounded EMA decay required')
        self.encoder = copy.deepcopy(student.pair_encoder).requires_grad_(False).eval()
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def forward(self, clean_identity, candidates, candidate_valid):
        relations = candidate_relations(clean_identity.detach()[:, None], candidates.detach(), candidate_valid).squeeze(2)
        return F.normalize(self.encoder(relations), dim=-1).detach()

    @torch.no_grad()
    def update(self, student, successful):
        if not successful:
            return
        for target, value in zip(self.encoder.parameters(), student.pair_encoder.parameters()):
            target.lerp_(value.detach(), 1 - self.decay)
        for target, value in zip(self.encoder.buffers(), student.pair_encoder.buffers()):
            target.copy_(value)
        self.updates += 1
        if any(p.grad is not None for p in self.parameters()):
            raise RuntimeError('Target encoder has a gradient')

