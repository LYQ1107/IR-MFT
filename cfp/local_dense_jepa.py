"""Past-only object-local prediction and current-candidate identity evidence.

Immutable DINO observations are not a trained JEPA. This module learns a local
encoder, temporal context, regional decoder and candidate relation separately.
The target encoder is an independent, stopped training-only EMA copy.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


MODES = ('global', 'global_gru', 'local_attention', 'local_temporal',
         'local_contrastive', 'no_ema', 'no_prediction', 'no_id', 'full')


class ObservedAttention(nn.Module):
    """Missing tokens are zeroed before attention and after each residual."""
    def __init__(self, dimension, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=0., batch_first=True)
        self.norm2 = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(nn.Linear(dimension, dimension*2), nn.GELU(),
                                 nn.Linear(dimension*2, dimension))

    def forward(self, values, valid):
        if not len(values):
            return values
        values = values*valid[..., None]
        safe = valid.clone()
        safe[~safe.any(1), 0] = True
        normalized = self.norm1(values)
        update = self.attention(normalized, normalized, normalized,
                                key_padding_mask=~safe, need_weights=False)[0]
        values = (values+update)*valid[..., None]
        return (values+self.ffn(self.norm2(values)))*valid[..., None]


class LocalObservationEncoder(nn.Module):
    def __init__(self, count=4, dimension=384, heads=6):
        super().__init__()
        self.count, self.dimension = count, dimension
        self.projection = nn.Linear(dimension, dimension, bias=False)
        nn.init.eye_(self.projection.weight)
        self.position = nn.Parameter(torch.randn(count, dimension)*.02)
        self.block = ObservedAttention(dimension, heads)
        self.residual_logit = nn.Parameter(torch.tensor(math.log(.1/.9)))
        self.reliability = nn.Sequential(nn.Linear(dimension+1, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.zeros_(self.reliability[-1].bias)

    def forward(self, observations, content_fraction):
        if (observations.shape != (len(observations), self.count, self.dimension)
                or content_fraction.shape != observations.shape[:2]
                or not torch.isfinite(observations).all() or not torch.isfinite(content_fraction).all()
                or (content_fraction < 0).any() or (content_fraction > 1).any()):
            raise ValueError('Finite aligned real local observations/content fractions required')
        valid = content_fraction > 0
        if not len(observations):
            return dict(tokens=observations, weights=content_fraction, valid=valid,
                        feature=observations.new_zeros(0, self.dimension))
        base = self.projection(observations*valid[..., None])
        refined = self.block((base+self.position)*valid[..., None], valid)
        tokens = (base+self.residual_logit.sigmoid()*refined)*valid[..., None]
        score = self.reliability(torch.cat([F.normalize(tokens, dim=-1), content_fraction[..., None]], -1)).squeeze(-1)
        safe = valid.clone()
        safe[~safe.any(1), 0] = True
        score = score+content_fraction.clamp_min(1.e-6).log()
        weights = score.masked_fill(~safe, -torch.inf).softmax(-1)*valid.any(1)[:, None]
        feature = (tokens*weights[..., None]).sum(1)
        return dict(tokens=tokens, weights=weights, valid=valid, feature=feature)


class LocalDenseStudent(nn.Module):
    def __init__(self, identities, mode='full', count=4, dimension=384, heads=6):
        super().__init__()
        if mode not in MODES or count not in (4, 8) or identities < 1 or dimension % heads:
            raise ValueError('Registered mode, K4/K8, physical fitting vocabulary and width required')
        self.mode, self.count, self.dimension = mode, count, dimension
        self.local_encoder = LocalObservationEncoder(count, dimension, heads)
        self.temporal = ObservedAttention(dimension, heads)
        self.time_encoder = nn.Sequential(nn.Linear(2, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.queries = nn.Parameter(torch.randn(count, dimension)*.02)
        self.horizon_encoder = nn.Linear(1, dimension)
        self.predictor = nn.MultiheadAttention(dimension, heads, dropout=0., batch_first=True)
        self.prediction_norm = nn.LayerNorm(dimension)
        self.prediction_ffn = nn.Sequential(nn.Linear(dimension, dimension*2), nn.GELU(), nn.Linear(dimension*2, dimension))
        self.output_norm = nn.LayerNorm(dimension)
        self.global_projection = nn.Linear(dimension, dimension, bias=False)
        nn.init.eye_(self.global_projection.weight)
        self.gru = nn.GRU(dimension, dimension, batch_first=True)
        self.relation = nn.Sequential(nn.Linear(5, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.relation[-1].weight)
        nn.init.zeros_(self.relation[-1].bias)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.)))
        self.classifier = nn.Linear(dimension, identities, bias=False)
        self._freeze_inactive()

    def _freeze_inactive(self):
        inactive = []
        if self.mode in ('global', 'global_gru'):
            inactive = ['local_encoder.', 'temporal.', 'time_encoder.', 'queries',
                        'horizon_encoder.', 'predictor.', 'prediction_norm.', 'prediction_ffn.', 'output_norm.']
            if self.mode == 'global':
                inactive += ['gru.']
        else:
            inactive = ['global_projection.', 'gru.']
            if self.mode == 'local_attention':
                inactive += ['temporal.', 'time_encoder.']
            if self.mode in ('local_attention', 'local_temporal'):
                inactive += ['queries', 'horizon_encoder.', 'predictor.', 'prediction_norm.', 'prediction_ffn.', 'output_norm.']
            if self.mode == 'no_id':
                inactive += ['relation.', 'classifier.', 'logit_scale']
        for name, parameter in self.named_parameters():
            if any(name.startswith(prefix) for prefix in inactive):
                parameter.requires_grad_(False)

    def current(self, local, content_fraction, global_observation):
        if global_observation.shape != (len(local), self.dimension) or not torch.isfinite(global_observation).all():
            raise ValueError('Same current crop CLS observations required')
        encoded = self.local_encoder(local, content_fraction)
        if self.mode in ('global', 'global_gru'):
            available = encoded['valid'].any(1)
            feature = self.global_projection(global_observation)*available[:, None]
            encoded = dict(tokens=feature[:, None], weights=available[:, None].float(),
                           valid=available[:, None], feature=feature)
        return encoded

    def history_context(self, local, fractions, global_observations, times, valid, frame):
        n, length, count, dimension = local.shape
        if (length < 1 or (count, dimension) != (self.count, self.dimension)
                or fractions.shape != (n, length, count) or global_observations.shape != (n, length, dimension)
                or times.shape != (n, length) or valid.shape != times.shape or valid.dtype != torch.bool
                or not torch.isfinite(times).all() or int(frame) != frame or frame < 1
                or (times[valid] < 1).any() or (times[valid] >= frame).any()
                or not torch.isfinite(local).all() or not torch.isfinite(global_observations).all()):
            raise ValueError('Aligned strictly earlier observed object history required; no current/future input')
        if length > 1 and ((times[:, 1:] <= times[:, :-1]) & valid[:, 1:] & valid[:, :-1]).any():
            raise ValueError('Valid object history must be strictly ordered, never repeated frames')
        if (valid[:, :-1] & ~valid[:, 1:]).any():
            raise ValueError('History is left padded; no hole or trailing padding')
        fractions = fractions*valid[..., None]
        encoded = self.current(local.reshape(-1, count, dimension), fractions.reshape(-1, count),
                               global_observations.reshape(-1, dimension))
        k = encoded['tokens'].shape[1]
        tokens = encoded['tokens'].reshape(n, length, k, dimension)
        observed = encoded['valid'].reshape(n, length, k)
        available = observed.flatten(1).any(1)
        age = torch.log1p((frame-times).clamp_min(0).float())/math.log(15001)
        delta = torch.zeros_like(age)
        if length > 1:
            delta[:, 1:] = torch.log1p((times[:, 1:]-times[:, :-1]).clamp_min(0).float())/math.log(15001)
        if self.mode == 'global_gru':
            result = tokens.new_zeros(n, 1, dimension)
            for i in range(n):
                indices = torch.nonzero(observed[i, :, 0], as_tuple=False).flatten()
                if len(indices):
                    result[i] = self.gru(tokens[i:i+1, indices, 0])[1][-1]
            return dict(context=tokens.flatten(1, 2), valid=observed.flatten(1), available=available,
                        pooled=result, age=age, fractions=fractions)
        if self.mode == 'global':
            pooled = (tokens*observed[..., None]).sum(1)/observed.sum(1).clamp_min(1)[..., None]
            return dict(context=tokens.flatten(1, 2), valid=observed.flatten(1), available=available,
                        pooled=pooled, age=age, fractions=fractions)
        weighted = tokens*encoded['weights'].reshape(n, length, k, 1)*k
        context = weighted
        if self.mode != 'local_attention':
            context = context+self.time_encoder(torch.stack([age, delta], -1))[:, :, None]
            context = self.temporal(context.flatten(1, 2), observed.flatten(1)).reshape(n, length, k, dimension)
        context = context*observed[..., None]
        recency = torch.exp(-(frame-times).clamp_min(0).float()/100)[..., None]*observed
        pooled = (context*recency[..., None]).sum(1)/recency.sum(1).clamp_min(1.e-6)[..., None]
        return dict(context=context.flatten(1, 2), valid=observed.flatten(1), available=available,
                    pooled=pooled, age=age, fractions=fractions)

    def predict(self, context, horizon=0):
        n = len(context['context'])
        if horizon < 0 or int(horizon) != horizon:
            raise ValueError('Nonnegative declared prediction offset; no future observations')
        if self.mode in ('global', 'global_gru', 'local_attention', 'local_temporal'):
            return F.normalize(context['pooled'], dim=-1)*context['available'][:, None, None]
        if not n:
            return context['context'].new_zeros(0, self.count, self.dimension)
        safe = context['valid'].clone()
        safe[~safe.any(1), 0] = True
        age = context['context'].new_full((n, 1), math.log1p(horizon)/math.log(15001))
        queries = self.queries[None].expand(n, -1, -1)+self.horizon_encoder(age)[:, None]
        decoded = self.predictor(queries, context['context'], context['context'],
                                 key_padding_mask=~safe, need_weights=False)[0]
        decoded = self.prediction_norm(queries+decoded)
        decoded = self.output_norm(decoded+self.prediction_ffn(decoded))
        return F.normalize(decoded, dim=-1)*context['available'][:, None, None]

    def relation_scores(self, predicted, candidates, history_available, last_age):
        n, k, dimension = predicted.shape
        m, c, _ = candidates['tokens'].shape
        if dimension != self.dimension or history_available.shape != (n,) or last_age.shape != (n,):
            raise ValueError('Aligned current/history relation context required')
        if not n or not m:
            return predicted.new_zeros(n, m)
        query = F.normalize(predicted, dim=-1)
        values = F.normalize(candidates['tokens'], dim=-1)
        similarities = torch.einsum('nkd,mjd->nmkj', query, values)
        query_valid = predicted.square().sum(-1) > 0
        query_safe = query_valid.clone()
        query_safe[~query_safe.any(1), 0] = True
        legal = candidates['valid'].any(1)
        safe = candidates['valid'].clone()
        safe[~safe.any(1), 0] = True
        row = (similarities/.1).masked_fill(~safe[None, :, None], -torch.inf).softmax(-1)
        column = (similarities/.1).masked_fill(~query_safe[:, None, :, None], -torch.inf).softmax(-2)
        row_mean = ((row*similarities).sum(-1)*query_valid[:, None]).sum(-1)/query_valid.sum(1).clamp_min(1)[:, None]
        column_mean = ((column*similarities).sum(-2)*candidates['weights'][None]).sum(-1)
        base = (row_mean+column_mean)/2
        diagonal_values = similarities.diagonal(dim1=-2, dim2=-1)
        diagonal_valid = query_valid[:, None, :min(k, c)] & candidates['valid'][None, :, :min(k, c)]
        diagonal = (diagonal_values*diagonal_valid).sum(-1)/diagonal_valid.sum(-1).clamp_min(1)
        pair_valid = query_valid[:, None, :, None] & candidates['valid'][None, :, None]
        mass = pair_valid.sum((-1, -2)).clamp_min(1)
        mean = (similarities*pair_valid).sum((-1, -2))/mass
        spread = (((similarities-mean[..., None, None]).square()*pair_valid).sum((-1, -2))/mass+1.e-12).sqrt()
        features = torch.stack([base, diagonal, spread, last_age[:, None].expand(n, m),
            candidates['valid'].float().mean(1)[None].expand(n, m)], -1)
        correction = .5*self.relation(features).squeeze(-1).tanh()
        scores = (self.logit_scale.clamp(math.log(1.), math.log(50.)).exp()*base+correction)*history_available[:, None]
        return scores.masked_fill(~legal[None], -torch.inf)

    def identity_logits(self, features):
        return 10*F.linear(F.normalize(features, dim=-1), F.normalize(self.classifier.weight, dim=-1))

    def forward(self, history, history_fraction, history_global, times, history_valid,
                frame, candidates, candidate_fraction, candidate_global):
        context = self.history_context(history, history_fraction, history_global, times, history_valid, frame)
        predicted = self.predict(context)
        current = self.current(candidates, candidate_fraction, candidate_global)
        age = torch.where(context['available'], context['age'][:, -1], torch.zeros_like(context['age'][:, -1]))
        scores = self.relation_scores(predicted, current, context['available'], age)
        return dict(scores=scores, predicted=predicted, context=context, current=current,
                    history_available=context['available'])


class DenseLocalTarget(nn.Module):
    """Independent EMA local encoder; its observations never enter inference."""
    def __init__(self, student, decay=.99):
        super().__init__()
        if not 0 <= decay < 1:
            raise ValueError('EMA decay in[0,1) required')
        self.encoder = copy.deepcopy(student.local_encoder).requires_grad_(False).eval()
        self.decay = decay
        self.register_buffer('updates', torch.zeros((), dtype=torch.int64))
        if any(a.data_ptr() == b.data_ptr() for a, b in zip(self.encoder.parameters(), student.local_encoder.parameters())):
            raise ValueError('Target must not share mutable student parameters')

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, clean_observations, observed_fraction):
        result = self.encoder(clean_observations.detach(), observed_fraction.detach())
        return {name: value.detach() for name, value in result.items()}

    @torch.no_grad()
    def update(self, student, successful=True):
        if not successful:
            return
        for target, source in zip(self.encoder.parameters(), student.local_encoder.parameters()):
            target.lerp_(source.detach(), 1-self.decay)
        for target, source in zip(self.encoder.buffers(), student.local_encoder.buffers()):
            target.copy_(source)
        self.updates.add_(1)
        if any(p.grad is not None or p.requires_grad for p in self.parameters()):
            raise RuntimeError('EMA target must remain stopped')


def known_candidate_loss(scores, track_ids, candidate_ids, history_available):
    """Unknown columns never become implicit negatives in a softmax loss."""
    if track_ids.shape != (len(scores),) or candidate_ids.shape != (scores.shape[1],):
        raise ValueError('Physical offline relation labels must align')
    known = (candidate_ids >= 0)[None] & torch.isfinite(scores)
    positives = (track_ids[:, None] == candidate_ids[None]) & (track_ids[:, None] >= 0) & known
    valid = positives.any(1) & known.any(1) & history_available
    if not valid.any():
        finite_scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
        return finite_scores.sum()*0, 0
    selected = scores[valid]
    loss = torch.logsumexp(selected.masked_fill(~known.expand_as(scores)[valid], -torch.inf), -1)
    loss = loss-torch.logsumexp(selected.masked_fill(~positives[valid], -torch.inf), -1)
    return loss.mean(), int(valid.sum())


def dense_prediction_loss(predicted, target, target_mask, history_available):
    if predicted.shape != target['tokens'].shape or target_mask.shape != predicted.shape[:2] or target_mask.dtype != torch.bool:
        raise ValueError('Masked current/future target queries and local latent shapes required')
    selected = target_mask & target['valid'] & history_available[:, None]
    if not selected.any():
        return predicted.sum()*0, 0
    cosine = (F.normalize(predicted, dim=-1)*F.normalize(target['tokens'].detach(), dim=-1)).sum(-1)
    return (1-cosine[selected]).mean(), int(selected.sum())
