"""Observable candidate-conditioned risk features and small selective estimator.

Offline labels live in training tools, never here. NULL expresses abstention; it
does not claim a physically absent fish. Stable memory writes are a separate action.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .candidate_graph import CandidateDecision


FEATURE_NAMES = ('prototype_similarity', 'mean_history_similarity', 'last_similarity',
    'best_history_similarity', 'worst_history_similarity', 'history_consistency',
    'anchor_last_conflict', 'similarity_margin', 'candidate_entropy', 'iou',
    'motion_dx', 'motion_dy', 'log_width_ratio', 'log_height_ratio', 'confidence',
    'age', 'gap', 'stage', 'crowding', 'baseline_cost', 'competitor_margin')


def observable_candidate_features(graph):
    graph.validate()
    past = F.normalize(graph.history, dim=-1)
    candidate = F.normalize(graph.candidates, dim=-1)
    n, t, d = past.shape
    m = len(candidate)
    if n == 0 or m == 0:
        return past.new_zeros(n, m, len(FEATURE_NAMES))
    valid = graph.history_valid
    length = valid.sum(1)
    mean = F.normalize((past * valid[..., None]).sum(1) / length.clamp_min(1)[:, None], dim=-1)
    last = past[torch.arange(n, device=past.device), (length - 1).clamp_min(0)] if t else mean
    anchor = past[:, 0] if t else mean
    prototype = F.normalize(graph.prototype, dim=-1) if graph.prototype is not None else mean
    similarity = prototype @ candidate.T
    history_similarity = torch.einsum('ntd,md->ntm', past, candidate)
    if t:
        best = history_similarity.masked_fill(~valid[..., None], -torch.inf).amax(1)
        worst = history_similarity.masked_fill(~valid[..., None], torch.inf).amin(1)
        best = torch.where(length[:, None] > 0, best, torch.zeros_like(best))
        worst = torch.where(length[:, None] > 0, worst, torch.zeros_like(worst))
    else:
        best = worst = similarity.new_zeros(n, m)
    consistency = ((past * mean[:, None]).sum(-1) * valid).sum(1) / length.clamp_min(1)
    masked = similarity.masked_fill(~graph.legal, -1e4)
    probs = torch.softmax(masked / .1, dim=1) * graph.legal
    probs = probs / probs.sum(1, keepdim=True).clamp_min(1e-8)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(1) / max(float(torch.log(torch.tensor(max(m, 2)))), 1e-8)
    if m > 1:
        # Candidate-vs-best-other margin for every edge, not one row-global score.
        top = masked.topk(2, dim=1)
        other = top.values[:, :1].expand(n, m).clone()
        other.scatter_(1, top.indices[:, :1], top.values[:, 1:2])
        margin = (similarity - other).clamp(-2, 2)
    else:
        margin = similarity.new_zeros(n, m)
    boxes, predictions = graph.candidate_boxes, graph.track_boxes
    bsize = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-6)
    psize = (predictions[:, 2:] - predictions[:, :2]).clamp_min(1e-6)
    residual = ((boxes[:, 2:] + boxes[:, :2])[None] - (predictions[:, 2:] + predictions[:, :2])[:, None]) / (2 * psize[:, None])
    ratio = (bsize[None] / psize[:, None]).log().clamp(-4, 4)
    intersection = (torch.minimum(predictions[:, None, 2:], boxes[None, :, 2:]) - torch.maximum(predictions[:, None, :2], boxes[None, :, :2])).clamp_min(0).prod(-1)
    iou = intersection / (psize.prod(1)[:, None] + bsize.prod(1)[None] - intersection).clamp_min(1e-8)
    overlaps = (torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:]) - torch.maximum(boxes[:, None, :2], boxes[None, :, :2])).clamp_min(0).prod(-1)
    overlaps = overlaps / (bsize.prod(1)[:, None] + bsize.prod(1)[None] - overlaps).clamp_min(1e-8)
    overlaps.fill_diagonal_(0)
    crowd = overlaps.amax(1)
    if n > 1:
        competing = similarity.masked_fill(~graph.legal, -1e4).topk(2, dim=0)
        other_track = competing.values[:1].expand(n, m).clone()
        other_track.scatter_(0, competing.indices[:1], competing.values[1:2])
        competitor_margin = (similarity - other_track).clamp(-2, 2)
    else:
        competitor_margin = similarity.new_zeros(n, m)
    expand = lambda value: value[:, None].expand(n, m)
    channels = [similarity, mean @ candidate.T, last @ candidate.T, best, worst,
        expand(consistency), expand(1 - (anchor * last).sum(1)), margin, expand(entropy),
        iou, residual[..., 0].clamp(-4, 4), residual[..., 1].clamp(-4, 4), ratio[..., 0],
        ratio[..., 1], graph.confidence[None].expand(n, m),
        expand((graph.age / 37).clamp(0, 4)), expand((graph.gap / 37).clamp(0, 4)),
        similarity.new_full((n, m), graph.stage / 2), crowd[None].expand(n, m),
        graph.baseline_cost, competitor_margin]
    result = torch.stack(channels, dim=-1)
    if not torch.isfinite(result).all():
        raise ValueError('Observable risk feature extraction must be finite')
    return result


class RecoverabilityEstimator(nn.Module):
    """Small edge-risk MLP plus optional candidate-set pooling, no GT input."""
    def __init__(self, hidden=32, candidate_context=True, history_evidence=True):
        super().__init__()
        self.candidate_context = candidate_context
        self.history_evidence = history_evidence
        size = len(FEATURE_NAMES)
        self.edge = nn.Sequential(nn.Linear(size, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.risk = nn.Linear(hidden * (2 if candidate_context else 1), 1)

    def forward(self, features, legal):
        if features.shape[:-1] != legal.shape or features.shape[-1] != len(FEATURE_NAMES):
            raise ValueError('Risk input shape differs')
        if not self.history_evidence:
            features = features.clone()
            features[..., :7] = 0
        encoded = self.edge(features)
        if self.candidate_context:
            context = (encoded * legal[..., None]).sum(-2, keepdim=True) / legal.sum(-1, keepdim=True).clamp_min(1)[..., None]
            encoded = torch.cat([encoded, context.expand_as(encoded)], dim=-1)
        return self.risk(encoded).squeeze(-1)

    def decide(self, graph, temperature=1., null_logit=0.):
        if temperature <= 0:
            raise ValueError('Calibration temperature must be positive')
        risk_logits = self(observable_candidate_features(graph), graph.legal) / temperature
        scores = (-risk_logits).masked_fill(~graph.legal, -torch.inf)
        null = scores.new_full((scores.shape[0], 1), float(null_logit))
        return CandidateDecision(torch.cat([scores, null], dim=1), risk_logits.sigmoid(), graph.legal).validate()

