"""Small bounded correction to an explicit frozen ordinary association prior."""
import torch
from torch import nn
from torch.nn import functional as F

from .candidate_relation_jepa import CandidateRelationStudent


def bounded_candidate_delta(logits, legal, scale=.05):
    mean = (logits * legal).sum(-1, keepdim=True) / legal.sum(-1, keepdim=True).clamp_min(1)
    return scale * (logits - mean).tanh() * legal


def residual_scores(prior, delta, appearance_eligible, legal, threshold=.97):
    cost = prior - delta * appearance_eligible
    return torch.cat([(-cost / .1).masked_fill(~legal, -torch.inf),
                      cost.new_full((len(cost), 1), -threshold / .1)], dim=-1)


def centered_prediction_loss(prediction, target, legal, target_valid):
    denominator = legal.sum(1, keepdim=True).clamp_min(1)[..., None]
    p = prediction - (prediction * legal[..., None]).sum(1, keepdim=True) / denominator
    t = target.detach() - (target.detach() * legal[..., None]).sum(1, keepdim=True) / denominator
    selected = legal & target_valid[:, None] & (legal.sum(1) > 1)[:, None]
    if not selected.any():
        return prediction.sum() * 0
    return (1 - (F.normalize(p, dim=-1) * F.normalize(t, dim=-1)).sum(-1))[selected].mean()


class ResidualRelationStudent(CandidateRelationStudent):
    def __init__(self, latent=16, scale=.05):
        super().__init__(latent)
        self.residual_head = nn.Linear(latent, 1)
        nn.init.zeros_(self.residual_head.weight); nn.init.zeros_(self.residual_head.bias)
        self.scale = scale

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        legal = args[5] if len(args) > 5 else kwargs['candidate_valid']
        logits = self.residual_head(output['predicted_latent']).squeeze(-1)
        output['candidate_delta'] = bounded_candidate_delta(logits, legal, self.scale)
        return output
