"""Matched risk controls without indirect identity-history leakage."""
import torch
from torch import nn

from .recoverability import RecoverabilityEstimator


class NoIdentityHistoryRisk(RecoverabilityEstimator):
    def forward(self, features, legal):
        features = features.clone()
        features[..., :9] = 0  # Prototype/history, derived margins and entropy.
        features[..., 20] = 0  # Other tracks' identity prototype margin.
        # Original fused baseline cost includes history appearance; keep only
        # the observable current motion/geometry/confidence cost in this control.
        features[..., 19] = 1 - features[..., 9] * features[..., 14]
        return super().forward(features, legal)


class LearnedCostThreshold(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, features, legal):
        if features.shape[:-1] != legal.shape:
            raise ValueError('Aligned same-input risk edge shapes required')
        return self.weight * features[..., 19] + self.bias


def risk_model_from_config(config):
    family = config.get('model_family', 'context')
    if family == 'no_identity_history':
        return NoIdentityHistoryRisk(config['hidden'], config['candidate_context'], True)
    if family == 'learned_threshold':
        return LearnedCostThreshold()
    if family in ('context', 'untrained_context'):
        return RecoverabilityEstimator(config['hidden'], config['candidate_context'], config['history_evidence'])
    raise ValueError('Declared risk-control family required')

