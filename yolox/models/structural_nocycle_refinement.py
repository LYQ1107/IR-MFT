"""Separate the surviving structure/centering/risk hypotheses, always seed0."""
import torch

from .structural_relative_repair import StructuralRepair

VARIANTS = {
    'no_cycle': dict(centered=True, disable_feature_risk=False),
    'no_cycle_no_centering': dict(centered=False, disable_feature_risk=False),
    'no_cycle_no_risk': dict(centered=True, disable_feature_risk=True),
}


class NoCycleRefinement(StructuralRepair):
    def __init__(self, *args, refinement='no_cycle', **kwargs):
        if refinement not in VARIANTS or 'variant' in kwargs:
            raise ValueError('Use the registered independent refinement flags')
        super().__init__(*args, variant='no_cycle', **kwargs)
        self.refinement = refinement
        self.centered = VARIANTS[refinement]['centered']
        self.disable_feature_risk = VARIANTS[refinement]['disable_feature_risk']

    def forward(self, rgb, auxiliary, teacher_scores, query, key, query_valid, key_valid,
                availability=True, null=False):
        # The risk removal changes only the teacher-confidence factor in the
        # feature update. Structural trust, tanh/RMS bound and final semantic
        # bound remain; original reference logits/scores are never modified.
        scores = [torch.zeros_like(value) for value in teacher_scores] if self.disable_feature_risk else teacher_scores
        return super().forward(rgb, auxiliary, scores, query, key, query_valid, key_valid, availability, null)
