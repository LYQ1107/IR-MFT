"""Complete the S/C/R component study, preserving all previously tested pairs."""
import torch
from .structural_relative_repair import StructuralRepair

FLAGS = {f'{s}{c}{r}': dict(structure=bool(s), centered=bool(c), feature_risk=bool(r))
         for s in [0,1] for c in [0,1] for r in [0,1]}


class StructuralFactorial(StructuralRepair):
    def __init__(self, *args, components='111', **kwargs):
        if components not in FLAGS or 'variant' in kwargs:
            raise ValueError('Use registered S/C/R bits; hard cycle is always disabled')
        super().__init__(*args, variant='no_cycle', **kwargs)
        self.components = components
        self.centered = FLAGS[components]['centered']
        self.feature_risk = FLAGS[components]['feature_risk']
        for level in self.levels:
            level.use_structure = FLAGS[components]['structure']
            level.cycle = False

    def forward(self, rgb, auxiliary, teacher_scores, query, key, query_valid, key_valid,
                availability=True, null=False):
        scores = teacher_scores if self.feature_risk else [torch.zeros_like(v) for v in teacher_scores]
        return super().forward(rgb,auxiliary,scores,query,key,query_valid,key_valid,availability,null)
