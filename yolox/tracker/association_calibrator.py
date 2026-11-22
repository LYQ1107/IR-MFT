"""A bounded, signed cue contrast over fixed RGB point log-odds."""
import torch
from torch import nn


class AssociationCalibrator(nn.Module):
    def __init__(self,base_dimensions=23,cue_dimensions=17,hidden=64,logit_bound=4.):
        super().__init__();self.base_dimensions=base_dimensions;self.cue_dimensions=cue_dimensions
        self.logit_bound=float(logit_bound);dimension=base_dimensions+cue_dimensions
        self.net=nn.Sequential(nn.Linear(dimension,hidden),nn.LayerNorm(hidden),nn.GELU(),
                               nn.Linear(hidden,hidden//2),nn.GELU(),nn.Linear(hidden//2,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('mean',torch.zeros(dimension));self.register_buffer('std',torch.ones(dimension))

    def increment(self,base,cue,reference,available):
        first=(torch.cat([base,cue],dim=-1)-self.mean)/self.std
        second=(torch.cat([base,reference],dim=-1)-self.mean)/self.std
        contrast=(self.net(first)-self.net(second)).squeeze(-1)
        bounded=self.logit_bound*torch.tanh(contrast/self.logit_bound)
        return torch.where(available.bool(),bounded,torch.zeros_like(bounded))

    @staticmethod
    def RGB_offset(base):
        probability=(base[...,14]/.8).clamp(.01,.99)
        return torch.logit(probability)

    def forward(self,base,cue,reference,available):
        return self.RGB_offset(base)+self.increment(base,cue,reference,available)
