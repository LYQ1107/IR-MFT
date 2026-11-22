"""Inference-only safeguards around the same immutable fitted RGB reference."""
import torch
from torch import nn

from yolox.tracker.strong_rgb_tracker import StrongRGBForegroundTracker


class ResidualGuard(nn.Module):
    def __init__(self,model,scale=1.,uncertain_only=False):
        super().__init__();self.model=model;self.scale=float(scale);self.uncertain_only=bool(uncertain_only)
        self.base_dimensions=model.base_dimensions
        if not 0<=self.scale<=1:raise ValueError('Only residual shrinkage is permitted')

    def increment(self,base,cue,reference,available):
        if self.scale==0:return torch.zeros_like(base[...,0])
        delta=self.model.increment(base,cue,reference,available)*self.scale
        if self.uncertain_only:
            probability,difference,margin=base[...,23],base[...,24],base[...,25]
            row_maximum=torch.where(difference>=0,probability,probability-difference)
            uncertain=(row_maximum<.8)|(margin<.15)
            delta=torch.where(uncertain,delta,torch.zeros_like(delta))
        return delta


class GuardedStrongRGBTracker(StrongRGBForegroundTracker):
    def __init__(self,config,**kwargs):
        super().__init__(config,**kwargs)
        self.model=ResidualGuard(self.model,config.get('new_residual_scale',1.),config.get('new_residual_uncertain_only',False)).eval()
        self.calibration_metadata=dict(self.calibration_metadata,residual_guard=dict(scale=self.model.scale,uncertain_only=self.model.uncertain_only,scope='Newly added cue increment only; fitted RGB model remains intact'))
