"""Untrained controls for residual identifiability, not new identity models."""
import torch
from torch import nn

from .predictive import checked_history


class UniformPositiveResidual(nn.Module):
    """Use the preregistered +0.03 bound on every observable eligible edge.

    This identifies a common cost/acceptance-threshold effect. There is no set
    attention, trained weight, ID evidence decision or NULL classifier here.
    """
    def __init__(self, dimension=512, capacity=128):
        super().__init__()
        self.dimension, self.capacity = dimension, capacity
        self.device_marker = nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, history, times, valid, frame, detections, detection_valid, **observable):
        _, _, _, available, _ = checked_history(history, times, valid, frame, 'log1p_15000')
        if history.shape[2] != self.dimension or detections.shape[1] != self.dimension or len(history) > self.capacity:
            raise ValueError('Matched feature coordinates and full-set capacity required')
        legal = available[:, None] & detection_valid[None]
        # tanh(100)==1 at FP32; this is the existing fixed declared bound,
        # selected from fit/calibration residual saturation, not an outer sweep.
        return dict(residual=torch.full(legal.shape, 100., device=history.device), legal=legal)
