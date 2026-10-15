"""Deterministic robust identity-memory control for a registered native probe."""

import torch
from torch import nn
from torch.nn import functional as F

from .predictive import checked_history


class MedoidIdentityControl(nn.Module):
    """Select the observed history embedding with minimum total cosine distance.

    The rule is fixed before evaluation: it has no learned parameter, no
    threshold and no access to current candidates, labels or future frames.
    """

    def __init__(self, length=4, maximum_age=1500):
        super().__init__()
        if not 1 <= length <= 8:
            raise ValueError("Registered medoid history length must be in [1, 8]")
        if maximum_age <= 0:
            raise ValueError("Positive native medoid maximum age required")
        self.length, self.maximum_age = int(length), int(maximum_age)

    def forward(self, features, times, valid, frame):
        source, _, _, available, _ = checked_history(features, times, valid, frame)
        output = source.new_zeros(len(source), source.shape[-1])
        observed = torch.zeros_like(available)
        for row in range(len(source)):
            indices = torch.nonzero(
                valid[row] & (times[row] >= frame[row] - self.maximum_age),
                as_tuple=False,
            ).flatten()[-self.length :]
            if not len(indices):
                continue
            values = F.normalize(source[row, indices], dim=-1)
            distances = 1.0 - values @ values.transpose(0, 1)
            medoid = torch.argmin(distances.sum(dim=1))
            output[row] = values[medoid]
            observed[row] = True
        return output, observed
