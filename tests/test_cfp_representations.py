import numpy as np
import pytest
import torch

from cfp.representations import identity_view
from cfp.predictive import LatentIdentityPredictor


def test_paired_view_preserves_branch_order_equal_capacity_and_zero_fallback():
    rgb = np.zeros((3, 512), np.float32)
    ir = rgb.copy()
    rgb[0, 0] = ir[0, 1] = 1.
    rgb[1, 2] = 1.
    paired = identity_view(rgb, ir, 'rgb_ir')
    two_rgb = identity_view(rgb, ir, 'two_rgb', ir)
    assert np.array_equal(paired, two_rgb)
    assert np.allclose(paired[0, [0, 513]], 1 / np.sqrt(2))
    assert np.isclose(np.linalg.norm(paired[1]), 1) and not paired[2].any()
    assert identity_view(rgb, ir, 'rgb') is rgb
    assert identity_view(rgb, ir, 'ir') is ir
    assert LatentIdentityPredictor(1024, 'gru').input.in_features == 1026
    with pytest.raises(ValueError, match='Aligned frozen'):
        identity_view(rgb, ir, 'two_rgb')
    with pytest.raises(ValueError, match='Undeclared'):
        identity_view(rgb, ir, 'future_view')


def test_paired_prediction_scores_only_strictly_past_tokens():
    model = LatentIdentityPredictor(1024, 'gru')
    history = torch.randn(1, 2, 1024)
    frames = torch.tensor([[1, 3]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    output, available = model(history, frames, valid, torch.tensor([4]))
    assert available.all() and output.shape == (1, 1024) and torch.isfinite(output).all()
    with pytest.raises(ValueError, match='strictly before'):
        model(history, frames, valid, torch.tensor([3]))
