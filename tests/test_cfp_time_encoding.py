import numpy as np
import pytest
import torch

from cfp.predictive import checked_history, LatentIdentityPredictor


def test_log_time_encoding_distinguishes_old_anchors_without_more_parameters():
    features = torch.randn(1, 3, 512)
    frames = torch.tensor([[1, 900, 999]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    query = torch.tensor([1000])
    old = checked_history(features, frames, valid, query)
    new = checked_history(features, frames, valid, query, 'log1p_15000')
    assert old[1][0, 0, 0] == old[1][0, 1, 0] == 2
    assert new[1][0, 0, 0] > new[1][0, 1, 0] > new[1][0, 2, 0]
    assert torch.equal(old[0], new[0]) and torch.equal(old[-1], new[-1])
    for architecture in ['gru', 'transformer']:
        plain = LatentIdentityPredictor(512, architecture)
        log = LatentIdentityPredictor(512, architecture, time_encoding='log1p_15000')
        assert sum(p.numel() for p in plain.parameters()) == sum(p.numel() for p in log.parameters())
        log.load_state_dict(plain.state_dict(), strict=True)


def test_log_padding_stays_finite_and_preserves_strict_past_guard():
    features = torch.randn(2, 3, 512)
    frames = torch.tensor([[10, 20, 0], [0, 0, 0]])
    valid = torch.tensor([[True, True, False], [False, False, False]])
    query = torch.tensor([30, 30])
    result = checked_history(features, frames, valid, query, 'log1p_15000')
    assert torch.isfinite(result[1]).all() and not result[1][1].any() and not result[1][0, 2].any()
    assert result[3].tolist() == [True, False]
    with pytest.raises(ValueError, match='strictly before'):
        checked_history(features, frames, valid, torch.tensor([20, 30]), 'log1p_15000')
    with pytest.raises(ValueError, match='Unknown declared'):
        checked_history(features, frames, valid, query, 'unfrozen')


def test_default_time_encoding_retains_original_clipped_formula_exactly():
    features = torch.randn(1, 4, 512)
    frames = torch.tensor([[1, 5, 100, 101]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    query = torch.tensor([102])
    times = checked_history(features, frames, valid, query)[1]
    expected_age = ((query[:, None] - frames).float() / 37).clamp(0, 2) * valid
    expected_gap = torch.zeros_like(expected_age)
    expected_gap[:, 1:] = ((frames[:, 1:] - frames[:, :-1]).float() / 37).clamp(0, 2) * valid[:, 1:]
    assert torch.equal(times, torch.stack([expected_age, expected_gap], -1))
