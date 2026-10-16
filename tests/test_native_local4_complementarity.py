import numpy as np
import pytest

from cfp.native_local4_complementarity import visibility_weighted_fifo_costs


def test_visibility_weighted_fifo_is_zero_for_identical_visible_tokens():
    history = np.zeros((2, 4, 3), dtype=np.float32)
    history[:, :, 0] = 1.0
    candidate = history[:1].copy()
    visibility = np.ones((2, 4), dtype=np.float32)
    costs, available = visibility_weighted_fifo_costs(
        history, visibility, candidate, visibility[:1])
    assert available.tolist() == [True]
    assert costs.tolist() == pytest.approx([0.0])


def test_zero_visibility_is_not_local_support():
    history = np.ones((1, 4, 2), dtype=np.float32)
    candidate = np.ones((2, 4, 2), dtype=np.float32)
    history_visibility = np.zeros((1, 4), dtype=np.float32)
    candidate_visibility = np.ones((2, 4), dtype=np.float32)
    costs, available = visibility_weighted_fifo_costs(
        history, history_visibility, candidate, candidate_visibility)
    assert not available.any()
    assert np.all(costs == 1.0)


def test_local_costs_are_visibility_weighted_and_bounded():
    history = np.zeros((1, 4, 2), dtype=np.float32)
    history[:, 0, 0] = 1.0
    history[:, 1, 0] = 1.0
    candidate = history.copy()
    candidate[0, 0] = [0.0, 1.0]
    candidate[0, 1] = [1.0, 0.0]
    history_visibility = np.array([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    candidate_visibility = np.ones((1, 4), dtype=np.float32)
    costs, available = visibility_weighted_fifo_costs(
        history, history_visibility, candidate, candidate_visibility)
    assert available.tolist() == [True]
    assert costs[0] == pytest.approx(0.25)
    assert 0.0 <= costs[0] <= 1.0


def test_shape_and_visibility_validation():
    with pytest.raises(ValueError):
        visibility_weighted_fifo_costs(
            np.ones((1, 4, 2), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
            np.ones((1, 4, 2), dtype=np.float32),
            np.ones((1, 4), dtype=np.float32))
