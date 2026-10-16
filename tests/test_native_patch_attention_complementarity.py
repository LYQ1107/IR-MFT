import numpy as np
import pytest

from cfp.native_patch_attention_complementarity import fifo_cosine_costs


def test_fifo_patch_cost_is_zero_for_identical_unit_features():
    history = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    candidates = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    costs, available = fifo_cosine_costs(history, candidates)
    assert available.tolist() == [True, True]
    assert costs.tolist() == pytest.approx([0.0, 0.5])


def test_fifo_patch_normalizes_each_history_observation():
    history = np.array([[10.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    candidates = np.array([[1.0, 1.0]], dtype=np.float32)
    costs, available = fifo_cosine_costs(history, candidates)
    assert available.tolist() == [True]
    assert costs[0] == pytest.approx(0.0, abs=1e-6)


def test_empty_history_disables_patch_cue():
    costs, available = fifo_cosine_costs(
        np.empty((0, 3), dtype=np.float32), np.ones((2, 3), dtype=np.float32))
    assert np.all(costs == 1.0)
    assert not available.any()


def test_patch_feature_shape_is_checked():
    with pytest.raises(ValueError):
        fifo_cosine_costs(np.ones((1, 2)), np.ones((1, 3)))
