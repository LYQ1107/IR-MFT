import numpy as np
import pytest
import torch

from cfp.utility import candidate_ce, counterfactual_utility, utility_mix, decision_evidence, UtilityMLP, EVIDENCE_FIELDS


def test_counterfactual_target_uses_same_candidates_and_explicit_null():
    rgb = np.array([.2, .1], np.float32)
    ir = np.array([.0, .3], np.float32)
    assert counterfactual_utility(rgb, ir, 0) > 0
    assert counterfactual_utility(rgb, ir, 1) < 0
    assert np.isfinite(candidate_ce(rgb, 2))
    assert candidate_ce([], 0) == 0
    assert counterfactual_utility(rgb, None, 1) == 0
    assert counterfactual_utility(rgb, rgb.copy(), 1) == 0
    with pytest.raises(ValueError):
        counterfactual_utility(rgb, ir[:1], 0)
    with pytest.raises(ValueError):
        candidate_ce(rgb, 3)


def test_rejected_or_missing_ir_returns_exact_rgb_and_accepted_weight_is_capped():
    rgb = np.array([.12345678, .9876543], np.float32)
    for ir, utility in [(None, 100.), (np.zeros(2), -.1), (np.zeros(2), .02)]:
        result, weight = utility_mix(rgb, ir, utility, threshold=.02)
        assert result is rgb and weight == 0 and np.array_equal(result, rgb)
    result, weight = utility_mix(rgb, np.zeros(2), 100)
    assert weight == .5 and np.array_equal(result, .5 * rgb)
    with pytest.raises(ValueError):
        utility_mix(rgb, np.array([np.nan, 0]), 1)


def test_evidence_permutation_invariance_and_quality_gate_cannot_read_branch_scores():
    rgb, ir = np.array([.1, .2]), np.array([.2, .1])
    geometry, confidence = np.array([.2, .5]), np.array([.9, .8])
    history = np.eye(2, dtype=np.float32)
    args = (np.ones(2, bool), 100, 1, history)
    first = decision_evidence(rgb, ir, geometry, confidence, *args)
    second = decision_evidence(rgb[::-1], ir[::-1], geometry[::-1], confidence[::-1], *args)
    assert first.shape == (len(EVIDENCE_FIELDS),) and np.allclose(first, second)
    model = UtilityMLP(quality_only=True).eval()
    altered = first.copy()
    for name in ['rgb_min','ir_margin_with_null','branch_best_disagreement']:
        altered[EVIDENCE_FIELDS.index(name)] += 10
    with torch.no_grad():
        assert torch.equal(model(torch.from_numpy(first)), model(torch.from_numpy(altered)))
