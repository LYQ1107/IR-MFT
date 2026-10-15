import numpy as np

from cfp.native_unknown_null import apply_unknown_preserving_null


def test_fixed_ambiguity_rule_abstains_without_changing_ineligible_edges():
    baseline = np.array([[0.20, 0.24, 0.91], [0.10, 0.13, 0.80]], dtype=np.float64)
    raw = np.array([[0.20, 0.30, 0.95], [0.20, 0.30, 0.91]], dtype=np.float64)
    edited, counts = apply_unknown_preserving_null(
        baseline, raw, null_cost=0.97, ambiguity_margin=0.05, legal_geometry_max=0.9)
    assert counts["ambiguous_rows"] == 2
    assert counts["abstained_rows"] == 2
    assert edited[0, 0] > 0.97 and edited[0, 1] > 0.97
    assert edited[0, 2] == baseline[0, 2]
    assert edited[1, 2] == baseline[1, 2]


def test_fixed_rule_keeps_clear_winner_and_empty_rows():
    baseline = np.array([[0.10, 0.30], [0.10, np.nan]], dtype=np.float64)
    raw = np.array([[0.20, 0.30], [0.95, 0.95]], dtype=np.float64)
    edited, counts = apply_unknown_preserving_null(
        baseline, raw, null_cost=0.97, ambiguity_margin=0.05, legal_geometry_max=0.9)
    assert counts["abstained_rows"] == 0
    assert np.array_equal(edited[0], baseline[0])
    assert np.isnan(edited[1, 1])
