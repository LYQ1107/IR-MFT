import pytest

from cfp.selective_metrics import risk_metrics


def test_unknown_edges_are_excluded_and_auc_ties_are_fair():
    m = risk_metrics([.1, .1, .9, .9, .01], [0, 1, 0, 1, -1])
    assert m['known'] == 4 and m['unknown_excluded'] == 1
    assert m['auc'] == .5 and m['wrong'] == 2
    assert m['risk_coverage'][-1]['risk'] == .5


def test_perfect_risk_and_nonfinite_rejection():
    m = risk_metrics([0, 0, 1, 1], [0, 0, 1, 1])
    assert m['auc'] == 1 and m['ece'] == 0 and m['brier'] == 0
    with pytest.raises(ValueError):
        risk_metrics([float('nan')], [0])


def test_single_class_is_not_fake_auc():
    assert risk_metrics([.2], [0])['auc'] is None
    assert risk_metrics([.9], [-1])['known'] == 0

