import numpy as np
import pytest

from cfp.native_raw_cls_complementarity import margin_gate, top1_positive, top_two_margin


def test_top1_positive_uses_stable_legal_order():
    rank, is_top = top1_positive([0.2, 0.2, 0.1], [True, True, False], [False, True, True])
    assert rank == 2
    assert not is_top


def test_top_two_margin_ignores_illegal_edges():
    assert top_two_margin(np.array([0.1, 0.9, 0.3]), np.array([True, False, True])) == pytest.approx(0.2)
    assert top_two_margin([0.1], [True]) is None


def test_margin_gate_is_strict_and_does_not_search_thresholds():
    assert margin_gate([0.10, 0.20], [0.10, 0.31], [True, True])
    assert not margin_gate([0.10, 0.20], [0.10, 0.20], [True, True])


def test_alignment_is_required():
    with pytest.raises(ValueError):
        top1_positive([0.1], [True, False], [True])
