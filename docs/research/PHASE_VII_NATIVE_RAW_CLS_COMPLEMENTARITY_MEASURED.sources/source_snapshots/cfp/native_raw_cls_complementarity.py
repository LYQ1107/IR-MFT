"""Pure helpers for a source-matched raw-CLS complementarity audit.

The audit compares two causal appearance coordinates on the exact same
candidate graph.  These helpers deliberately contain no labels, tracker state,
or threshold search; physical labels are consumed only by the retrospective
reducer after all prediction hashes are locked.
"""

from __future__ import annotations

import numpy as np


def top1_positive(cost, legal, positive):
    """Return the rank of the first positive and whether top-1 is positive."""
    cost = np.asarray(cost, dtype=np.float64)
    legal = np.asarray(legal, dtype=bool)
    positive = np.asarray(positive, dtype=bool)
    if cost.ndim != 1 or legal.shape != cost.shape or positive.shape != cost.shape:
        raise ValueError("cost, legal and positive must be aligned vectors")
    values = np.where(legal, cost, np.inf)
    order = np.argsort(values, kind="stable")
    order = order[np.isfinite(values[order])]
    if not len(order):
        return None, False
    positive_positions = np.flatnonzero(positive[order])
    return (int(positive_positions[0]) + 1 if len(positive_positions) else None,
            bool(positive[order[0]]))


def top_two_margin(cost, legal):
    """Return the fixed top-two margin, or ``None`` for fewer than two edges."""
    cost = np.asarray(cost, dtype=np.float64)
    legal = np.asarray(legal, dtype=bool)
    if cost.ndim != 1 or legal.shape != cost.shape:
        raise ValueError("cost and legal must be aligned vectors")
    values = np.sort(cost[legal])
    return float(values[1] - values[0]) if len(values) >= 2 else None


def margin_gate(embedding_cost, raw_cls_cost, legal):
    """Use raw CLS only when its fixed margin exceeds the embedding margin.

    This is a pre-registered deterministic rule.  It has no fitted threshold
    and returns only the decision mask; the caller still owns the unchanged
    geometry/lifecycle graph.
    """
    embedding_margin = top_two_margin(embedding_cost, legal)
    raw_margin = top_two_margin(raw_cls_cost, legal)
    return (embedding_margin is not None and raw_margin is not None
            and raw_margin > embedding_margin)
