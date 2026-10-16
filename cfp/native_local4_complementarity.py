"""Pure helpers for a fixed native local-token complementarity audit.

The local descriptor is deliberately small and deterministic: each of the four
geometric DINO strips is normalized, averaged over the last four strictly-past
observations with its observable-content fraction as weight, and compared with
the current strip using the same visibility weights.  There is no identity
label, fitting, threshold search, or tracker state in this module.
"""

from __future__ import annotations

import numpy as np


def top1_positive(cost, legal, positive):
    """Return the first-positive rank and whether legal top-1 is positive."""
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
    positions = np.flatnonzero(positive[order])
    return (int(positions[0]) + 1 if len(positions) else None,
            bool(positive[order[0]]))


def top_two_margin(cost, legal):
    """Return the strict top-two cost margin for legal edges."""
    cost = np.asarray(cost, dtype=np.float64)
    legal = np.asarray(legal, dtype=bool)
    if cost.ndim != 1 or legal.shape != cost.shape:
        raise ValueError("cost and legal must be aligned vectors")
    values = np.sort(cost[legal])
    return float(values[1] - values[0]) if len(values) >= 2 else None


def margin_gate(embedding_cost, local_cost, legal):
    """Use local4 only when its top-two margin exceeds embedding's margin."""
    embedding_margin = top_two_margin(embedding_cost, legal)
    local_margin = top_two_margin(local_cost, legal)
    return (embedding_margin is not None and local_margin is not None
            and local_margin > embedding_margin)


def _normalized(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim < 2 or not np.isfinite(values).all():
        raise ValueError("local token values must be finite and have a feature axis")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    valid = norms[..., 0] > 1e-8
    return values / np.maximum(norms, 1e-8), valid


def visibility_weighted_fifo_costs(
    history_local,
    history_visibility,
    candidate_local,
    candidate_visibility,
):
    """Return fixed visibility-weighted local FIFO costs and availability.

    ``history_local`` is ``(H, 4, D)`` and ``candidate_local`` is
    ``(M, 4, D)``.  A candidate is locally available when at least one strip
    has both positive visibility and a finite nonzero descriptor, and the
    corresponding historical strip has support.  The returned cost is in
    ``[0, 1]`` for available candidates and ``1`` otherwise.
    """
    history_local = np.asarray(history_local, dtype=np.float32)
    candidate_local = np.asarray(candidate_local, dtype=np.float32)
    history_visibility = np.asarray(history_visibility, dtype=np.float32)
    candidate_visibility = np.asarray(candidate_visibility, dtype=np.float32)
    if (history_local.ndim != 3 or candidate_local.ndim != 3
            or history_local.shape[1] != candidate_local.shape[1]
            or history_local.shape[2] != candidate_local.shape[2]
            or history_visibility.shape != history_local.shape[:2]
            or candidate_visibility.shape != candidate_local.shape[:2]):
        raise ValueError("local tokens and visibility arrays must be aligned")
    if (not np.isfinite(history_visibility).all()
            or not np.isfinite(candidate_visibility).all()
            or (history_visibility < 0).any() or (candidate_visibility < 0).any()
            or (history_visibility > 1).any() or (candidate_visibility > 1).any()):
        raise ValueError("visibility fractions must be finite in [0, 1]")
    if not len(history_local):
        return np.ones(len(candidate_local), dtype=np.float64), np.zeros(
            len(candidate_local), dtype=bool)

    history_norm, history_finite = _normalized(history_local)
    candidate_norm, candidate_finite = _normalized(candidate_local)
    history_weights = history_visibility * history_finite.astype(np.float32)
    support = history_weights.sum(axis=0)
    prototype = (history_norm * history_weights[..., None]).sum(axis=0)
    prototype = prototype / np.maximum(np.linalg.norm(prototype, axis=-1, keepdims=True), 1e-8)
    prototype_finite = (support > 0) & (np.linalg.norm(prototype, axis=-1) > 1e-8)

    candidate_weights = candidate_visibility * candidate_finite.astype(np.float32)
    candidate_weights = candidate_weights * prototype_finite[None, :]
    available = candidate_weights.sum(axis=1) > 0
    cosine = np.sum(candidate_norm * prototype[None, :, :], axis=-1)
    cosine = np.clip(cosine, -1.0, 1.0)
    costs = (1.0 - cosine) / 2.0
    costs = (costs * candidate_weights).sum(axis=1) / np.maximum(
        candidate_weights.sum(axis=1), 1e-8)
    costs = np.where(available, costs, 1.0)
    return costs.astype(np.float64), available.astype(bool)
