"""Pure fixed-FIFO helpers for the native patch-attention audit.

The audit uses these helpers on an already locked global-embedding candidate
graph.  They contain no identity labels, tracker state, fitting, or threshold
search.  Patch features are normalized per observation, averaged over the
strictly-past FIFO history, and normalized once more before cosine distance.
"""

from __future__ import annotations

import numpy as np


def fifo_cosine_costs(history_features, candidate_features):
    """Return fixed FIFO cosine costs and candidate availability.

    ``history_features`` is ``(H, D)`` and ``candidate_features`` is
    ``(M, D)``.  Every finite, nonzero historical vector contributes equally
    after per-vector normalization.  A candidate is available only when its
    vector is finite and nonzero.  Unavailable candidates receive cost 1.
    """
    history_features = np.asarray(history_features, dtype=np.float32)
    candidate_features = np.asarray(candidate_features, dtype=np.float32)
    if (history_features.ndim != 2 or candidate_features.ndim != 2
            or history_features.shape[1] != candidate_features.shape[1]):
        raise ValueError("history and candidate features must be aligned matrices")
    if not np.isfinite(history_features).all() or not np.isfinite(candidate_features).all():
        raise ValueError("patch-attention features must be finite")
    if not len(history_features):
        return np.ones(len(candidate_features), dtype=np.float64), np.zeros(
            len(candidate_features), dtype=bool)
    history_norms = np.linalg.norm(history_features, axis=1, keepdims=True)
    candidate_norms = np.linalg.norm(candidate_features, axis=1, keepdims=True)
    history_valid = history_norms[:, 0] > 1e-8
    candidate_valid = candidate_norms[:, 0] > 1e-8
    if not history_valid.any():
        return np.ones(len(candidate_features), dtype=np.float64), np.zeros(
            len(candidate_features), dtype=bool)
    history_normalized = history_features / np.maximum(history_norms, 1e-8)
    prototype = history_normalized[history_valid].mean(axis=0)
    prototype_norm = np.linalg.norm(prototype)
    if not np.isfinite(prototype_norm) or prototype_norm <= 1e-8:
        return np.ones(len(candidate_features), dtype=np.float64), np.zeros(
            len(candidate_features), dtype=bool)
    prototype = prototype / prototype_norm
    candidate_normalized = candidate_features / np.maximum(candidate_norms, 1e-8)
    cosine = np.clip(candidate_normalized @ prototype, -1.0, 1.0)
    costs = (1.0 - cosine) / 2.0
    costs = np.where(candidate_valid, costs, 1.0)
    return costs.astype(np.float64), candidate_valid.astype(bool)
