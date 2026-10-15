"""Fixed observable NULL abstention for native candidate-set replay.

The rule operates only on the current association cost matrix and current
geometry.  It never inspects physical proposal labels, annotations, or future
frames.  A row is abstained when two legal candidates are both within the
declared NULL cost and their costs are within a fixed ambiguity margin.  The
caller keeps the ordinary lifecycle and lets the unmatched-track path handle
the abstention.
"""

from __future__ import annotations

import numpy as np

from yolox.tracker import matching


def apply_unknown_preserving_null(
    baseline: np.ndarray,
    raw_geometry: np.ndarray,
    *,
    null_cost: float,
    ambiguity_margin: float,
    legal_geometry_max: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply one fixed row-wise ambiguity/NULL rule to an association matrix.

    ``null_cost`` is an abstention cost, not a physical unknown label.  The
    assignment threshold remains owned by the frozen tracker.  Rows rejected
    by the rule are assigned a value just above that cost, so Hungarian
    matching leaves them unmatched at the frozen threshold.
    """

    baseline = np.asarray(baseline, dtype=np.float64)
    raw_geometry = np.asarray(raw_geometry, dtype=np.float64)
    if baseline.ndim != 2 or raw_geometry.shape != baseline.shape:
        raise ValueError("Aligned association and geometry matrices required")
    if not np.isfinite([null_cost, ambiguity_margin, legal_geometry_max]).all():
        raise ValueError("Finite NULL rule parameters required")
    if null_cost < 0 or ambiguity_margin < 0 or not 0 <= legal_geometry_max <= 1:
        raise ValueError("Invalid NULL rule parameters")

    result = baseline.copy()
    counts = dict(
        rows=baseline.shape[0],
        legal_rows=0,
        two_legal_rows=0,
        ambiguous_rows=0,
        abstained_rows=0,
    )
    # A value immediately above the frozen assignment threshold is supplied by
    # the installer as ``null_cost``.  ``nextafter`` keeps the operation
    # deterministic without inventing an arbitrary large sentinel.
    rejected_cost = np.nextafter(float(null_cost), np.inf)
    for row in range(len(baseline)):
        legal = np.isfinite(raw_geometry[row]) & (raw_geometry[row] <= legal_geometry_max)
        legal &= np.isfinite(baseline[row])
        legal_count = int(legal.sum())
        if legal_count:
            counts["legal_rows"] += 1
        if legal_count < 2:
            continue
        counts["two_legal_rows"] += 1
        values = np.sort(baseline[row, legal])
        ambiguous = bool(
            values[1] <= null_cost
            and (values[1] - values[0]) <= ambiguity_margin
        )
        if not ambiguous:
            continue
        counts["ambiguous_rows"] += 1
        result[row, legal] = rejected_cost
        counts["abstained_rows"] += 1
    return result, counts


def install_unknown_preserving_null(
    tracker,
    *,
    enabled: bool,
    null_cost: float,
    ambiguity_margin: float,
    legal_geometry_max: float,
):
    """Install the fixed rule while preserving the base tracker when disabled."""

    base = type(tracker)

    class NullRuleTracker(base):
        def association_cost(self, tracks, detections):
            baseline = super().association_cost(tracks, detections)
            if not self.null_rule_enabled or not baseline.size:
                return baseline
            # The rule is for confirmed native histories.  Unconfirmed tracks
            # retain the frozen baseline's normal birth/confirmation behavior.
            if any(not track.is_activated for track in tracks):
                return baseline
            raw_geometry = matching.iou_distance(tracks, detections)
            result, counts = apply_unknown_preserving_null(
                baseline,
                raw_geometry,
                null_cost=self.null_rule_null_cost,
                ambiguity_margin=self.null_rule_ambiguity_margin,
                legal_geometry_max=self.null_rule_legal_geometry_max,
            )
            for key, value in counts.items():
                self.null_rule_counts[key] += int(value)
            return result

    tracker.__class__ = NullRuleTracker
    tracker.null_rule_enabled = bool(enabled)
    tracker.null_rule_null_cost = float(null_cost)
    tracker.null_rule_ambiguity_margin = float(ambiguity_margin)
    tracker.null_rule_legal_geometry_max = float(legal_geometry_max)
    tracker.null_rule_counts = {
        key: 0
        for key in ("rows", "legal_rows", "two_legal_rows", "ambiguous_rows", "abstained_rows")
    }
    return tracker
