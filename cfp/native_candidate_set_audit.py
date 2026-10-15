"""Retrospective candidate-set and unknown/NULL boundary measurements."""

from __future__ import annotations

import collections
import gzip
import json
from pathlib import Path

import numpy as np


def new_counts() -> collections.Counter:
    return collections.Counter(
        observed_track_rows=0,
        empty_history_rows=0,
        clean_history_rows=0,
        mixed_or_unknown_history_rows=0,
        candidate_rows=0,
        no_legal_candidate_rows=0,
        unknown_legal_rows=0,
        known_only_legal_rows=0,
        no_known_legal_rows=0,
        competition_rows=0,
        duplicate_same_identity_rows=0,
        clean_positive_rows=0,
        clean_positive_missing_rows=0,
        clean_positive_unknown_competition_rows=0,
        clean_positive_other_identity_competition_rows=0,
        clean_positive_duplicate_same_identity_rows=0,
        clean_baseline_top1_correct=0,
        clean_baseline_top1_wrong=0,
        history_index_checks=0,
        candidate_index_checks=0,
    )


def _history_kind(history, labels, frames, current_frame):
    indices = []
    history_labels = []
    for frame, index in history:
        if index is None:
            continue
        index = int(index)
        if not 0 <= index < len(labels) or int(frames[index]) != int(frame):
            raise ValueError("Native history proposal/frame alignment drift")
        indices.append(index)
        history_labels.append(int(labels[index]))
    if not indices:
        return "empty", None, indices
    known = [label for label in history_labels if label >= 0]
    if len(known) == len(history_labels) and len(set(known)) == 1:
        return "clean", known[0], indices
    return "mixed_or_unknown", None, indices


def summarize_trace(trace_path: Path, physical: dict, raw_geometry_max: float = 0.9) -> dict:
    """Summarize legal candidate labels without changing any tracker output."""
    labels = np.asarray(physical["labels"])
    frames = np.asarray(physical["frame_of_proposal"])
    counts = new_counts()
    by_history = collections.defaultdict(new_counts)
    legal_candidate_counts = []
    known_identity_counts = []
    unknown_candidate_counts = []

    with gzip.open(trace_path, "rt") as stream:
        for line in stream:
            event = json.loads(line)
            current_frame = int(event["frame"])
            for row in event["rows"]:
                counts["observed_track_rows"] += 1
                history = [(int(frame), None if index is None else int(index))
                           for frame, index in row["history"]]
                kind, target, _ = _history_kind(history, labels, frames, current_frame)
                counts[f"{kind}_history_rows"] += 1
                by_history[kind][f"{kind}_history_rows"] += 1
                proposals = np.asarray(row["proposals"], dtype=np.int64)
                raw = np.asarray(row["raw_geometry"], dtype=np.float64)
                baseline = np.asarray(row["baseline"], dtype=np.float64)
                if len(proposals) != len(raw) or len(raw) != len(baseline):
                    raise ValueError("Candidate edge dimensions disagree")
                for index in proposals:
                    if not 0 <= index < len(labels) or int(frames[index]) != current_frame:
                        raise ValueError("Native current proposal/frame alignment drift")
                    counts["candidate_index_checks"] += 1
                if not len(proposals):
                    continue
                counts["candidate_rows"] += 1
                candidate_labels = labels[proposals]
                legal = np.isfinite(raw) & (raw <= raw_geometry_max)
                legal_labels = candidate_labels[legal]
                legal_cost = baseline[legal]
                legal_candidate_counts.append(int(len(legal_labels)))
                unknown_count = int(np.sum(legal_labels < 0))
                known_labels = [int(label) for label in legal_labels if label >= 0]
                known_identity_counts.append(len(set(known_labels)))
                unknown_candidate_counts.append(unknown_count)
                if not len(legal_labels):
                    counts["no_legal_candidate_rows"] += 1
                    continue
                if unknown_count:
                    counts["unknown_legal_rows"] += 1
                else:
                    counts["known_only_legal_rows"] += 1
                if not known_labels:
                    counts["no_known_legal_rows"] += 1
                if len(set(known_labels)) >= 2:
                    counts["competition_rows"] += 1
                known_counts = collections.Counter(known_labels)
                if any(value >= 2 for value in known_counts.values()):
                    counts["duplicate_same_identity_rows"] += 1

                if kind != "clean":
                    continue
                positive = legal_labels == target
                if not np.any(positive):
                    counts["clean_positive_missing_rows"] += 1
                    continue
                counts["clean_positive_rows"] += 1
                if unknown_count:
                    counts["clean_positive_unknown_competition_rows"] += 1
                if any(label >= 0 and label != target for label in legal_labels):
                    counts["clean_positive_other_identity_competition_rows"] += 1
                if int(np.sum(positive)) >= 2:
                    counts["clean_positive_duplicate_same_identity_rows"] += 1
                order = np.argsort(np.where(legal, baseline, np.inf), kind="stable")
                order = order[np.isfinite(np.where(legal, baseline, np.inf)[order])]
                if len(order) and bool(candidate_labels[order[0]] == target):
                    counts["clean_baseline_top1_correct"] += 1
                elif len(order):
                    counts["clean_baseline_top1_wrong"] += 1

    return {
        "counts": dict(counts),
        "legal_candidate_count": {
            "n": len(legal_candidate_counts),
            "mean": float(np.mean(legal_candidate_counts)) if legal_candidate_counts else None,
            "q10_q50_q90": np.quantile(legal_candidate_counts, [0.1, 0.5, 0.9]).tolist()
            if legal_candidate_counts else None,
        },
        "known_identity_count": {
            "n": len(known_identity_counts),
            "mean": float(np.mean(known_identity_counts)) if known_identity_counts else None,
            "q10_q50_q90": np.quantile(known_identity_counts, [0.1, 0.5, 0.9]).tolist()
            if known_identity_counts else None,
        },
        "unknown_candidate_count": {
            "n": len(unknown_candidate_counts),
            "mean": float(np.mean(unknown_candidate_counts)) if unknown_candidate_counts else None,
            "q10_q50_q90": np.quantile(unknown_candidate_counts, [0.1, 0.5, 0.9]).tolist()
            if unknown_candidate_counts else None,
        },
        "history_classes": {key: dict(value) for key, value in by_history.items()},
    }
