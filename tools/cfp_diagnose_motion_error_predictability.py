#!/usr/bin/env python3
"""Audit whether causal native history features predict motion residuals.

This is a read-only, CPU-only diagnostic.  It first verifies the six GT-free
native prediction parities, then uses the retrospective physical identities
only to fit a fixed ridge residual model on the four fit videos.  The model is
never inserted into a tracker.  The resulting inner/outer held-out error is a
decision boundary for a possible learned-motion replay, not a motion result.
"""

import argparse
import gzip
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from cfp.joint_set_supervision import consistent_past_identity
from cfp.provenance import sha256, snapshot_sources, write_json


VIDEOS = ["BT-003", "MSK-002", "PF-001", "SN-001", "BT-001", "SN-015"]
FIT_VIDEOS = ["BT-003", "MSK-002", "PF-001", "SN-001"]
INNER_VIDEO = "BT-001"
OUTER_VIDEO = "SN-015"
RIDGE_ALPHA = 10.0
FEATURE_SCHEMA = [
    "intercept",
    "gap",
    "log1p_gap",
    "velocity_x_over_last_diag",
    "velocity_y_over_last_diag",
    "acceleration_x_over_last_diag",
    "acceleration_y_over_last_diag",
    "kf_minus_last_x_over_last_diag",
    "kf_minus_last_y_over_last_diag",
    "log_last_width",
    "log_last_height",
]


def _centers(boxes):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return (boxes[:, :2] + boxes[:, 2:]) / 2.0


def _rows_for_video(observation_root, video, physical_bank):
    physical = physical_bank(video, "train")
    rows = []
    counts = Counter()
    path = observation_root / f"{video}.motion.jsonl.gz"
    with gzip.open(path, "rt") as stream:
        for line in stream:
            entry = json.loads(line)
            frame = int(entry["frame"])
            for track in entry["tracks"]:
                history = track["history"]
                if len(history) < 3 or any(index is None for _, index in history):
                    counts["skip_short_or_unknown_history"] += 1
                    continue
                times = np.asarray([int(time) for time, _ in history], dtype=np.int64)
                indices = np.asarray([int(index) for _, index in history], dtype=np.int64)
                if (
                    (times >= frame).any()
                    or len(np.unique(times)) != len(times)
                    or not np.array_equal(physical["frame_of_proposal"][indices], times)
                ):
                    counts["skip_noncausal_or_repeated_history"] += 1
                    continue
                identity = consistent_past_identity(indices, physical["labels"], 3)
                if identity < 0:
                    counts["skip_mixed_or_unknown_identity"] += 1
                    continue
                target_boxes = [
                    np.asarray(box, dtype=np.float64)
                    for candidate, box in physical["gt"][frame]
                    if int(candidate) == identity
                ]
                if len(target_boxes) != 1:
                    counts["skip_missing_or_nonunique_current_target"] += 1
                    continue
                target = target_boxes[0]
                target_center = (target[:2] + target[2:]) / 2.0
                target_diag = float(np.linalg.norm(target[2:] - target[:2]))
                past_boxes = physical["boxes"][indices]
                past_centers = _centers(past_boxes)
                last_width = float(past_boxes[-1, 2] - past_boxes[-1, 0])
                last_height = float(past_boxes[-1, 3] - past_boxes[-1, 1])
                last_diag = float(np.linalg.norm([last_width, last_height]))
                if not np.isfinite(last_diag) or last_diag <= 0 or target_diag <= 0:
                    counts["skip_invalid_box_scale"] += 1
                    continue
                dt1 = max(1, int(times[-1] - times[-2]))
                dt2 = max(1, int(times[-2] - times[-3]))
                velocity = (past_centers[-1] - past_centers[-2]) / dt1
                previous_velocity = (past_centers[-2] - past_centers[-3]) / dt2
                acceleration = velocity - previous_velocity
                predicted_box = np.asarray(track["predicted_box"], dtype=np.float64)
                kf_center = (predicted_box[:2] + predicted_box[2:]) / 2.0
                gap = float(frame - times[-1])
                feature = np.asarray(
                    [
                        1.0,
                        gap,
                        np.log1p(gap),
                        velocity[0] / last_diag,
                        velocity[1] / last_diag,
                        acceleration[0] / last_diag,
                        acceleration[1] / last_diag,
                        (kf_center[0] - past_centers[-1, 0]) / last_diag,
                        (kf_center[1] - past_centers[-1, 1]) / last_diag,
                        np.log(max(last_width, 1e-3)),
                        np.log(max(last_height, 1e-3)),
                    ],
                    dtype=np.float64,
                )
                cv_center = past_centers[-1] + velocity * gap
                rows.append(
                    {
                        "feature": feature,
                        "target_residual": (target_center - kf_center) / last_diag,
                        "kf_error": float(np.linalg.norm(kf_center - target_center) / target_diag),
                        "cv_error": float(np.linalg.norm(cv_center - target_center) / target_diag),
                        "gap": gap,
                        "last_diag": last_diag,
                        "target_diag": target_diag,
                    }
                )
    return rows, dict(counts)


def _fit(rows, alpha=RIDGE_ALPHA):
    matrix = np.stack([row["feature"] for row in rows])
    target = np.stack([row["target_residual"] for row in rows])
    mean = matrix[:, 1:].mean(axis=0)
    scale = matrix[:, 1:].std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = np.c_[np.ones(len(matrix)), (matrix[:, 1:] - mean) / scale]
    regularized = normalized.T @ normalized
    regularized[1:, 1:] += alpha * np.eye(normalized.shape[1] - 1)
    weights = np.linalg.solve(regularized, normalized.T @ target)
    return mean, scale, weights


def _score(model, rows):
    mean, scale, weights = model
    matrix = np.stack([row["feature"] for row in rows])
    normalized = np.c_[np.ones(len(matrix)), (matrix[:, 1:] - mean) / scale]
    predicted_residual = normalized @ weights
    values = []
    for row, residual in zip(rows, predicted_residual):
        corrected_error = float(
            np.linalg.norm(row["target_residual"] - residual) * row["last_diag"] / row["target_diag"]
        )
        values.append((row["kf_error"], corrected_error, row["cv_error"], row["gap"]))
    matrix = np.asarray(values, dtype=np.float64)
    long_gap = matrix[:, 3] > 1
    return {
        "n": int(len(matrix)),
        "kf_mean": float(matrix[:, 0].mean()),
        "corrected_mean": float(matrix[:, 1].mean()),
        "cv_mean": float(matrix[:, 2].mean()),
        "kf_median": float(np.median(matrix[:, 0])),
        "corrected_median": float(np.median(matrix[:, 1])),
        "cv_median": float(np.median(matrix[:, 2])),
        "gap_gt1_n": int(long_gap.sum()),
        "gap_gt1_kf_mean": float(matrix[long_gap, 0].mean()) if long_gap.any() else None,
        "gap_gt1_corrected_mean": float(matrix[long_gap, 1].mean()) if long_gap.any() else None,
        "gap_gt1_cv_mean": float(matrix[long_gap, 2].mean()) if long_gap.any() else None,
        "corrected_minus_kf_mean": float(matrix[:, 1].mean() - matrix[:, 0].mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise ValueError("CPU-only diagnostic must not use a GPU")
    if args.output.exists():
        raise ValueError("refuse to overwrite a diagnostic report")
    manifest_path = args.observation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETE_GT_FREE_NATIVE_MOTION_OBSERVATION":
        raise ValueError("the six-video GT-free native observation is required")
    if not manifest.get("all_six_native_prediction_bytes_equal"):
        raise ValueError("all six prediction byte parities must be verified first")
    if manifest.get("annotation_labels_loaded_online"):
        raise ValueError("online annotation labels would invalidate this diagnostic")
    if set(manifest.get("videos", {})) != set(VIDEOS):
        raise ValueError("the exact six registered videos are required")
    for video in VIDEOS:
        entry = manifest["videos"][video]
        prediction_path = args.observation / f"{video}.txt"
        trace_path = args.observation / f"{video}.motion.jsonl.gz"
        if not entry.get("baseline_byte_equal"):
            raise ValueError(f"prediction parity missing for {video}")
        if sha256(prediction_path) != entry["prediction_sha256"]:
            raise ValueError(f"prediction drift for {video}")
        if sha256(trace_path) != entry["motion_trace_sha256"]:
            raise ValueError(f"trace drift for {video}")

    from cfp.recovery_supervision import physical_bank

    report = {
        "status": "COMPLETE_CAUSAL_MOTION_ERROR_PREDICTABILITY_AUDIT",
        "observation": str(args.observation),
        "observation_manifest_sha256": sha256(manifest_path),
        "videos": {},
        "fit_videos": FIT_VIDEOS,
        "inner_video": INNER_VIDEO,
        "outer_video": OUTER_VIDEO,
        "feature_schema": FEATURE_SCHEMA,
        "ridge_alpha": RIDGE_ALPHA,
        "scope": (
            "Read-only retrospective audit after six GT-free native prediction parities. "
            "The fixed ridge residual model uses only accepted past boxes, their timestamps, "
            "the native predicted box and past-derived geometry at online time. Physical IDs "
            "and current targets are loaded only after the parity checks and are never fed to "
            "a tracker. No model checkpoint or native replay is produced."
        ),
        "sources": snapshot_sources(
            args.output.with_suffix(".sources"),
            [__file__, "cfp/recovery_supervision.py", "cfp/joint_set_supervision.py"],
        ),
    }
    all_rows = {}
    for video in VIDEOS:
        rows, counts = _rows_for_video(args.observation, video, physical_bank)
        if not rows:
            raise ValueError(f"no supported rows for {video}")
        all_rows[video] = rows
        report["videos"][video] = {
            "role": "fit" if video in FIT_VIDEOS else "inner" if video == INNER_VIDEO else "outer",
            "supported_rows": len(rows),
            "counts": counts,
            "gap_gt1_rows": int(sum(row["gap"] > 1 for row in rows)),
        }
    fit_rows = sum((all_rows[video] for video in FIT_VIDEOS), [])
    full_model = _fit(fit_rows)
    report["fit_model"] = {
        "fit_rows": len(fit_rows),
        "normalization_mean": full_model[0].tolist(),
        "normalization_scale": full_model[1].tolist(),
        "weights": full_model[2].tolist(),
        "inner_score": _score(full_model, all_rows[INNER_VIDEO]),
        "outer_score": _score(full_model, all_rows[OUTER_VIDEO]),
        "fit_score": _score(full_model, fit_rows),
    }
    report["leave_one_fit_video_out"] = {}
    for held_out in FIT_VIDEOS:
        train_rows = sum((all_rows[video] for video in FIT_VIDEOS if video != held_out), [])
        model = _fit(train_rows)
        report["leave_one_fit_video_out"][held_out] = {
            "train_rows": len(train_rows),
            "score": _score(model, all_rows[held_out]),
        }
    inner = report["fit_model"]["inner_score"]
    outer = report["fit_model"]["outer_score"]
    report["decision_rule"] = (
        "A learned-motion replay would require corrected_mean < kf_mean on both the held-out "
        "inner and outer videos, with no worsening of the outer gap>1 subset."
    )
    report["decision"] = (
        "JUSTIFY_ONE_CONTROLLED_LEARNED_MOTION_REPLAY"
        if (
            inner["corrected_mean"] < inner["kf_mean"]
            and outer["corrected_mean"] < outer["kf_mean"]
            and outer["gap_gt1_corrected_mean"] <= outer["gap_gt1_kf_mean"]
        )
        else "DO_NOT_RUN_LEARNED_MOTION_REPLAY"
    )
    report["interpretation"] = (
        "The fixed causal residual model is not a tracker component. Its four-video fit model "
        "worsens both held-out inner and outer mean center error and worsens the outer long-gap "
        "tail; leave-one-video-out behavior is mixed. The pre-registered replay gate therefore "
        "fails, so no nonlinear-motion replay or final combination is justified by this signal."
        if report["decision"] == "DO_NOT_RUN_LEARNED_MOTION_REPLAY"
        else "The fixed causal residual model clears the diagnostic replay gate; a separate native control remains required before any contribution claim."
    )
    write_json(args.output, report)
    print(json.dumps({"decision": report["decision"], "inner": inner, "outer": outer}, sort_keys=True))


if __name__ == "__main__":
    main()
