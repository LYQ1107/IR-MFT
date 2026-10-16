#!/usr/bin/env python3
"""Audit patch-attention complementarity on a locked native candidate graph.

This reducer deliberately does not replay a tracker.  It verifies the 18
accepted, source-matched fitted-global replays and their causal traces first,
then evaluates a fixed patch-attention FIFO coordinate on those exact edges.
Physical labels are opened only after every prediction and trace hash is
verified.  A failed gate is a scientific negative boundary and forbids a new
combination replay.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from cfp.native_local4_complementarity import margin_gate, top1_positive
from cfp.native_patch_attention_complementarity import fifo_cosine_costs
from cfp.provenance import sha256, snapshot_sources, write_json


VIDEOS = ("BT-001", "BT-003", "MSK-002", "PF-001", "SN-001", "SN-015")
SEEDS = (0, 1, 2)
METRICS = (
    "rows", "candidate_missing_rows", "patch_unavailable_rows",
    "embedding_top1", "patch_top1", "patch_correction", "patch_harm",
    "patch_disagreement", "margin_activation", "margin_top1",
    "margin_correction", "margin_harm", "margin_changed_top",
)


def _new_counts():
    return collections.Counter({key: 0 for key in METRICS})


def _add_counts(left, right):
    out = collections.Counter(left)
    out.update(right)
    return out


def _load_arrays(path):
    return {
        name: np.load(path / f"{name}.npy", mmap_mode="r")
        for name in ("frame_ids", "offsets", "embedding", "valid")
    }


def _load_patch_bank(assets_root, video, seed):
    detections_root = assets_root / "phase0" / "causal_v3_train_detections"
    detection_path = detections_root / f"{video}.npz"
    detection_manifest = json.loads((detections_root / "manifest.json").read_text())
    patch_path = assets_root / "phase7" / (
        f"native_dino_patch_attention_{video}_k4_s{seed}_v1")
    global_path = assets_root / "phase7" / (
        f"native_dino_global_{video}_k4_s{seed}_v1")
    patch_manifest_path = patch_path / "manifest.json"
    global_manifest_path = global_path / "manifest.json"
    patch_manifest = json.loads(patch_manifest_path.read_text())
    global_manifest = json.loads(global_manifest_path.read_text())
    expected_detection_sha = detection_manifest["caches"][video]["sha256"]
    if (patch_manifest.get("status") != "COMPLETE_NATIVE_CURRENT_DINO_FEATURES"
            or patch_manifest.get("video") != video
            or patch_manifest.get("detection_sha256") != expected_detection_sha
            or patch_manifest.get("encoder", {}).get("trained_encoder", {}).get("outer") != video
            or patch_manifest.get("encoder", {}).get("trained_encoder", {}).get("seed") != seed
            or global_manifest.get("status") != "COMPLETE_NATIVE_CURRENT_DINO_FEATURES"
            or global_manifest.get("detection_sha256") != expected_detection_sha
            or sha256(detection_path) != expected_detection_sha):
        raise ValueError(f"Patch/global source provenance mismatch: {video} s{seed}")
    with np.load(detection_path, allow_pickle=False) as archive:
        frames, offsets, boxes = [archive[name].copy() for name in
                                  ("frame_ids", "offsets", "boxes")]
    patch = _load_arrays(patch_path)
    global_arrays = _load_arrays(global_path)
    if (not np.array_equal(frames, patch["frame_ids"])
            or not np.array_equal(offsets, patch["offsets"])
            or not np.array_equal(frames, global_arrays["frame_ids"])
            or not np.array_equal(offsets, global_arrays["offsets"])
            or patch["embedding"].shape != (len(boxes), 384)
            or len(patch["valid"]) != len(boxes)
            or global_arrays["embedding"].shape != (len(boxes), 384)
            or len(global_arrays["valid"]) != len(boxes)):
        raise ValueError(f"Patch/global proposal indexing drift: {video} s{seed}")
    return patch, dict(
        patch_manifest_sha256=sha256(patch_manifest_path),
        global_manifest_sha256=sha256(global_manifest_path),
        patch_embedding_sha256=sha256(patch_path / "embedding.npy"),
        patch_valid_sha256=sha256(patch_path / "valid.npy"),
        frame_ids_sha256=sha256(patch_path / "frame_ids.npy"),
        offsets_sha256=sha256(patch_path / "offsets.npy"),
        detection_sha256=expected_detection_sha,
        patch_path=str(patch_path),
        global_path=str(global_path),
    )


def _verify_locked_controls(config, assets_root, trace_root, control_report_path):
    if not control_report_path.exists():
        raise FileNotFoundError(control_report_path)
    control_report = json.loads(control_report_path.read_text())
    if (control_report.get("status") != "COMPLETE_LOCAL4_COMPLEMENTARITY_AUDIT"
            or control_report.get("coverage", {}).get("completed_replays") != 18):
        raise ValueError("The matched-control complementarity report is not complete")
    expected = control_report.get("per_replay", {})
    controls = {}
    for video in VIDEOS:
        for seed in SEEDS:
            name = f"{video}_s{seed}"
            if name not in expected:
                raise ValueError(f"Missing matched-control replay report: {name}")
            row = expected[name]
            trace_path = trace_root / name / "complementarity.jsonl.gz"
            prediction_path = trace_root / name / f"{video}.txt"
            if not trace_path.exists() or not prediction_path.exists():
                raise FileNotFoundError(f"Missing locked matched-control artifact: {name}")
            trace_sha = sha256(trace_path)
            prediction_sha = sha256(prediction_path)
            if (trace_sha != row.get("trace_sha256")
                    or prediction_sha != row.get("prediction_sha256")
                    or row.get("prediction_byte_equal") is not True):
                raise ValueError(f"Matched-control hash/parity drift: {name}")
            controls[name] = dict(
                trace_path=str(trace_path), prediction_path=str(prediction_path),
                trace_sha256=trace_sha, prediction_sha256=prediction_sha,
                frames=int(row["frames"]), output_rows=int(row["output_rows"]),
            )
    return control_report, controls


def _patch_costs(row, patch, frame, frame_of_proposal):
    history = [(int(timestamp), int(index)) for timestamp, index in row["history"]]
    if any(timestamp >= frame for timestamp, _ in history):
        raise ValueError("Patch history contains a current or future proposal")
    if any(history[i][0] >= history[i + 1][0] for i in range(len(history) - 1)):
        raise ValueError("Patch history timestamps are not strictly increasing")
    history = [(timestamp, index) for timestamp, index in history
               if frame - 1500 <= timestamp][-4:]
    history_indices = []
    for timestamp, index in history:
        if not 0 <= index < len(patch["embedding"]):
            raise ValueError("Patch history index is out of range")
        if int(frame_of_proposal[index]) != timestamp:
            raise ValueError("Patch history frame/index alignment drift")
        if not bool(patch["valid"][index]):
            raise ValueError("Patch identity history contains an invalid source row")
        history_indices.append(index)
    proposals = np.asarray(row["proposals"], dtype=np.int64)
    if np.any(proposals < 0) or np.any(proposals >= len(patch["embedding"])):
        raise ValueError("Patch current proposal index is out of range")
    candidate_features = np.asarray(patch["embedding"][proposals], dtype=np.float32)
    history_features = np.asarray(patch["embedding"][history_indices], dtype=np.float32)
    costs, available = fifo_cosine_costs(history_features, candidate_features)
    available &= np.asarray(patch["valid"][proposals], dtype=bool)
    return costs, available


def _summarize_trace(trace_path, patch, physical):
    labels = np.asarray(physical["labels"])
    frame_of_proposal = np.asarray(physical["frame_of_proposal"])
    patch_frame_of_proposal = np.repeat(
        np.asarray(patch["frame_ids"]), np.diff(np.asarray(patch["offsets"])))
    if not np.array_equal(frame_of_proposal, patch_frame_of_proposal):
        raise ValueError("Patch and physical proposal frame indexing drift")
    counts = _new_counts()
    with gzip.open(trace_path, "rt") as stream:
        for line in stream:
            event = json.loads(line)
            frame = int(event["frame"])
            for row in event["rows"]:
                history = row["history"]
                if not history:
                    continue
                history_labels = []
                for timestamp, index in history:
                    if (not 0 <= int(index) < len(labels)
                            or int(frame_of_proposal[int(index)]) != int(timestamp)):
                        raise ValueError("Retrospective history frame/index drift")
                    history_labels.append(int(labels[int(index)]))
                known = [value for value in history_labels if value >= 0]
                if len(known) != len(history_labels) or len(set(known)) != 1:
                    continue
                proposals = np.asarray(row["proposals"], dtype=np.int64)
                if any(not 0 <= int(index) < len(labels)
                       or int(frame_of_proposal[int(index)]) != frame
                       for index in proposals):
                    raise ValueError("Retrospective current proposal/frame drift")
                candidate_labels = labels[proposals]
                positive = candidate_labels == known[0]
                raw_geometry = np.asarray(row["raw_geometry"], dtype=np.float64)
                geometry = np.asarray(row["geometry"], dtype=np.float64)
                embedding = np.asarray(row["embedding_appearance"], dtype=np.float64)
                baseline = np.asarray(row["baseline"], dtype=np.float64)
                candidate_valid = np.asarray(row["candidate_valid"], dtype=bool)
                if not all(len(value) == len(proposals) for value in
                           (raw_geometry, geometry, embedding, baseline, candidate_valid)):
                    raise ValueError("Retrospective edge dimensions disagree")
                patch_cost, patch_valid = _patch_costs(row, patch, frame,
                                                       patch_frame_of_proposal)
                if not np.isfinite(patch_cost).all():
                    raise ValueError("Nonfinite patch-attention retrospective cost")
                legal = (candidate_valid & np.isfinite(raw_geometry)
                         & (raw_geometry <= 0.9))
                if not np.any(positive & legal):
                    counts["candidate_missing_rows"] += 1
                    continue
                counts["rows"] += 1
                counts["patch_unavailable_rows"] += int(not patch_valid.any())
                embedding_combined = geometry.copy()
                embedding_combined[legal] = (.7 * geometry[legal]
                                             + .3 * embedding[legal])
                patch_legal = legal & patch_valid
                patch_combined = embedding_combined.copy()
                patch_combined[patch_legal] = (.7 * geometry[patch_legal]
                                               + .3 * patch_cost[patch_legal])
                _, embedding_top = top1_positive(embedding_combined, legal, positive)
                _, patch_top = top1_positive(patch_combined, legal, positive)
                _, baseline_top = top1_positive(baseline, legal, positive)
                if embedding_top != baseline_top:
                    raise ValueError("Recomputed embedding graph differs from locked baseline")
                counts["embedding_top1"] += int(embedding_top)
                counts["patch_top1"] += int(patch_top)
                counts["patch_correction"] += int(not embedding_top and patch_top)
                counts["patch_harm"] += int(embedding_top and not patch_top)
                counts["patch_disagreement"] += int(embedding_top != patch_top)
                use_patch = margin_gate(embedding, patch_cost, legal & patch_valid)
                if use_patch:
                    counts["margin_activation"] += 1
                    selected = patch_combined
                else:
                    selected = embedding_combined
                _, margin_top = top1_positive(selected, legal, positive)
                counts["margin_top1"] += int(margin_top)
                counts["margin_correction"] += int(not embedding_top and margin_top)
                counts["margin_harm"] += int(embedding_top and not margin_top)
                counts["margin_changed_top"] += int(margin_top != embedding_top)
    return dict(counts=dict(counts), promotion=dict(
        patch_correction_gt_harm=bool(counts["patch_correction"] > counts["patch_harm"]),
        margin_activation_positive=bool(counts["margin_activation"] > 0),
        margin_correction_gt_harm=bool(
            counts["margin_correction"] > counts["margin_harm"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise ValueError("This audit is CPU-only")
    if args.output.exists() or args.output.resolve().is_relative_to(args.repo.resolve()):
        raise ValueError("Frozen immutable external output required")
    config = json.loads(args.config.read_text())
    if (config.get("state") != "FROZEN" or tuple(config.get("videos", ())) != VIDEOS
            or tuple(config.get("seeds", ())) != SEEDS):
        raise ValueError("Unexpected immutable patch-attention registry")
    reserve_mount = Path(config["guards"].get("reserve_mount", args.output.parent))
    if shutil.disk_usage(reserve_mount).free < config["guards"]["minimum_free_bytes"]:
        raise OSError("40GiB reserve on external staging mount")
    args.output.mkdir(parents=True)
    trace_root = Path(config["matched_control_trace_root"])
    control_report_path = args.repo / config["matched_control_report"]
    manifest = dict(
        status="RUNNING_GT_FREE_PATCH_ATTENTION_COMPLEMENTARITY_AUDIT",
        experiment_id=config["experiment_id"], config=config,
        config_sha256=sha256(args.config),
        source_commit=subprocess.check_output(
            ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
        source_snapshots=snapshot_sources(args.output / "sources", [
            __file__, args.config, "cfp/native_patch_attention_complementarity.py",
            "cfp/native_local4_complementarity.py", "cfp/provenance.py",
            "cfp/recovery_supervision.py"]),
        matched_control_report_sha256=sha256(control_report_path),
        rows={}, online_physical_labels_loaded=False,
        prediction_hashes_locked=False, patch_banks_verified=False,
    )
    write_json(args.output / "manifest.json", manifest)

    # Verify all accepted control hashes and all patch source/index provenance
    # before importing the offline physical-label reader.
    control_report, controls = _verify_locked_controls(
        config, args.assets_root, trace_root, control_report_path)
    patch_banks = {}
    patch_provenance = {}
    for video in VIDEOS:
        for seed in SEEDS:
            name = f"{video}_s{seed}"
            patch_banks[name], patch_provenance[name] = _load_patch_bank(
                args.assets_root, video, seed)
            manifest["rows"][name] = dict(
                video=video, seed=seed, **controls[name],
                **patch_provenance[name],
                online_GT_teacher_future_used=False,
                cached_detection_or_feature_used_for_inference=False,
                future_or_teacher_online_used=False,
                precomputed_IR_used_for_inference=False,
            )
            write_json(args.output / "manifest.json", manifest)
    manifest["prediction_hashes_locked"] = True
    manifest["patch_banks_verified"] = True
    manifest["status"] = "ALL_18_CONTROL_HASHES_LOCKED_PATCH_BANKS_VERIFIED"
    write_json(args.output / "manifest.json", manifest)

    # Only now may the retrospective reducer open physical labels.
    from cfp.recovery_supervision import physical_bank

    manifest["online_physical_labels_loaded"] = False
    manifest["offline_labels_opened_after_all_prediction_hashes"] = True
    aggregate = _new_counts()
    per_video = collections.defaultdict(_new_counts)
    per_seed = collections.defaultdict(_new_counts)
    for video in VIDEOS:
        physical = physical_bank(video, "train")
        for seed in SEEDS:
            name = f"{video}_s{seed}"
            summary = _summarize_trace(
                Path(controls[name]["trace_path"]), patch_banks[name], physical)
            manifest.setdefault("offline", {})[name] = summary
            aggregate = _add_counts(aggregate, summary["counts"])
            per_video[video] = _add_counts(per_video[video], summary["counts"])
            per_seed[str(seed)] = _add_counts(per_seed[str(seed)], summary["counts"])
            write_json(args.output / "manifest.json", manifest)
    promotion_videos = {}
    for video in VIDEOS:
        counts = per_video[video]
        promotion_videos[video] = dict(
            rows=int(counts["rows"]),
            patch_correction=int(counts["patch_correction"]),
            patch_harm=int(counts["patch_harm"]),
            patch_correction_gt_harm=bool(
                counts["patch_correction"] > counts["patch_harm"]),
            margin_activation=int(counts["margin_activation"]),
            margin_correction=int(counts["margin_correction"]),
            margin_harm=int(counts["margin_harm"]),
            margin_correction_gt_harm=bool(
                counts["margin_correction"] > counts["margin_harm"]),
        )
    promotion_pass = all(
        value["patch_correction_gt_harm"]
        and value["margin_activation"] > 0
        and value["margin_correction_gt_harm"]
        for value in promotion_videos.values())
    manifest["offline"] = dict(
        aggregate_counts=dict(aggregate),
        per_video={key: dict(value) for key, value in per_video.items()},
        per_seed={key: dict(value) for key, value in per_seed.items()},
        promotion_gate=dict(
            required=("Every registered video must have patch-attention correction "
                      "> patch-attention harm and fixed-margin correction > harm "
                      "with at least one activation."),
            per_video=promotion_videos, pass_=promotion_pass),
        decision=("RUN_FIXED_PATCH_ATTENTION_MARGIN_REPLAY" if promotion_pass
                  else "DO_NOT_RUN_PATCH_ATTENTION_COMBINATION_REPLAY"),
    )
    manifest["status"] = "COMPLETE_PATCH_ATTENTION_COMPLEMENTARITY_AUDIT"
    manifest["online_physical_labels_loaded"] = False
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "decision": manifest["offline"]["decision"],
        "aggregate": manifest["offline"]["aggregate_counts"],
        "promotion_gate": manifest["offline"]["promotion_gate"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
