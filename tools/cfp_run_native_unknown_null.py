#!/usr/bin/env python3
"""Run the fixed candidate-set NULL rule and matched no-rule native controls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch

from cfp.native_unknown_null import install_unknown_preserving_null
from cfp.native_local_identity import RecentIdentityControl
from cfp.predictive_association import PredictiveAssociationTracker
from cfp.provenance import sha256, snapshot_sources, write_json
from yolox.tracker.basetrack import BaseTrack


ASSETS = Path("/data3/liuyeqiang/IR-MFT_assets/cfp")
DATA_ROOT = Path("/data3/liuyeqiang/datasets/mft26-ir-causal-v1")
DERIVED_GT = ASSETS / "protocol_v2" / "gt"
TRACKEVAL = Path("/data3/liuyeqiang/CFP-FishMOT_references/TrackEval")


def _load_inputs(video: str, config: dict):
    detections = ASSETS / "phase0" / "causal_v3_train_detections"
    features = ASSETS / "phase7" / f"native_dino_frozen_{video}_k4_v1"
    dp = detections / f"{video}.npz"
    fm = json.loads((features / "manifest.json").read_text())
    dm = json.loads((detections / "manifest.json").read_text())
    if (fm["status"] != "COMPLETE_NATIVE_CURRENT_DINO_FEATURES" or fm["video"] != video
            or sha256(dp) != fm["detection_sha256"]
            or sha256(dp) != dm["caches"][video]["sha256"]
            or fm["detection_manifest_sha256"] != sha256(detections / "manifest.json")):
        raise ValueError(f"Frozen native DINO inputs required for {video}")
    with np.load(dp, allow_pickle=False) as archive:
        frames, offsets, boxes, scores = [archive[key].copy() for key in
                                          ("frame_ids", "offsets", "boxes", "scores")]
    if (sha256(features / "frame_ids.npy") != fm["frame_ids_sha256"]
            or sha256(features / "offsets.npy") != fm["offsets_sha256"]
            or not np.array_equal(frames, np.load(features / "frame_ids.npy"))
            or not np.array_equal(offsets, np.load(features / "offsets.npy"))):
        raise ValueError(f"Native feature/proposal indexing drift for {video}")
    embedding_path = features / "embedding.npy"
    if sha256(embedding_path) != fm["artifacts"]["embedding"]["sha256"]:
        raise ValueError(f"Native embedding producer drift for {video}")
    embedding = np.load(embedding_path, mmap_mode="r")
    if (embedding.shape != (len(boxes), 384)
            or not np.array_equal(frames, np.arange(1, len(frames) + 1))):
        raise ValueError(f"Incomplete native embedding bank for {video}")
    return detections, features, dp, fm, frames, offsets, boxes, scores, embedding


def _evaluate(video: str, prediction_dir: Path, evaluation: Path, config_path: Path):
    evaluation.mkdir(parents=True)
    protocol = json.loads((Path.cwd() / "configs/cfp/protocol_v2.json").read_text())
    protocol.update(
        train=[video], dev=[], historical_test=[],
        role="Offline official TrackEval scoring of complete six-video candidate-set replay",
        parent_protocol_sha256=sha256(Path.cwd() / "configs/cfp/protocol_v2.json"),
        experiment_config_sha256=sha256(config_path),
    )
    write_json(evaluation / "protocol.json", protocol)
    command = [
        os.environ.get("PYTHON", "/data3/liuyeqiang/.venvs/cfp-fishmot/bin/python"),
        "tools/cfp_evaluate.py", "--data-root", str(DATA_ROOT),
        "--derived-gt-root", str(DERIVED_GT), "--protocol", str(evaluation / "protocol.json"),
        "--partition", "train", "--predictions", str(prediction_dir), "--output", str(evaluation),
        "--trackeval-root", str(TRACKEVAL),
    ]
    completed = subprocess.run(command, cwd=Path.cwd(), text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    metrics = json.loads((evaluation / "metrics.json").read_text())
    if metrics["artifacts"][video]["prediction_sha256"] != sha256(prediction_dir / f"{video}.txt"):
        raise ValueError(f"TrackEval prediction hash mismatch for {video}")
    (evaluation / "stdout.log").write_text(completed.stdout)
    return metrics


def _run_video(video: str, variant: dict, config: dict, config_path: Path, output: Path):
    if output.exists():
        raise ValueError(f"Immutable output already exists: {output}")
    # The per-video directory is created below, so check the already-existing
    # external asset mount rather than a not-yet-created child path.
    if shutil.disk_usage(ASSETS).free < config["guards"]["minimum_free_bytes"]:
        raise OSError("40GiB reserve")
    detections, features, dp, fm, frames, offsets, boxes, scores, embedding = _load_inputs(video, config)
    output.mkdir(parents=True)
    rule = config["null_rule"]
    BaseTrack._count = 0
    torch.set_num_threads(2)
    ordinary = RecentIdentityControl(config["memory"], config["history_length"],
                                     config["maximum_history_age_frames"])
    tracker = PredictiveAssociationTracker(
        SimpleNamespace(track_thresh=config["track_thresh"], track_buffer=config["track_buffer"],
                        match_thresh=config["match_thresh"], mot20=config["mot20"]),
        frame_rate=config["frame_rate"], predictor=ordinary, memory="predictor",
        history_view="recent", appearance_weight=config["appearance_weight"],
        proximity_iou=config["proximity_iou"], ema=.9)
    install_unknown_preserving_null(
        tracker, enabled=variant["enabled"], null_cost=rule["null_cost"],
        ambiguity_margin=rule["ambiguity_margin"], legal_geometry_max=rule["legal_geometry_max"])
    prediction = output / f"{video}.txt"
    report = dict(
        status="RUNNING", experiment_id=config["experiment_id"], variant=variant,
        config_sha256=sha256(config_path), video=video,
        feature_manifest_sha256=sha256(features / "manifest.json"), detection_sha256=sha256(dp),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        sources=snapshot_sources(output / "sources", [__file__, config_path,
            "cfp/native_unknown_null.py", "cfp/native_local_identity.py",
            "cfp/predictive_association.py", "cfp/association.py", "cfp/provenance.py"]),
        online_GT_teacher_future_used=False,
    )
    write_json(output / "manifest.json", report)
    timings, rows = [], 0
    try:
        with prediction.open("w") as stream:
            for index, frame in enumerate(frames):
                lo, hi = offsets[index:index + 2]
                started = time.perf_counter()
                targets = tracker.update(
                    np.c_[boxes[lo:hi].copy(), scores[lo:hi]], (1, 1), (1, 1),
                    np.array(embedding[lo:hi], copy=True), proposal_indices=np.arange(lo, hi))
                for track in targets:
                    if np.prod(track.tlwh[2:]) <= 100:
                        continue
                    proposal = track.current_proposal_index
                    if track.frame_id != frame or not lo <= proposal < hi:
                        raise ValueError("Output lost current proposal mapping")
                    measured = boxes[proposal].copy()
                    measured[2:] -= measured[:2]
                    stream.write(f"{frame},{int(track.track_id)},{measured[0]:.3f},{measured[1]:.3f},"
                                 f"{measured[2]:.3f},{measured[3]:.3f},{float(track.score):.6f},-1,-1,-1\n")
                    rows += 1
                timings.append(time.perf_counter() - started)
                if (index + 1) % 256 == 0:
                    report.update(completed_frames=index + 1, output_rows=rows)
                    write_json(output / "manifest.json", report)
                    if shutil.disk_usage(output).free < config["guards"]["minimum_free_bytes"]:
                        raise OSError("40GiB reserve")
        reference = ASSETS / "phase7" / f"native_identity_frozen_{video}_fifo_s0_v1" / f"{video}.txt"
        parity = prediction.read_bytes() == reference.read_bytes()
        if variant["expected_prediction_parity"] and not parity:
            raise ValueError(f"Matched no-rule control failed frozen prediction parity for {video}")
        report.update(
            status="COMPLETE_NATIVE_UNKNOWN_NULL_REPLAY", frames=len(timings), completed_frames=len(timings),
            output_rows=rows, prediction_sha256=sha256(prediction), reference_sha256=sha256(reference),
            prediction_byte_equal=parity, association_seconds=float(sum(timings)),
            association_latency_p50_p95_ms=(1000 * np.percentile(timings, [50, 95])).tolist(),
            null_rule_counts=tracker.null_rule_counts,
            labels_opened_during_prediction=False,
        )
        write_json(output / "manifest.json", report)
        return report
    except Exception as error:
        report.update(status="FAILED_ACTUAL_EXECUTION", error=str(error))
        write_json(output / "manifest.json", report)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" or args.output.exists()
            or args.output.resolve().is_relative_to(Path.cwd().resolve())):
        raise ValueError("CPU-only immutable external output required")
    config = json.loads(args.config.read_text())
    if config["state"] != "FROZEN" or len(config["variants"]) != 2:
        raise ValueError("Frozen two-arm configuration required")
    args.output.mkdir(parents=True)
    manifest = dict(status="RUNNING_NATIVE_UNKNOWN_NULL_REPLAY", experiment_id=config["experiment_id"],
                    config_sha256=sha256(args.config), variants=config["variants"], videos=config["videos"],
                    online_labels_loaded=False, source_report=config["source_report"], results=[])
    write_json(args.output / "manifest.json", manifest)
    for variant in config["variants"]:
        for video in config["videos"]:
            result = _run_video(video, variant, config, args.config,
                                args.output / variant["name"] / video)
            result["output_dir"] = str(args.output / variant["name"] / video)
            manifest["results"].append(result)
            write_json(args.output / "manifest.json", manifest)
    # Freeze every prediction and reference hash before any official evaluator
    # opens derived physical labels.  This keeps the evaluation boundary
    # separate from the causal online replay.
    if len(manifest["results"]) != len(config["variants"]) * len(config["videos"]):
        raise ValueError("Prediction coverage incomplete before label opening")
    for result in manifest["results"]:
        prediction = Path(result["output_dir"]) / f"{result['video']}.txt"
        if sha256(prediction) != result["prediction_sha256"]:
            raise ValueError(f"Prediction changed before evaluation: {prediction}")
        if result["variant"]["expected_prediction_parity"] and not result["prediction_byte_equal"]:
            raise ValueError(f"Matched control parity missing: {prediction}")
    manifest["prediction_hashes_locked"] = True
    manifest["physical_labels_opened_after_all_prediction_hashes"] = True
    write_json(args.output / "manifest.json", manifest)
    for result in manifest["results"]:
        evaluation = _evaluate(result["video"], Path(result["output_dir"]),
                               Path(result["output_dir"] + ".trackeval"), args.config)
        result.update(official_metrics=evaluation["metrics"][result["video"]],
                      official_metrics_sha256=sha256(Path(result["output_dir"] + ".trackeval") / "metrics.json"),
                      labels_opened_during_prediction=False,
                      labels_opened_after_all_prediction_hashes=True)
        write_json(Path(result["output_dir"]) / "manifest.json", result)
        write_json(args.output / "manifest.json", manifest)
    manifest.update(status="COMPLETE_NATIVE_UNKNOWN_NULL_REPLAY", online_labels_loaded=False,
                    all_variants_complete=True, all_videos_complete=True)
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "results": len(manifest["results"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
