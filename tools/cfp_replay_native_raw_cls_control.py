#!/usr/bin/env python3
"""Replay source-matched raw CLS versus the existing global-DINO embedding."""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cfp.native_local_identity import RecentIdentityControl
from cfp.native_raw_cls_control import source_matched_raw_cls
from cfp.predictive_association import PredictiveAssociationTracker
from cfp.provenance import sha256, snapshot_sources, write_json
from yolox.tracker.basetrack import BaseTrack


def _load_inputs(config, features, detections, video, seed):
    if features.name != f"native_dino_global_{video}_k4_s{seed}_v1":
        raise ValueError("Raw-CLS control requires the registered source-matched global bank")
    fm = json.loads((features / "manifest.json").read_text())
    dm = json.loads((detections / "manifest.json").read_text())
    detection_path = detections / f"{video}.npz"
    if (fm.get("status") != "COMPLETE_NATIVE_CURRENT_DINO_FEATURES" or fm.get("video") != video
            or fm.get("encoder", {}).get("trained_encoder") is None
            or fm["encoder"]["trained_encoder"].get("seed") != seed
            or fm["encoder"]["trained_encoder"].get("outer") != video
            or sha256(detection_path) != fm["detection_sha256"]
            or sha256(detection_path) != dm["caches"][video]["sha256"]
            or fm["detection_manifest_sha256"] != sha256(detections / "manifest.json")):
        raise ValueError("Source-matched native bank/proposal provenance mismatch")
    with np.load(detection_path, allow_pickle=False) as cached:
        frames, offsets, boxes, scores = [cached[name].copy() for name in
                                          ("frame_ids", "offsets", "boxes", "scores")]
    if (sha256(features / "frame_ids.npy") != fm["frame_ids_sha256"]
            or sha256(features / "offsets.npy") != fm["offsets_sha256"]
            or not np.array_equal(frames, np.load(features / "frame_ids.npy"))
            or not np.array_equal(offsets, np.load(features / "offsets.npy"))
            or fm["completed_frames"] != len(frames)
            or not np.array_equal(frames, np.arange(1, len(frames) + 1))):
        raise ValueError("Native frame/proposal indexing guard failed")
    bank = {}
    for field in ("raw_cls", "embedding", "valid"):
        path = features / f"{field}.npy"
        if sha256(path) != fm["artifacts"][field]["sha256"]:
            raise ValueError("Native feature artifact drift: " + field)
        bank[field] = np.load(path, mmap_mode="r")
    raw_cls = source_matched_raw_cls(bank)
    if len(raw_cls) != len(boxes) or bank["embedding"].shape != raw_cls.shape:
        raise ValueError("Raw-CLS and matched embedding rows are not aligned")
    bank["frame_of_proposal"] = np.repeat(frames, np.diff(offsets))
    return frames, offsets, boxes, scores, bank, fm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--detections", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    if (config["state"] != "FROZEN" or args.video not in config["videos"]
            or args.seed not in config["seeds"] or config["feature_field"] != "raw_cls"
            or args.output.exists() or os.environ.get("CUDA_VISIBLE_DEVICES") != ""):
        raise ValueError("Immutable CPU-only registered raw-CLS replay required")
    if shutil.disk_usage(args.output.parent).free < config["guards"]["minimum_free_bytes"]:
        raise OSError("40GiB reserve")
    frames, offsets, boxes, scores, bank, feature_manifest = _load_inputs(
        config, args.features, args.detections, args.video, args.seed)
    torch.set_num_threads(2)
    BaseTrack._count = 0
    ordinary = RecentIdentityControl(config["memory"], config["history_length"],
                                     config["maximum_history_age_frames"])
    tracker = PredictiveAssociationTracker(
        SimpleNamespace(track_thresh=config["track_thresh"], track_buffer=config["track_buffer"],
                         match_thresh=config["match_thresh"], mot20=config["mot20"]),
        frame_rate=config["frame_rate"], predictor=ordinary, memory="predictor",
        history_view="recent", appearance_weight=config["appearance_weight"],
        proximity_iou=config["proximity_iou"], ema=.9)
    args.output.mkdir(parents=True)
    prediction = args.output / f"{args.video}.txt"
    report = dict(status="RUNNING", experiment_id=config["experiment_id"], config=config,
                  config_sha256=sha256(args.config), video=args.video, seed=args.seed,
                  memory=config["memory"], feature_field=config["feature_field"],
                  matched_control_field=config["matched_control_field"],
                  feature_manifest_sha256=sha256(args.features / "manifest.json"),
                  detection_sha256=sha256(args.detections / f"{args.video}.npz"),
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  sources=snapshot_sources(args.output / "sources", [
                      __file__, args.config, "cfp/native_raw_cls_control.py",
                      "cfp/native_local_identity.py", "cfp/predictive_association.py",
                      "cfp/predictive.py", "cfp/association.py", "cfp/provenance.py"]),
                  scope=config["scope"], online_GT_teacher_future_used=False,
                  loaded_GT_annotation_labels=False, cached_detection_or_feature_used_for_inference=False,
                  future_or_teacher_online_used=False, precomputed_IR_used_for_inference=False,
                  source_matched_backbone_raw_cls=True)
    write_json(args.output / "manifest.json", report)
    timings, rows = [], 0
    try:
        with prediction.open("w") as stream:
            for index, frame in enumerate(frames):
                lo, hi = offsets[index:index + 2]
                started = time.perf_counter()
                targets = tracker.update(
                    np.c_[boxes[lo:hi].copy(), scores[lo:hi]], (1, 1), (1, 1),
                    np.array(bank[config["feature_field"]][lo:hi], copy=True),
                    proposal_indices=np.arange(lo, hi))
                timings.append(time.perf_counter() - started)
                for track in targets:
                    if np.prod(track.tlwh[2:]) <= 100:
                        continue
                    proposal = track.current_proposal_index
                    if track.frame_id != frame or not lo <= proposal < hi:
                        raise ValueError("Output must use the actual current assigned proposal")
                    measured = boxes[proposal].copy()
                    measured[2:] -= measured[:2]
                    stream.write(f"{frame},{int(track.track_id)},{measured[0]:.3f},{measured[1]:.3f},"
                                 f"{measured[2]:.3f},{measured[3]:.3f},{float(track.score):.6f},-1,-1,-1\n")
                    rows += 1
                if (index + 1) % 256 == 0:
                    report.update(completed_frames=index + 1, output_rows=rows)
                    write_json(args.output / "manifest.json", report)
                    if shutil.disk_usage(args.output.parent).free < config["guards"]["minimum_free_bytes"]:
                        raise OSError("40GiB reserve")
        report.update(status="COMPLETE_NATIVE_SOURCE_MATCHED_RAW_CLS_REPLAY",
                      frames=len(timings), completed_frames=len(timings), output_rows=rows,
                      prediction_sha256=sha256(prediction), association_seconds=float(sum(timings)),
                      association_latency_p50_p95_ms=(1000 * np.percentile(timings, [50, 95])).tolist(),
                      prediction_counts=tracker.prediction_counts,
                      actual_native_history_source=True)
        write_json(args.output / "manifest.json", report)
        print(json.dumps({"video": args.video, "seed": args.seed, "frames": len(timings),
                          "rows": rows, "seconds": sum(timings)}), flush=True)
    except Exception as error:
        report.update(status="FAILED_ACTUAL_EXECUTION", error=str(error))
        write_json(args.output / "manifest.json", report)
        raise


if __name__ == "__main__":
    main()
