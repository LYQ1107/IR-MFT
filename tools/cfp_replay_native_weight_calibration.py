#!/usr/bin/env python3
"""Replay one frozen native DINO video at one pre-registered blend weight."""
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

from cfp.native_local_identity import RecentIdentityControl
from cfp.predictive_association import PredictiveAssociationTracker
from cfp.provenance import sha256, snapshot_sources, write_json
from yolox.tracker.basetrack import BaseTrack


def main():
    ap = argparse.ArgumentParser()
    for name in ("config", "features", "detections", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--appearance-weight", type=float, required=True)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    weight = float(args.appearance_weight)
    if (config["state"] != "FROZEN" or args.video not in config["videos"]
            or weight not in config["candidate_weights"] or args.output.exists()
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""):
        raise ValueError("Frozen candidate, CPU-only and immutable output required")
    if shutil.disk_usage(args.output.parent).free < config["guards"]["minimum_free_bytes"]:
        raise OSError("40GiB reserve")
    fm = json.loads((args.features / "manifest.json").read_text())
    dm = json.loads((args.detections / "manifest.json").read_text())
    detection_path = args.detections / (args.video + ".npz")
    if (fm["status"] != "COMPLETE_NATIVE_CURRENT_DINO_FEATURES"
            or fm["video"] != args.video
            or sha256(detection_path) != fm["detection_sha256"]
            or sha256(detection_path) != dm["caches"][args.video]["sha256"]
            or fm["detection_manifest_sha256"] != sha256(args.detections / "manifest.json")):
        raise ValueError("Complete native frozen DINO features and unchanged proposals required")
    with np.load(detection_path, allow_pickle=False) as arrays:
        frames, offsets, boxes, scores = [arrays[key].copy() for key in
                                          ("frame_ids", "offsets", "boxes", "scores")]
    if (sha256(args.features / "frame_ids.npy") != fm["frame_ids_sha256"]
            or sha256(args.features / "offsets.npy") != fm["offsets_sha256"]
            or not np.array_equal(frames, np.load(args.features / "frame_ids.npy"))
            or not np.array_equal(offsets, np.load(args.features / "offsets.npy"))):
        raise ValueError("Native feature/proposal full-frame indexing differs")
    embedding_path = args.features / "embedding.npy"
    if sha256(embedding_path) != fm["artifacts"]["embedding"]["sha256"]:
        raise ValueError("Current feature producer drift")
    embedding = np.load(embedding_path, mmap_mode="r")
    if (len(embedding) != len(boxes) or embedding.shape[1] != 384
            or fm["completed_frames"] != len(frames)
            or not np.array_equal(frames, np.arange(1, len(frames) + 1))):
        raise ValueError("Every native frame/proposal/empty state must be preserved")

    torch.set_num_threads(2)
    BaseTrack._count = 0
    ordinary = RecentIdentityControl(config["memory"], config["history_length"],
                                     config["maximum_history_age_frames"])
    tracker = PredictiveAssociationTracker(
        SimpleNamespace(track_thresh=config["track_thresh"],
                        track_buffer=config["track_buffer"],
                        match_thresh=config["match_thresh"],
                        mot20=config["mot20"]),
        frame_rate=config["frame_rate"], predictor=ordinary, memory="predictor",
        history_view="recent", appearance_weight=weight,
        proximity_iou=config["proximity_iou"], ema=0.9,
        appearance_max_distance=config["appearance_max_distance"])
    args.output.mkdir(parents=True)
    sources = [__file__, args.config, "cfp/native_local_identity.py",
               "cfp/predictive_association.py", "cfp/association.py",
               "cfp/predictive.py", "cfp/provenance.py"]
    report = dict(status="RUNNING", experiment_id=config["experiment_id"],
                  config=config, config_sha256=sha256(args.config), video=args.video,
                  appearance_weight=weight, memory=config["memory"],
                  feature_manifest_sha256=sha256(args.features / "manifest.json"),
                  detection_sha256=sha256(detection_path),
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                        text=True).strip(),
                  sources=snapshot_sources(args.output / "sources", sources),
                  scope=config["scope"])
    write_json(args.output / "manifest.json", report)
    timings, rows = [], 0
    prediction_path = args.output / (args.video + ".txt")
    try:
        with prediction_path.open("w") as stream:
            for index, frame in enumerate(frames):
                lo, hi = offsets[index:index + 2]
                started = time.perf_counter()
                targets = tracker.update(
                    np.c_[boxes[lo:hi].copy(), scores[lo:hi]], (1, 1), (1, 1),
                    np.array(embedding[lo:hi], copy=True),
                    proposal_indices=np.arange(lo, hi))
                for track in targets:
                    if np.prod(track.tlwh[2:]) <= 100:
                        continue
                    proposal = track.current_proposal_index
                    if track.frame_id != frame or not lo <= proposal < hi:
                        raise ValueError("Confirmed output must use the actual current proposal")
                    measured = boxes[proposal].copy()
                    measured[2:] -= measured[:2]
                    stream.write(f"{frame},{int(track.track_id)},{measured[0]:.3f},"
                                 f"{measured[1]:.3f},{measured[2]:.3f},{measured[3]:.3f},"
                                 f"{float(track.score):.6f},-1,-1,-1\n")
                    rows += 1
                timings.append(time.perf_counter() - started)
                if (index + 1) % 256 == 0:
                    report.update(completed_frames=index + 1, output_rows=rows)
                    write_json(args.output / "manifest.json", report)
                    if shutil.disk_usage(args.output).free < config["guards"]["minimum_free_bytes"]:
                        raise OSError("40GiB reserve")
        report.update(status="COMPLETE_NATIVE_WEIGHT_REPLAY", frames=len(timings),
                      completed_frames=len(timings), output_rows=rows,
                      prediction_sha256=sha256(prediction_path),
                      association_seconds=sum(timings),
                      association_latency_p50_p95_ms=(1000 * np.percentile(
                          timings, [50, 95])).tolist(),
                      identity_counts=tracker.decision_counts,
                      prediction_counts=tracker.prediction_counts,
                      actual_native_history_source=True,
                      online_GT_teacher_future_used=False)
        write_json(args.output / "manifest.json", report)
        print(json.dumps(dict(video=args.video, appearance_weight=weight,
                              frames=len(timings), rows=rows,
                              seconds=sum(timings))), flush=True)
    except Exception as error:
        report.update(status="FAILED_ACTUAL_EXECUTION", error=str(error))
        write_json(args.output / "manifest.json", report)
        raise


if __name__ == "__main__":
    main()
