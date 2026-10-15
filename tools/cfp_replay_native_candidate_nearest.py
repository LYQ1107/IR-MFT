#!/usr/bin/env python3
"""Replay a fixed candidate-conditioned nearest-history native probe."""

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

from cfp.native_candidate_nearest import CandidateNearestAppearanceTracker
from cfp.provenance import sha256, snapshot_sources, write_json
from yolox.tracker.basetrack import BaseTrack


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "features", "detections", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if (config["state"] != "FROZEN" or args.video not in config["videos"]
            or args.output.exists() or os.environ.get("CUDA_VISIBLE_DEVICES") != ""):
        raise ValueError("Frozen CPU-only and immutable candidate-nearest output required")
    if shutil.disk_usage(args.output.parent).free < config["guards"]["minimum_free_bytes"]:
        raise OSError("40GiB reserve")
    feature_manifest = json.loads((args.features / "manifest.json").read_text())
    detection_manifest = json.loads((args.detections / "manifest.json").read_text())
    detection_path = args.detections / (args.video + ".npz")
    if (feature_manifest["status"] != "COMPLETE_NATIVE_CURRENT_DINO_FEATURES"
            or feature_manifest["video"] != args.video
            or sha256(detection_path) != feature_manifest["detection_sha256"]
            or sha256(detection_path) != detection_manifest["caches"][args.video]["sha256"]
            or feature_manifest["detection_manifest_sha256"] != sha256(args.detections / "manifest.json")):
        raise ValueError("Native frozen DINO/proposal provenance mismatch")
    with np.load(detection_path, allow_pickle=False) as cached:
        frames, offsets, boxes, scores = [cached[key].copy() for key in
                                          ("frame_ids", "offsets", "boxes", "scores")]
    if (sha256(args.features / "frame_ids.npy") != feature_manifest["frame_ids_sha256"]
            or sha256(args.features / "offsets.npy") != feature_manifest["offsets_sha256"]
            or not np.array_equal(frames, np.load(args.features / "frame_ids.npy"))
            or not np.array_equal(offsets, np.load(args.features / "offsets.npy"))
            or feature_manifest["completed_frames"] != len(frames)
            or not np.array_equal(frames, np.arange(1, len(frames) + 1))):
        raise ValueError("Native full-frame indexing guard failed")
    embedding_path = args.features / "embedding.npy"
    if sha256(embedding_path) != feature_manifest["artifacts"]["embedding"]["sha256"]:
        raise ValueError("Native embedding producer drift")
    embedding = np.load(embedding_path, mmap_mode="r")
    if len(embedding) != len(boxes) or embedding.shape[1] != 384:
        raise ValueError("Native embedding shape guard failed")
    torch.set_num_threads(2)
    BaseTrack._count = 0
    tracker = CandidateNearestAppearanceTracker(
        SimpleNamespace(track_thresh=config["track_thresh"],
                        track_buffer=config["track_buffer"],
                        match_thresh=config["match_thresh"],
                        mot20=config["mot20"]),
        frame_rate=config["frame_rate"],
        appearance_weight=config["appearance_weight"],
        proximity_iou=config["proximity_iou"],
        appearance_max_distance=config["appearance_max_distance"],
        history_length=config["history_length"],
        maximum_history_age=config["maximum_history_age_frames"])
    args.output.mkdir(parents=True)
    source_files = [__file__, args.config, "cfp/native_candidate_nearest.py",
                    "cfp/association.py", "cfp/provenance.py"]
    report = dict(status="RUNNING", experiment_id=config["experiment_id"],
                  config=config, config_sha256=sha256(args.config), video=args.video,
                  memory="candidate_nearest", appearance_weight=config["appearance_weight"],
                  feature_manifest_sha256=sha256(args.features / "manifest.json"),
                  detection_sha256=sha256(detection_path),
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                        text=True).strip(),
                  sources=snapshot_sources(args.output / "sources", source_files),
                  scope=config["scope"])
    write_json(args.output / "manifest.json", report)
    timings, rows = [], 0
    prediction_path = args.output / (args.video + ".txt")
    try:
        with prediction_path.open("w") as stream:
            for index, frame in enumerate(frames):
                lo, hi = offsets[index:index + 2]
                started = time.perf_counter()
                targets = tracker.update(np.c_[boxes[lo:hi].copy(), scores[lo:hi]],
                                         (1, 1), (1, 1),
                                         np.array(embedding[lo:hi], copy=True),
                                         proposal_indices=np.arange(lo, hi))
                for track in targets:
                    if np.prod(track.tlwh[2:]) <= 100:
                        continue
                    proposal = track.current_proposal_index
                    if track.frame_id != frame or not lo <= proposal < hi:
                        raise ValueError("Output must use actual current assigned proposal")
                    measured = boxes[proposal].copy(); measured[2:] -= measured[:2]
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
        report.update(status="COMPLETE_NATIVE_CANDIDATE_NEAREST_REPLAY",
                      frames=len(timings), completed_frames=len(timings), output_rows=rows,
                      prediction_sha256=sha256(prediction_path),
                      association_seconds=sum(timings),
                      association_latency_p50_p95_ms=(1000 * np.percentile(
                          timings, [50, 95])).tolist(),
                      identity_counts=tracker.decision_counts,
                      candidate_nearest_counts=tracker.candidate_nearest_counts,
                      actual_native_history_source=True,
                      online_GT_teacher_future_used=False)
        write_json(args.output / "manifest.json", report)
        print(json.dumps(dict(video=args.video, frames=len(timings), rows=rows,
                              seconds=sum(timings))), flush=True)
    except Exception as error:
        report.update(status="FAILED_ACTUAL_EXECUTION", error=str(error))
        write_json(args.output / "manifest.json", report)
        raise


if __name__ == "__main__":
    main()
