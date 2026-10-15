#!/usr/bin/env python3
"""Replay one fixed medoid identity-memory probe on a native video."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cfp.medoid_identity import MedoidIdentityControl
from cfp.predictive_association import PredictiveAssociationTracker
from cfp.provenance import sha256, snapshot_sources, write_json
from yolox.tracker.basetrack import BaseTrack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if (
        config["state"] != "FROZEN"
        or args.video not in config["videos"]
        or args.output.exists()
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise ValueError("Immutable CPU-only medoid probe requires frozen configuration")
    if args.features.name != f"native_dino_frozen_{args.video}_k4_v1":
        raise ValueError("Probe must use the registered frozen DINO K4 bank")
    if args.output.exists():
        raise FileExistsError("Native probe output is immutable")

    feature_manifest = json.loads((args.features / "manifest.json").read_text())
    detection_manifest = json.loads((args.detections / "manifest.json").read_text())
    detection_path = args.detections / f"{args.video}.npz"
    if (
        feature_manifest["status"] != "COMPLETE_NATIVE_CURRENT_DINO_FEATURES"
        or feature_manifest["video"] != args.video
        or sha256(detection_path) != feature_manifest["detection_sha256"]
        or sha256(detection_path) != detection_manifest["caches"][args.video]["sha256"]
        or feature_manifest["detection_manifest_sha256"]
        != sha256(args.detections / "manifest.json")
    ):
        raise ValueError("Native feature/proposal provenance mismatch")
    with np.load(detection_path, allow_pickle=False) as cached:
        frames, offsets, boxes, scores = [
            cached[key].copy() for key in ("frame_ids", "offsets", "boxes", "scores")
        ]
    if (
        sha256(args.features / "frame_ids.npy") != feature_manifest["frame_ids_sha256"]
        or sha256(args.features / "offsets.npy") != feature_manifest["offsets_sha256"]
        or not np.array_equal(frames, np.load(args.features / "frame_ids.npy"))
        or not np.array_equal(offsets, np.load(args.features / "offsets.npy"))
        or feature_manifest["completed_frames"] != len(frames)
        or not np.array_equal(frames, np.arange(1, len(frames) + 1))
    ):
        raise ValueError("Native full-frame indexing guard failed")
    bank = {}
    for name in ("embedding", "raw_cls", "local4", "visibility4", "valid"):
        path = args.features / f"{name}.npy"
        if sha256(path) != feature_manifest["artifacts"][name]["sha256"]:
            raise ValueError("Native feature producer drift")
        bank[name] = np.load(path, mmap_mode="r")
    bank["frame_of_proposal"] = np.repeat(frames, np.diff(offsets))
    if len(bank["embedding"]) != len(boxes) or bank["embedding"].shape[1] != 384:
        raise ValueError("Native embedding shape guard failed")

    torch.set_num_threads(2)
    BaseTrack._count = 0
    ordinary = MedoidIdentityControl(
        config["history_length"], config["maximum_history_age_frames"]
    )
    tracker = PredictiveAssociationTracker(
        SimpleNamespace(
            track_thresh=config["track_thresh"],
            track_buffer=config["track_buffer"],
            match_thresh=config["match_thresh"],
            mot20=config["mot20"],
        ),
        frame_rate=config["frame_rate"],
        predictor=ordinary,
        memory="predictor",
        history_view="recent",
        appearance_weight=config["appearance_weight"],
        proximity_iou=config["proximity_iou"],
        ema=0.9,
    )
    args.output.mkdir(parents=True)
    output_path = args.output / f"{args.video}.txt"
    report = dict(
        status="RUNNING",
        config=config,
        config_sha256=sha256(args.config),
        video=args.video,
        memory="medoid",
        feature_manifest_sha256=sha256(args.features / "manifest.json"),
        detection_sha256=sha256(detection_path),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        sources=snapshot_sources(
            args.output / "sources",
            [
                __file__,
                args.config,
                "cfp/medoid_identity.py",
                "cfp/predictive_association.py",
                "cfp/association.py",
                "cfp/predictive.py",
                "cfp/provenance.py",
            ],
        ),
        scope=config["scope"],
    )
    write_json(args.output / "manifest.json", report)
    timings, rows = [], 0
    try:
        with output_path.open("w") as stream:
            for index, frame in enumerate(frames):
                lo, hi = offsets[index : index + 2]
                start = time.perf_counter()
                targets = tracker.update(
                    np.c_[boxes[lo:hi].copy(), scores[lo:hi]],
                    (1, 1),
                    (1, 1),
                    np.array(bank["embedding"][lo:hi], copy=True),
                    proposal_indices=np.arange(lo, hi),
                )
                for track in targets:
                    if np.prod(track.tlwh[2:]) <= 100:
                        continue
                    proposal = track.current_proposal_index
                    if track.frame_id != frame or not lo <= proposal < hi:
                        raise ValueError("Output must use the actual current assigned proposal")
                    measured = boxes[proposal].copy()
                    measured[2:] -= measured[:2]
                    stream.write(
                        f"{frame},{int(track.track_id)},{measured[0]:.3f},{measured[1]:.3f},"
                        f"{measured[2]:.3f},{measured[3]:.3f},{float(track.score):.6f},-1,-1,-1\n"
                    )
                    rows += 1
                timings.append(time.perf_counter() - start)
                if (index + 1) % 256 == 0:
                    report.update(completed_frames=index + 1, output_rows=rows)
                    write_json(args.output / "manifest.json", report)
        report.update(
            status="COMPLETE_NATIVE_MEDOID_IDENTITY_REPLAY",
            frames=len(timings),
            completed_frames=len(timings),
            output_rows=rows,
            prediction_sha256=sha256(output_path),
            association_seconds=sum(timings),
            association_latency_p50_p95_ms=(1000 * np.percentile(timings, [50, 95])).tolist(),
            prediction_counts=tracker.prediction_counts,
            actual_native_history_source=True,
            online_GT_teacher_future_used=False,
        )
        write_json(args.output / "manifest.json", report)
        print(json.dumps({"video": args.video, "frames": len(timings), "rows": rows}))
    except Exception as error:
        report.update(status="FAILED_ACTUAL_EXECUTION", error=str(error))
        write_json(args.output / "manifest.json", report)
        raise


if __name__ == "__main__":
    main()
