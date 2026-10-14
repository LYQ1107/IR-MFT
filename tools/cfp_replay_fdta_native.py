#!/usr/bin/env python3
"""Replay one frozen-DINO native video with an FDTA-style adapter."""
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

from cfp.fdta_fish_adapter import FDTAFishAdapter
from cfp.fdta_native import NativeFDTAWrapper
from cfp.native_local_identity import RecentIdentityControl, install_native_local_identity
from cfp.predictive_association import PredictiveAssociationTracker
from cfp.provenance import sha256, snapshot_sources, write_json
from yolox.tracker.basetrack import BaseTrack


def load_bank(root, video, detections):
    directory = root / f"native_dino_frozen_{video}_k4_v1"
    manifest = json.loads((directory / "manifest.json").read_text())
    detection_path = detections / f"{video}.npz"
    detection_manifest = json.loads((detections / "manifest.json").read_text())
    if (manifest["status"] != "COMPLETE_NATIVE_CURRENT_DINO_FEATURES" or manifest["video"] != video
            or sha256(detection_path) != manifest["detection_sha256"]
            or sha256(detection_path) != detection_manifest["caches"][video]["sha256"]):
        raise ValueError("Exact complete frozen DINO native source required")
    with np.load(detection_path, allow_pickle=False) as values:
        frames, offsets, boxes, scores = [values[name].copy() for name in ("frame_ids", "offsets", "boxes", "scores")]
    bank = {name: np.load(directory / (name + ".npy"), mmap_mode="r") for name in
            ("local4", "visibility4", "raw_cls", "embedding", "valid")}
    if len(bank["raw_cls"]) != len(boxes) or bank["raw_cls"].shape[1:] != (384,):
        raise ValueError("Current native DINO proposal bank is not 384-dimensional")
    bank["frame_of_proposal"] = np.repeat(frames, np.diff(offsets))
    bank.update(frames=frames, offsets=offsets, boxes=boxes, scores=scores, manifest=manifest)
    return bank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory", choices=("fifo", "last", "ema"), default="fifo")
    args = parser.parse_args()
    if args.output.exists() or os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", None) or shutil.disk_usage(args.root).free < 40*2**30:
        raise ValueError("New CPU-only outside-Git native output and 40GiB reserve required")
    train_manifest = json.loads((args.training / "manifest.json").read_text())
    if train_manifest["status"] != "COMPLETE" or train_manifest["guard"]["outer"] != "SN-015" or args.video not in ("SN-015", "BT-001"):
        raise ValueError("Held-out complete FDTA adapter and declared video required")
    guard = train_manifest["guard"]
    state = torch.load(args.training / "final.pt", map_location="cpu", weights_only=False)
    classifier = state["model"]["classifier.weight"]
    model = FDTAFishAdapter(classifier.shape[0], 384, 128, guard["mode"])
    model.load_state_dict(state["model"], strict=True)
    wrapper = NativeFDTAWrapper(model).eval()
    bank = load_bank(args.root, args.video, args.detections)
    config = dict(track_thresh=.58, track_buffer=45, match_thresh=.97, frame_rate=25, mot20=False,
                  appearance_weight=.3, proximity_iou=.1, history_length=4, maximum_history_age_frames=1500,
                  scope="Actual causal native frozen-DINO proposal replay with FDTA-style adapter; only accepted strictly-past proposals and current proposals are inputs. GT/physical labels/teacher/future are absent online. This is not official FDTA and excludes raw detector/IR startup timing.")
    BaseTrack._count = 0
    ordinary = RecentIdentityControl(args.memory, config["history_length"], config["maximum_history_age_frames"])
    tracker = PredictiveAssociationTracker(SimpleNamespace(**{k: config[k] for k in ("track_thresh", "track_buffer", "match_thresh", "mot20")}),
        frame_rate=config["frame_rate"], predictor=ordinary, memory="predictor", history_view="recent",
        appearance_weight=config["appearance_weight"], proximity_iou=config["proximity_iou"], ema=.9)
    tracker = install_native_local_identity(tracker, wrapper, bank, 4, 1500, context_identity=True)
    args.output.mkdir(parents=True)
    report = dict(status="RUNNING", video=args.video, memory=args.memory, config=config,
                  training_manifest_sha256=sha256(args.training / "manifest.json"),
                  training_checkpoint_sha256=sha256(args.training / "final.pt"),
                  feature_manifest_sha256=sha256(Path(bank["manifest"]["output"] if "output" in bank["manifest"] else args.root / f"native_dino_frozen_{args.video}_k4_v1" / "manifest.json")),
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  online_GT_teacher_future_used=False,
                  sources=snapshot_sources(args.output / "sources", [__file__, "cfp/fdta_fish_adapter.py", "cfp/fdta_native.py", "cfp/native_local_identity.py", "cfp/predictive_association.py", "cfp/predictive.py", "cfp/association.py", "cfp/provenance.py"]))
    write_json(args.output / "manifest.json", report)
    timings=[]; rows=0; output_path=args.output / f"{args.video}.txt"
    try:
        with output_path.open("w") as stream:
            for index, frame in enumerate(bank["frames"]):
                lo, hi = bank["offsets"][index:index+2]
                started=time.perf_counter()
                targets=tracker.update(np.c_[bank["boxes"][lo:hi].copy(), bank["scores"][lo:hi]], (1,1), (1,1),
                                       np.array(bank["embedding"][lo:hi], copy=True), proposal_indices=np.arange(lo,hi))
                timings.append(time.perf_counter()-started)
                for track in targets:
                    if np.prod(track.tlwh[2:]) <= 100: continue
                    proposal=track.current_proposal_index
                    if track.frame_id != frame or not lo <= proposal < hi: raise ValueError("Native output lost current proposal")
                    measured=bank["boxes"][proposal].copy(); measured[2:] -= measured[:2]
                    stream.write(f"{frame},{int(track.track_id)},{measured[0]:.3f},{measured[1]:.3f},{measured[2]:.3f},{measured[3]:.3f},{float(track.score):.6f},-1,-1,-1\n")
                    rows += 1
                if (index+1) % 256 == 0:
                    report.update(completed_frames=index+1, output_rows=rows); write_json(args.output / "manifest.json", report)
        report.update(status="COMPLETE_NATIVE_FDTA_ADAPTER_REPLAY", frames=len(timings), completed_frames=len(timings), output_rows=rows,
                      prediction_sha256=sha256(output_path), association_seconds=float(sum(timings)),
                      association_latency_p50_p95_ms=(1000*np.percentile(timings, [50,95])).tolist(),
                      decision_counts=tracker.decision_counts, model_counts=getattr(tracker, "local_counts", None))
        write_json(args.output / "manifest.json", report)
        print(json.dumps({"status":report["status"],"video":args.video,"frames":len(timings),"seconds":sum(timings)}))
    except Exception as error:
        report.update(status="FAILED_ACTUAL_EXECUTION", error=str(error)); write_json(args.output / "manifest.json", report); raise


if __name__ == "__main__": main()
