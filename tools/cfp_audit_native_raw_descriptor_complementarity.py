#!/usr/bin/env python3
"""Audit a fixed raw-crop cue on the accepted DINO candidate graph.

The tracker replay is deliberately the accepted source-matched global-DINO
FIFO route.  The raw descriptor is observed on the same causal histories and
current proposals, while the reducer opens physical labels only after every
prediction hash has been checked byte-for-byte against that route.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import runpy
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cfp.native_raw_cls_complementarity import margin_gate, top1_positive
from cfp.predictive import bounded_identity_prototype
from cfp.predictive_association import PredictiveAssociationTracker
from cfp.provenance import sha256, snapshot_sources, write_json
from cfp.recovery_supervision import physical_bank
from yolox.tracker import matching
from yolox.tracker.basetrack import BaseTrack

_RAW_HELPERS = runpy.run_path(str(Path(__file__).with_name(
    "cfp_audit_native_raw_cls_complementarity.py")))
_add_counts = _RAW_HELPERS["_add_counts"]
_costs = _RAW_HELPERS["_costs"]
_new_counts = _RAW_HELPERS["_new_counts"]
_summarize_trace = _RAW_HELPERS["_summarize_trace"]


def _install_observer(tracker, stream, bank):
    if tracker.frame_id != 0:
        raise ValueError("Raw descriptor observer must be installed before frame 1")
    base = type(tracker)

    class ObservedTracker(base):
        def association_cost(self, tracks, detections):
            result = super().association_cost(tracks, detections)
            raw_geometry = matching.iou_distance(tracks, detections)
            geometry = (matching.fuse_score(raw_geometry, detections)
                        if not self.args.mot20 else raw_geometry)
            if result.shape != geometry.shape:
                raise ValueError("Embedding baseline and geometry graph shape drift")
            rows = []
            for row, track in enumerate(tracks):
                observed = _costs(track, detections, bank, self.frame_id,
                                  raw_geometry[row], geometry[row])
                rows.append(dict(track_id=int(track.track_id),
                                 history=observed["history"],
                                 proposals=[int(d.current_proposal_index) for d in detections],
                                 raw_geometry=observed["raw_geometry"].tolist(),
                                 geometry=observed["geometry"].tolist(),
                                 embedding_appearance=observed["embedding"].tolist(),
                                 raw_cls_appearance=observed["raw_cls"].tolist(),
                                 candidate_valid=observed["candidate_valid"].tolist(),
                                 raw_valid=observed["raw_valid"].tolist(),
                                 baseline=np.asarray(result[row], dtype=np.float32).tolist()))
            stream.write(json.dumps({"frame": int(self.frame_id), "rows": rows},
                                    separators=(",", ":"), allow_nan=False) + "\n")
            return result

    tracker.__class__ = ObservedTracker


def _run_one(config, assets_root, descriptor_root, output, video, seed):
    detections_root = assets_root / "phase0" / "causal_v3_train_detections"
    global_root = assets_root / "phase7" / f"native_dino_global_{video}_k4_s{seed}_v1"
    detection_path = detections_root / f"{video}.npz"
    global_manifest = json.loads((global_root / "manifest.json").read_text())
    detection_manifest = json.loads((detections_root / "manifest.json").read_text())
    descriptor_source = config.get("descriptor_source", "raw_descriptor")
    descriptor = descriptor_root / f"native_raw_descriptor_{video}_v1"
    raw_manifest = None
    if descriptor_source == "raw_descriptor":
        raw_manifest = json.loads((descriptor / "manifest.json").read_text())
    elif descriptor_source == "legacy_rgb_postneck_seed0":
        descriptor = descriptor_root
    else:
        raise ValueError(f"Unknown independent descriptor source: {descriptor_source}")
    if (global_manifest.get("status") != "COMPLETE_NATIVE_CURRENT_DINO_FEATURES"
            or global_manifest.get("video") != video
            or global_manifest.get("encoder", {}).get("trained_encoder", {}).get("seed") != seed
            or global_manifest.get("encoder", {}).get("trained_encoder", {}).get("outer") != video
            or sha256(detection_path) != global_manifest["detection_sha256"]
            or sha256(detection_path) != detection_manifest["caches"][video]["sha256"]):
        raise ValueError(f"Source/proposal guard failed for {video} seed {seed}")
    if descriptor_source == "raw_descriptor":
        if (raw_manifest.get("status") != "COMPLETE_NATIVE_RAW_DESCRIPTOR"
                or raw_manifest.get("video") != video
                or sha256(detection_path) != raw_manifest["detection_sha256"]):
            raise ValueError(f"Raw descriptor provenance failed for {video}")
    with np.load(detection_path, allow_pickle=False) as archive:
        frames, offsets, boxes, scores = [archive[name].copy() for name in
                                          ("frame_ids", "offsets", "boxes", "scores")]
    if not np.array_equal(frames, np.load(global_root / "frame_ids.npy")) or not np.array_equal(
            offsets, np.load(global_root / "offsets.npy")):
        raise ValueError("DINO feature/proposal frame or offset drift")
    global_bank = {name: np.load(global_root / f"{name}.npy", mmap_mode="r")
                   for name in ("embedding", "valid")}
    if descriptor_source == "raw_descriptor":
        raw_bank = {"raw_cls": np.load(descriptor / "raw_descriptor.npy", mmap_mode="r"),
                    "valid": np.load(descriptor / "valid.npy", mmap_mode="r")}
        descriptor_manifest_sha256 = sha256(descriptor / "manifest.json")
    else:
        legacy_manifest = json.loads((descriptor / "manifest.json").read_text())
        if (legacy_manifest.get("status") != "COMPLETE"
                or legacy_manifest.get("detection_manifest_sha256") != sha256(detections_root / "manifest.json")
                or legacy_manifest.get("crop_variant") != "original"
                or legacy_manifest.get("embedding_stage") != "post_neck"):
            raise ValueError("Legacy RGB feature manifest is not the fixed causal post-neck bank")
        with np.load(descriptor / f"{video}.npz", allow_pickle=False) as archive:
            raw_bank = {"raw_cls": archive["rgb"], "valid": archive["valid_rgb"]}
        descriptor_manifest_sha256 = sha256(descriptor / "manifest.json")
    if (global_bank["embedding"].shape != (len(boxes), 384)
            or raw_bank["raw_cls"].ndim != 2
            or raw_bank["raw_cls"].shape[0] != len(boxes)
            or not np.array_equal(global_bank["valid"], raw_bank["valid"])):
        raise ValueError("Global/raw descriptor rows are not aligned")
    bank = dict(raw_bank, frame_of_proposal=np.repeat(frames, np.diff(offsets)))

    torch.set_num_threads(2)
    BaseTrack._count = 0
    ordinary = __import__("cfp.native_local_identity", fromlist=["RecentIdentityControl"]).RecentIdentityControl(
        "fifo", config["history_length"], config["maximum_history_age_frames"])
    tracker = PredictiveAssociationTracker(
        SimpleNamespace(track_thresh=config["track_thresh"], track_buffer=config["track_buffer"],
                        match_thresh=config["match_thresh"], mot20=config["mot20"]),
        frame_rate=config["frame_rate"], predictor=ordinary, memory="predictor",
        history_view="recent", appearance_weight=config["appearance_weight"],
        proximity_iou=config["proximity_iou"], ema=.9)
    output.mkdir(parents=True)
    trace = output / "complementarity.jsonl.gz"
    stream = gzip.open(trace, "wt", compresslevel=3)
    _install_observer(tracker, stream, bank)
    prediction = output / f"{video}.txt"
    timings, rows = [], 0
    report = dict(status="RUNNING_GT_FREE_RAW_DESCRIPTOR_COMPLEMENTARITY_AUDIT",
                  video=video, seed=seed,
                  global_feature_manifest_sha256=sha256(global_root / "manifest.json"),
                  descriptor_source=descriptor_source,
                  raw_descriptor_manifest_sha256=descriptor_manifest_sha256,
                  detection_sha256=sha256(detection_path), prediction_sha256=None,
                  trace_sha256=None, online_GT_teacher_future_used=False,
                  loaded_GT_annotation_labels=False, cached_detection_or_feature_used_for_inference=False,
                  future_or_teacher_online_used=False, precomputed_IR_used_for_inference=False)
    write_json(output / "manifest.json", report)
    try:
        with prediction.open("w") as stream_prediction:
            for frame_index, frame in enumerate(frames):
                lo, hi = offsets[frame_index:frame_index + 2]
                started = time.perf_counter()
                targets = tracker.update(np.c_[boxes[lo:hi].copy(), scores[lo:hi]],
                                         (1, 1), (1, 1),
                                         np.array(global_bank["embedding"][lo:hi], copy=True),
                                         proposal_indices=np.arange(lo, hi))
                timings.append(time.perf_counter() - started)
                for track in targets:
                    if np.prod(track.tlwh[2:]) <= 100:
                        continue
                    proposal = track.current_proposal_index
                    if track.frame_id != frame or not lo <= proposal < hi:
                        raise ValueError("Output lost current proposal mapping")
                    measured = boxes[proposal].copy(); measured[2:] -= measured[:2]
                    stream_prediction.write(
                        f"{frame},{int(track.track_id)},{measured[0]:.3f},{measured[1]:.3f},"
                        f"{measured[2]:.3f},{measured[3]:.3f},{float(track.score):.6f},-1,-1,-1\n")
                    rows += 1
        stream.close()
        reference = assets_root / "phase7" / f"native_identity_global_{video}_fifo_s{seed}_v1" / f"{video}.txt"
        if not reference.exists() or prediction.read_bytes() != reference.read_bytes():
            raise ValueError(f"Embedding prediction parity failed for {video} seed {seed}")
        report.update(status="COMPLETE_GT_FREE_RAW_DESCRIPTOR_COMPLEMENTARITY_REPLAY",
                      frames=len(timings), output_rows=rows, prediction_sha256=sha256(prediction),
                      trace_sha256=sha256(trace), prediction_byte_equal=True,
                      association_seconds=float(sum(timings)),
                      association_latency_p50_p95_ms=(1000 * np.percentile(timings, [50, 95])).tolist())
        write_json(output / "manifest.json", report)
        return report
    except Exception as error:
        stream.close()
        report.update(status="FAILED_ACTUAL_EXECUTION", error=str(error))
        write_json(output / "manifest.json", report)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--descriptor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise ValueError("The reducer is CPU-only; GPU feature extraction is a separate step")
    config = json.loads(args.config.read_text())
    videos, seeds = tuple(config["videos"]), tuple(config["seeds"])
    if config.get("state") != "FROZEN" or not videos or not seeds:
        raise ValueError("Frozen nonempty pilot registry required")
    if args.output.exists() or args.output.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError("Immutable external output required")
    reserve_mount = Path(config["guards"]["reserve_mount"])
    if shutil.disk_usage(reserve_mount).free < int(config["guards"]["minimum_free_bytes"]):
        raise OSError("40GiB reserve on the scientific asset mount")
    args.output.mkdir(parents=True)
    source_paths = [__file__, args.config, "cfp/raw_identity_descriptor.py",
                    "tools/cfp_audit_native_raw_cls_complementarity.py",
                    "cfp/native_raw_cls_complementarity.py", "cfp/predictive_association.py",
                    "cfp/predictive.py", "cfp/association.py", "cfp/native_local_identity.py",
                    "cfp/provenance.py", "cfp/recovery_supervision.py"]
    manifest = dict(status="RUNNING_GT_FREE_RAW_DESCRIPTOR_COMPLEMENTARITY_AUDIT",
                    experiment_id=config["experiment_id"], config=config,
                    config_sha256=sha256(args.config), source_commit=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True).strip(),
                    source_snapshots=snapshot_sources(args.output / "sources", source_paths),
                    rows={}, prediction_hashes_locked=False, online_physical_labels_loaded=False)
    write_json(args.output / "manifest.json", manifest)
    for video in videos:
        for seed in seeds:
            name = f"{video}_s{seed}"
            manifest["rows"][name] = _run_one(
                config, args.assets_root, args.descriptor_root, args.output / name, video, seed)
            write_json(args.output / "manifest.json", manifest)
    expected = len(videos) * len(seeds)
    if len(manifest["rows"]) != expected or not all(
            row["prediction_byte_equal"] for row in manifest["rows"].values()):
        raise ValueError("All registered prediction parities are required before labels")
    manifest["prediction_hashes_locked"] = True
    manifest["status"] = "ALL_GT_FREE_RAW_DESCRIPTOR_COMPLEMENTARITY_REPLAYS_COMPLETE"
    write_json(args.output / "manifest.json", manifest)
    aggregate = _new_counts()
    per_video = collections.defaultdict(_new_counts)
    per_seed = collections.defaultdict(_new_counts)
    for video in videos:
        physical = physical_bank(video, "train")
        for seed in seeds:
            name = f"{video}_s{seed}"
            summary = _summarize_trace(args.output / name / "complementarity.jsonl.gz", physical)
            manifest.setdefault("offline", {})[name] = summary
            aggregate = _add_counts(aggregate, summary["counts"])
            per_video[video] = _add_counts(per_video[video], summary["counts"])
            per_seed[str(seed)] = _add_counts(per_seed[str(seed)], summary["counts"])
            write_json(args.output / "manifest.json", manifest)
    promotion_videos = {}
    for video in videos:
        counts = per_video[video]
        promotion_videos[video] = dict(
            rows=int(counts["rows"]), raw_correction=int(counts["raw_correction"]),
            raw_harm=int(counts["raw_harm"]),
            raw_correction_gt_harm=bool(counts["raw_correction"] > counts["raw_harm"]),
            margin_activation=int(counts["margin_activation"]),
            margin_correction=int(counts["margin_correction"]),
            margin_harm=int(counts["margin_harm"]),
            margin_correction_gt_harm=bool(counts["margin_correction"] > counts["margin_harm"]))
    promotion_pass = all(value["raw_correction_gt_harm"]
                          and value["margin_activation"] > 0
                          and value["margin_correction_gt_harm"]
                          for value in promotion_videos.values())
    manifest["offline"] = dict(
        aggregate_counts=dict(aggregate),
        per_video={key: dict(value) for key, value in per_video.items()},
        per_seed={key: dict(value) for key, value in per_seed.items()},
        promotion_gate=dict(required="Every registered pilot video must have raw correction > raw harm and fixed-margin correction > harm with at least one activation.",
                            per_video=promotion_videos, pass_=promotion_pass),
        decision=("EXPAND_RAW_DESCRIPTOR_TO_ALL_FOLDS" if promotion_pass
                  else "DO_NOT_EXPAND_RAW_DESCRIPTOR_OR_RUN_COMBINATION"))
    manifest["status"] = "COMPLETE_RAW_DESCRIPTOR_COMPLEMENTARITY_AUDIT"
    manifest["online_physical_labels_loaded"] = False
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "decision": manifest["offline"]["decision"],
                      "aggregate": manifest["offline"]["aggregate_counts"],
                      "promotion_gate": manifest["offline"]["promotion_gate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
