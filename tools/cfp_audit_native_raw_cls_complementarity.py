#!/usr/bin/env python3
"""Audit raw-CLS/embedding complementarity on the exact native candidate graph.

The first phase replays the existing fitted-embedding FIFO tracker and records
both causal appearance cost vectors for the same tracks, histories and current
candidate proposals.  It never reads physical labels.  After all 18 prediction
hashes pass byte-parity against the accepted source-matched embedding outputs,
the reducer opens labels retrospectively and evaluates a fixed, pre-registered
margin gate.  A failed gate explicitly forbids a new combination replay.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
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


VIDEOS = ("BT-001", "BT-003", "MSK-002", "PF-001", "SN-001", "SN-015")
SEEDS = (0, 1, 2)
METRICS = ("rows", "candidate_missing_rows", "embedding_top1", "raw_top1",
           "raw_correction", "raw_harm", "raw_disagreement",
           "margin_activation", "margin_top1", "margin_correction",
           "margin_harm", "margin_changed_top")


def _new_counts():
    return collections.Counter({key: 0 for key in METRICS})


def _fifo(values, times, query_frame):
    values = np.asarray(values, dtype=np.float32)
    times = np.asarray(times, dtype=np.int64)
    if values.ndim != 2 or times.shape != (len(values),):
        raise ValueError("FIFO history values/times must be aligned")
    if not len(values):
        return None
    keep = times >= int(query_frame) - 1500
    values = values[keep][-4:]
    if not len(values):
        return None
    feature = np.zeros((1, 8, values.shape[1]), dtype=np.float32)
    frames = np.zeros((1, 8), dtype=np.int64)
    valid = np.zeros((1, 8), dtype=bool)
    # checked_history requires a left-aligned, right-padded sequence.
    feature[0, :len(values)] = values
    frames[0, :len(values)] = times[keep][-4:]
    valid[0, :len(values)] = True
    with torch.inference_mode():
        prototype, available = bounded_identity_prototype(
            torch.from_numpy(feature), torch.from_numpy(frames),
            torch.from_numpy(valid), torch.tensor([int(query_frame)]), mode="fifo")
    return prototype[0].numpy() if bool(available[0]) else None


def _history_vectors(track, bank, frame):
    proposals = [(int(t), int(index)) for t, index in track.identity_proposals
                 if frame - 1500 <= int(t)]
    proposals = proposals[-4:]
    if not proposals:
        return [], None, None
    values_by_frame = {int(t): np.asarray(value, dtype=np.float32)
                       for t, value, _ in track.identity_history}
    raw_values, embedding_values, times = [], [], []
    for timestamp, index in proposals:
        if not 0 <= index < len(bank["raw_cls"]):
            raise ValueError("Native identity history proposal index is out of range")
        if int(bank["frame_of_proposal"][index]) != timestamp:
            raise ValueError("Native identity history frame/index alignment drift")
        if timestamp not in values_by_frame:
            raise ValueError("Embedding history and proposal history drift")
        if not bool(bank["valid"][index]):
            raise ValueError("Native identity memory contains an invalid source row")
        embedding_values.append(values_by_frame[timestamp])
        raw_values.append(np.asarray(bank["raw_cls"][index], dtype=np.float32))
        times.append(timestamp)
    return proposals, _fifo(np.asarray(embedding_values), times, frame), _fifo(
        np.asarray(raw_values), times, frame)


def _costs(track, detections, bank, frame, raw_geometry, geometry):
    proposals, embedding_proto, raw_proto = _history_vectors(track, bank, frame)
    embedding_cost = np.ones(len(detections), dtype=np.float64)
    raw_cost = np.ones(len(detections), dtype=np.float64)
    candidate_valid = []
    raw_valid = []
    candidate_indices = []
    for detection in detections:
        index = int(detection.current_proposal_index)
        candidate_indices.append(index)
        candidate_valid.append(detection.current_rgb is not None)
        valid = (0 <= index < len(bank["raw_cls"]) and bool(bank["valid"][index])
                 and np.isfinite(bank["raw_cls"][index]).all()
                 and np.linalg.norm(bank["raw_cls"][index]) > 1e-8)
        raw_valid.append(valid)
    candidate_valid = np.asarray(candidate_valid, dtype=bool)
    raw_valid = np.asarray(raw_valid, dtype=bool)
    if embedding_proto is not None:
        for col, detection in enumerate(detections):
            if detection.current_rgb is not None:
                embedding_cost[col] = float(np.clip(
                    (1.0 - embedding_proto @ detection.current_rgb) / 2.0, 0.0, 1.0))
    if raw_proto is not None:
        for col, index in enumerate(candidate_indices):
            if raw_valid[col]:
                raw_cost[col] = float(np.clip(
                    (1.0 - raw_proto @ bank["raw_cls"][index]) / 2.0, 0.0, 1.0))
    return dict(history=proposals, embedding=embedding_cost, raw_cls=raw_cost,
                candidate_valid=candidate_valid, raw_valid=raw_valid,
                raw_geometry=np.asarray(raw_geometry, dtype=np.float32),
                geometry=np.asarray(geometry, dtype=np.float32))


def _install_observer(tracker, stream, bank, history_length, maximum_history_age):
    if tracker.frame_id != 0:
        raise ValueError("Complementarity observer must be installed before frame 1")
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
    tracker._raw_cls_complementarity_stream = stream


def _run_one(config, assets, output, video, seed):
    detections_root = assets / "phase0" / "causal_v3_train_detections"
    features = assets / "phase7" / f"native_dino_global_{video}_k4_s{seed}_v1"
    detection_path = detections_root / f"{video}.npz"
    feature_manifest = json.loads((features / "manifest.json").read_text())
    detection_manifest = json.loads((detections_root / "manifest.json").read_text())
    if (features.name != f"native_dino_global_{video}_k4_s{seed}_v1"
            or feature_manifest.get("status") != "COMPLETE_NATIVE_CURRENT_DINO_FEATURES"
            or feature_manifest.get("video") != video
            or feature_manifest.get("encoder", {}).get("trained_encoder", {}).get("seed") != seed
            or feature_manifest.get("encoder", {}).get("trained_encoder", {}).get("outer") != video
            or sha256(detection_path) != feature_manifest["detection_sha256"]
            or sha256(detection_path) != detection_manifest["caches"][video]["sha256"]):
        raise ValueError(f"Source-matched global bank/proposal provenance mismatch: {video} s{seed}")
    with np.load(detection_path, allow_pickle=False) as archive:
        frames, offsets, boxes, scores = [archive[name].copy() for name in
                                          ("frame_ids", "offsets", "boxes", "scores")]
    if (not np.array_equal(frames, np.load(features / "frame_ids.npy"))
            or not np.array_equal(offsets, np.load(features / "offsets.npy"))):
        raise ValueError("Global bank frame/proposal index drift")
    bank = {name: np.load(features / f"{name}.npy", mmap_mode="r")
            for name in ("embedding", "raw_cls", "valid")}
    bank["frame_of_proposal"] = np.repeat(frames, np.diff(offsets))
    if (bank["embedding"].shape != bank["raw_cls"].shape
            or bank["embedding"].shape != (len(boxes), 384)
            or len(bank["valid"]) != len(boxes)):
        raise ValueError("Global bank feature rows are not aligned")

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
    _install_observer(tracker, stream, bank, config["history_length"],
                      config["maximum_history_age_frames"])
    prediction = output / f"{video}.txt"
    timings, rows = [], 0
    report = dict(status="RUNNING_GT_FREE_RAW_CLS_COMPLEMENTARITY_AUDIT",
                  video=video, seed=seed,
                  feature_manifest_sha256=sha256(features / "manifest.json"),
                  detection_sha256=sha256(detection_path),
                  prediction_sha256=None, trace_sha256=None,
                  online_GT_teacher_future_used=False, loaded_GT_annotation_labels=False,
                  cached_detection_or_feature_used_for_inference=False,
                  future_or_teacher_online_used=False, precomputed_IR_used_for_inference=False)
    with prediction.open("w") as stream_prediction:
        for frame_index, frame in enumerate(frames):
            lo, hi = offsets[frame_index:frame_index + 2]
            started = time.perf_counter()
            targets = tracker.update(np.c_[boxes[lo:hi].copy(), scores[lo:hi]],
                                     (1, 1), (1, 1),
                                     np.array(bank["embedding"][lo:hi], copy=True),
                                     proposal_indices=np.arange(lo, hi))
            timings.append(time.perf_counter() - started)
            for track in targets:
                if np.prod(track.tlwh[2:]) <= 100:
                    continue
                proposal = track.current_proposal_index
                if track.frame_id != frame or not lo <= proposal < hi:
                    raise ValueError("Output lost current proposal mapping")
                measured = boxes[proposal].copy()
                measured[2:] -= measured[:2]
                stream_prediction.write(
                    f"{frame},{int(track.track_id)},{measured[0]:.3f},{measured[1]:.3f},"
                    f"{measured[2]:.3f},{measured[3]:.3f},{float(track.score):.6f},-1,-1,-1\n")
                rows += 1
    stream.close()
    reference = assets / "phase7" / f"native_identity_global_{video}_fifo_s{seed}_v1" / f"{video}.txt"
    if not reference.exists() or prediction.read_bytes() != reference.read_bytes():
        raise ValueError(f"Embedding prediction parity failed for {video} s{seed}")
    report.update(status="COMPLETE_GT_FREE_RAW_CLS_COMPLEMENTARITY_REPLAY",
                  frames=len(frames), output_rows=rows,
                  prediction_sha256=sha256(prediction), trace_sha256=sha256(trace),
                  prediction_byte_equal=True,
                  association_seconds=float(sum(timings)),
                  association_latency_p50_p95_ms=(1000 * np.percentile(timings, [50, 95])).tolist())
    write_json(output / "manifest.json", report)
    return report


def _summarize_trace(trace_path, physical):
    labels = np.asarray(physical["labels"])
    frames = np.asarray(physical["frame_of_proposal"])
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
                    if not 0 <= int(index) < len(labels) or int(frames[int(index)]) != int(timestamp):
                        raise ValueError("Retrospective history frame/index drift")
                    history_labels.append(int(labels[int(index)]))
                known = [value for value in history_labels if value >= 0]
                if len(known) != len(history_labels) or len(set(known)) != 1:
                    continue
                proposals = np.asarray(row["proposals"], dtype=np.int64)
                if any(not 0 <= int(index) < len(labels) or int(frames[int(index)]) != frame
                       for index in proposals):
                    raise ValueError("Retrospective current proposal/frame drift")
                candidate_labels = labels[proposals]
                positive = candidate_labels == known[0]
                raw_geometry = np.asarray(row["raw_geometry"], dtype=np.float64)
                geometry = np.asarray(row["geometry"], dtype=np.float64)
                embedding = np.asarray(row["embedding_appearance"], dtype=np.float64)
                raw_cls = np.asarray(row["raw_cls_appearance"], dtype=np.float64)
                baseline = np.asarray(row["baseline"], dtype=np.float64)
                valid = np.asarray(row["candidate_valid"], dtype=bool)
                raw_valid = np.asarray(row["raw_valid"], dtype=bool)
                if not all(len(value) == len(proposals) for value in
                           (raw_geometry, geometry, embedding, raw_cls, baseline, valid, raw_valid)):
                    raise ValueError("Retrospective edge dimensions disagree")
                legal = valid & np.isfinite(raw_geometry) & (raw_geometry <= 0.9)
                if not np.any(positive & legal):
                    counts["candidate_missing_rows"] += 1
                    continue
                counts["rows"] += 1
                embedding_combined = geometry.copy()
                embedding_combined[legal] = .7 * geometry[legal] + .3 * embedding[legal]
                raw_legal = legal & raw_valid
                raw_combined = geometry.copy()
                raw_combined[raw_legal] = .7 * geometry[raw_legal] + .3 * raw_cls[raw_legal]
                _, embedding_top = top1_positive(embedding_combined, legal, positive)
                _, raw_top = top1_positive(raw_combined, raw_legal, positive)
                _, baseline_top = top1_positive(baseline, legal, positive)
                # The baseline is the accepted tracker result; the recomputed
                # embedding combined vector is retained as a parity diagnostic.
                if embedding_top != baseline_top:
                    raise ValueError("Recomputed embedding graph differs from accepted baseline")
                counts["embedding_top1"] += int(embedding_top)
                counts["raw_top1"] += int(raw_top)
                counts["raw_correction"] += int(not embedding_top and raw_top)
                counts["raw_harm"] += int(embedding_top and not raw_top)
                counts["raw_disagreement"] += int(embedding_top != raw_top)
                use_raw = margin_gate(embedding, raw_cls, legal & raw_valid)
                if use_raw:
                    counts["margin_activation"] += 1
                    selected = raw_combined
                    selected_legal = raw_legal
                else:
                    selected = embedding_combined
                    selected_legal = legal
                _, margin_top = top1_positive(selected, selected_legal, positive)
                counts["margin_top1"] += int(margin_top)
                counts["margin_correction"] += int(not embedding_top and margin_top)
                counts["margin_harm"] += int(embedding_top and not margin_top)
                counts["margin_changed_top"] += int(margin_top != embedding_top)
    return dict(counts=dict(counts), promotion=dict(
        raw_correction_gt_harm=bool(counts["raw_correction"] > counts["raw_harm"]),
        margin_activation_positive=bool(counts["margin_activation"] > 0),
        margin_correction_gt_harm=bool(counts["margin_correction"] > counts["margin_harm"])))


def _add_counts(left, right):
    out = collections.Counter(left)
    out.update(right)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise ValueError("This audit is CPU-only")
    if args.output.exists() or args.output.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError("Frozen immutable external output required")
    config = json.loads(args.config.read_text())
    if (config.get("state") != "FROZEN" or tuple(config.get("videos", ())) != VIDEOS
            or tuple(config.get("seeds", ())) != SEEDS):
        raise ValueError("Unexpected immutable complementarity registry")
    reserve_mount = Path(config["guards"].get("reserve_mount", args.output.parent))
    if shutil.disk_usage(reserve_mount).free < config["guards"]["minimum_free_bytes"]:
        raise OSError("40GiB reserve on external staging mount")
    args.output.mkdir(parents=True)
    manifest = dict(status="RUNNING_GT_FREE_RAW_CLS_COMPLEMENTARITY_AUDIT",
                    experiment_id=config["experiment_id"], config=config,
                    config_sha256=sha256(args.config), source_commit=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True).strip(),
                    source_snapshots=snapshot_sources(args.output / "sources", [
                        __file__, args.config, "cfp/native_raw_cls_complementarity.py",
                        "cfp/predictive_association.py", "cfp/predictive.py",
                        "cfp/association.py", "cfp/native_local_identity.py",
                        "cfp/provenance.py", "cfp/recovery_supervision.py"]),
                    rows={}, online_physical_labels_loaded=False,
                    prediction_hashes_locked=False)
    write_json(args.output / "manifest.json", manifest)

    # All replays are label-free.  Only accepted source-matched embedding
    # predictions are written, while both appearance vectors are observed.
    for video in VIDEOS:
        for seed in SEEDS:
            name = f"{video}_s{seed}"
            manifest["rows"][name] = _run_one(
                config, args.assets_root, args.output / name, video, seed)
            write_json(args.output / "manifest.json", manifest)
    if len(manifest["rows"]) != len(VIDEOS) * len(SEEDS) or not all(
            row["prediction_byte_equal"] for row in manifest["rows"].values()):
        raise ValueError("All 18 prediction parities are required before labels")
    manifest["prediction_hashes_locked"] = True
    manifest["status"] = "ALL_18_GT_FREE_RAW_CLS_COMPLEMENTARITY_REPLAYS_COMPLETE"
    write_json(args.output / "manifest.json", manifest)

    # The only label access occurs after every replay hash and parity check.
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
                args.output / name / "complementarity.jsonl.gz", physical)
            manifest.setdefault("offline", {})[name] = summary
            aggregate = _add_counts(aggregate, summary["counts"])
            per_video[video] = _add_counts(per_video[video], summary["counts"])
            per_seed[str(seed)] = _add_counts(per_seed[str(seed)], summary["counts"])
            write_json(args.output / "manifest.json", manifest)
    promotion_videos = {}
    for video in VIDEOS:
        c = per_video[video]
        promotion_videos[video] = dict(
            rows=int(c["rows"]), raw_correction=int(c["raw_correction"]),
            raw_harm=int(c["raw_harm"]), raw_correction_gt_harm=bool(
                c["raw_correction"] > c["raw_harm"]),
            margin_activation=int(c["margin_activation"]),
            margin_correction=int(c["margin_correction"]),
            margin_harm=int(c["margin_harm"]), margin_correction_gt_harm=bool(
                c["margin_correction"] > c["margin_harm"]))
    promotion_pass = all(value["raw_correction_gt_harm"]
                          and value["margin_activation"] > 0
                          and value["margin_correction_gt_harm"]
                          for value in promotion_videos.values())
    manifest["offline"] = dict(aggregate_counts=dict(aggregate),
                                per_video={key: dict(value) for key, value in per_video.items()},
                                per_seed={key: dict(value) for key, value in per_seed.items()},
                                promotion_gate=dict(required="Every registered video must have raw correction > raw harm and fixed-margin correction > harm with at least one activation.",
                                                    per_video=promotion_videos,
                                                    pass_=promotion_pass),
                                decision=("RUN_FIXED_RAW_CLS_MARGIN_REPLAY" if promotion_pass
                                          else "DO_NOT_RUN_RAW_CLS_COMBINATION_REPLAY"))
    manifest["status"] = "COMPLETE_RAW_CLS_COMPLEMENTARITY_AUDIT"
    manifest["online_physical_labels_loaded"] = False
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"],
                      "decision": manifest["offline"]["decision"],
                      "aggregate": manifest["offline"]["aggregate_counts"],
                      "promotion_gate": manifest["offline"]["promotion_gate"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
