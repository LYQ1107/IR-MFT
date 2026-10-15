#!/usr/bin/env python3
"""Measure causal native appearance/geometry disagreement without changing MOT output.

The first pass runs the frozen DINO384 FIFO route and records its association
inputs before any physical labels are loaded.  Only after every prediction is
byte-equal to the accepted FIFO reference are the proposal labels opened for
retrospective cue and history-consistency summaries.
"""

import argparse
import collections
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch

from cfp.native_cue_observation import close_native_cue_observer, install_native_cue_observer
from cfp.native_local_identity import RecentIdentityControl
from cfp.predictive_association import PredictiveAssociationTracker
from cfp.provenance import sha256, snapshot_sources, write_json
from cfp.recovery_supervision import physical_bank
from cfp.lifecycle_crossover import load_bot_reference
from yolox.tracker.basetrack import BaseTrack


def _quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return None
    return dict(n=int(values.size), mean=float(values.mean()),
                q10_q50_q90=np.quantile(values, [0.1, 0.5, 0.9]).tolist())


def _new_counts():
    return collections.Counter(
        observed_calls=0, observed_track_rows=0, empty_history_rows=0,
        mixed_or_unknown_history_rows=0, clean_history_rows=0,
        clean_rows_with_current_positive=0, candidate_missing_rows=0,
        geometry_top1=0, appearance_top1=0, baseline_top1=0,
        fixed_blend_top1=0, margin_gate_top1=0,
        appearance_correction=0, appearance_harm=0,
        fixed_blend_correction=0, fixed_blend_harm=0,
        margin_gate_activation=0, margin_gate_correction=0,
        margin_gate_harm=0, margin_gate_changed_top=0,
        history_frame_checks=0, candidate_frame_checks=0)


def _rank_and_top(cost, legal, positive):
    if not np.any(legal):
        return None, False
    order = np.argsort(np.where(legal, cost, np.inf), kind="stable")
    order = order[np.isfinite(np.where(legal, cost, np.inf)[order])]
    if not len(order):
        return None, False
    positive_positions = np.flatnonzero(positive[order])
    return (int(positive_positions[0]) + 1 if len(positive_positions) else None,
            bool(positive[order[0]]))


def _margin(cost, legal):
    values = np.sort(np.asarray(cost, dtype=np.float64)[legal])
    return float(values[1] - values[0]) if len(values) >= 2 else None


def _history_consistency(indices, embedding):
    if len(indices) < 2:
        return 1.0
    vectors = np.asarray(embedding[np.asarray(indices, dtype=np.int64)], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    values = vectors @ vectors.T
    tri = values[np.triu_indices(len(vectors), 1)]
    return float(np.mean(tri)) if len(tri) else 1.0


def _summarize_trace(trace_path, physical, embedding):
    labels = physical["labels"]
    frames = physical["frame_of_proposal"]
    counts = _new_counts()
    consistency_values = []
    margin_values = collections.defaultdict(list)
    groups = collections.defaultdict(_new_counts)

    def update_row(target_counts, row, current_frame):
        history = [(int(t), None if index is None else int(index))
                   for t, index in row["history"]]
        known_indices = []
        history_labels = []
        for time_id, index in history:
            if index is None:
                continue
            if not 0 <= index < len(labels) or int(frames[index]) != time_id:
                raise ValueError("Native history proposal/frame alignment drift")
            target_counts["history_frame_checks"] += 1
            known_indices.append(index)
            history_labels.append(int(labels[index]))
        if not known_indices:
            target_counts["empty_history_rows"] += 1
            return
        known = [value for value in history_labels if value >= 0]
        if len(known) != len(history_labels) or len(set(known)) != 1:
            target_counts["mixed_or_unknown_history_rows"] += 1
            return
        target = known[0]
        target_counts["clean_history_rows"] += 1
        consistency = _history_consistency(known_indices[-4:], embedding)
        consistency_values.append(consistency)
        group_name = "consistency_lt_0.80" if consistency < 0.80 else (
            "consistency_0.80_0.90" if consistency < 0.90 else "consistency_ge_0.90")
        group = groups[group_name]
        before_group = target_counts.copy()
        proposals = np.asarray(row["proposals"], dtype=np.int64)
        for index in proposals:
            if not 0 <= index < len(labels) or int(frames[index]) != current_frame:
                raise ValueError("Native current proposal/frame alignment drift")
            target_counts["candidate_frame_checks"] += 1
        candidate_labels = labels[proposals]
        positive = candidate_labels == target
        raw = np.asarray(row["raw_geometry"], dtype=np.float64)
        geometry = np.asarray(row["geometry"], dtype=np.float64)
        appearance = np.asarray(row["appearance"], dtype=np.float64)
        baseline = np.asarray(row["baseline"], dtype=np.float64)
        if not (len(raw) == len(proposals) == len(geometry) == len(appearance) == len(baseline)):
            raise ValueError("Native observed edge dimensions disagree")
        legal = np.isfinite(raw) & (raw <= 0.9)
        if not positive.any():
            target_counts["candidate_missing_rows"] += 1
            return
        target_counts["clean_rows_with_current_positive"] += 1
        if not np.any(positive & legal):
            target_counts["candidate_missing_rows"] += 1
            return
        target_counts["geometry_top1"] += int(_rank_and_top(geometry, legal, positive)[1])
        target_counts["appearance_top1"] += int(_rank_and_top(appearance, legal, positive)[1])
        target_counts["baseline_top1"] += int(_rank_and_top(baseline, legal, positive)[1])
        fixed = geometry.copy()
        fixed[legal] = 0.7 * geometry[legal] + 0.3 * appearance[legal]
        target_counts["fixed_blend_top1"] += int(_rank_and_top(fixed, legal, positive)[1])
        g_rank, g_top = _rank_and_top(geometry, legal, positive)
        a_rank, a_top = _rank_and_top(appearance, legal, positive)
        fixed_rank, fixed_top = _rank_and_top(fixed, legal, positive)
        target_counts["appearance_correction"] += int(not g_top and a_top)
        target_counts["appearance_harm"] += int(g_top and not a_top)
        target_counts["fixed_blend_correction"] += int(not g_top and fixed_top)
        target_counts["fixed_blend_harm"] += int(g_top and not fixed_top)
        g_margin = _margin(geometry, legal)
        a_margin = _margin(appearance, legal)
        if g_margin is not None and a_margin is not None:
            margin_values["geometry"].append(g_margin)
            margin_values["appearance"].append(a_margin)
            margin_values["difference"].append(a_margin - g_margin)
        gate = g_margin is not None and a_margin is not None and a_margin > g_margin
        if gate:
            target_counts["margin_gate_activation"] += 1
            gated = geometry.copy()
            gated[legal] = fixed[legal]
            _, gated_top = _rank_and_top(gated, legal, positive)
            target_counts["margin_gate_top1"] += int(gated_top)
            target_counts["margin_gate_correction"] += int(not g_top and gated_top)
            target_counts["margin_gate_harm"] += int(g_top and not gated_top)
            target_counts["margin_gate_changed_top"] += int(
                np.argmin(np.where(legal, gated, np.inf)) != np.argmin(np.where(legal, geometry, np.inf)))
        else:
            target_counts["margin_gate_top1"] += int(g_top)
        group_keys = ("clean_history_rows", "clean_rows_with_current_positive",
                      "candidate_missing_rows", "geometry_top1", "appearance_top1",
                      "baseline_top1", "fixed_blend_top1", "appearance_correction",
                      "appearance_harm", "fixed_blend_correction", "fixed_blend_harm",
                      "margin_gate_activation", "margin_gate_top1", "margin_gate_correction",
                      "margin_gate_harm", "margin_gate_changed_top")
        for key in group_keys:
            group[key] += target_counts[key] - before_group[key]

    with gzip.open(trace_path, "rt") as stream:
        for line in stream:
            event = json.loads(line)
            counts["observed_calls"] += 1
            frame = int(event["frame"])
            for row in event["rows"]:
                counts["observed_track_rows"] += 1
                update_row(counts, row, frame)
    summary = dict(counts=counts, history_consistency=_quantiles(consistency_values),
                   margins={key: _quantiles(value) for key, value in margin_values.items()},
                   groups={key: dict(value) for key, value in groups.items()})
    return summary


def _run_video(video, config, output, assets, native):
    detections = assets / "phase0" / "causal_v3_train_detections"
    features = assets / "phase7" / f"native_dino_frozen_{video}_k4_v1"
    dp = detections / f"{video}.npz"
    fm = json.loads((features / "manifest.json").read_text())
    dm = json.loads((detections / "manifest.json").read_text())
    if fm["status"] != "COMPLETE_NATIVE_CURRENT_DINO_FEATURES" or fm["video"] != video:
        raise ValueError("Frozen native DINO feature manifest required")
    if sha256(dp) != fm["detection_sha256"] or sha256(dp) != dm["caches"][video]["sha256"]:
        raise ValueError("Detection provenance differs from native DINO feature bank")
    with np.load(dp, allow_pickle=False) as archive:
        frames, offsets, boxes, scores = [archive[key].copy() for key in
                                          ("frame_ids", "offsets", "boxes", "scores")]
    for key, expected in (("frame_ids", fm["frame_ids_sha256"]), ("offsets", fm["offsets_sha256"])):
        if sha256(features / f"{key}.npy") != expected or not np.array_equal(
                np.load(features / f"{key}.npy"), frames if key == "frame_ids" else offsets):
            raise ValueError("Native feature indexing differs from proposals")
    embedding_path = features / "embedding.npy"
    if sha256(embedding_path) != fm["artifacts"]["embedding"]["sha256"]:
        raise ValueError("Native frozen embedding producer drift")
    embedding = np.load(embedding_path, mmap_mode="r")
    if embedding.shape != (len(boxes), 384) or not np.array_equal(frames, np.arange(1, len(frames) + 1)):
        raise ValueError("Complete native 384-D embedding bank required")

    video_output = output / f"native_cue_{video}_v1"
    video_output.mkdir(parents=True)
    trace = video_output / f"{video}.cue.jsonl.gz"
    torch.set_num_threads(2)
    BaseTrack._count = 0
    ordinary = RecentIdentityControl("fifo", config["history_length"],
                                     config["maximum_history_age_frames"])
    tracker = PredictiveAssociationTracker(
        SimpleNamespace(track_thresh=config["track_thresh"], track_buffer=config["track_buffer"],
                        match_thresh=config["match_thresh"], mot20=config["mot20"]),
        frame_rate=config["frame_rate"], predictor=ordinary, memory="predictor",
        history_view="recent", appearance_weight=config["appearance_weight"],
        proximity_iou=config["proximity_iou"], ema=.9)
    install_native_cue_observer(tracker, trace, config["history_length"],
                                 config["maximum_history_age_frames"])
    prediction = video_output / f"{video}.txt"
    timings, rows = [], 0
    with prediction.open("w") as stream:
        for index, frame in enumerate(frames):
            lo, hi = offsets[index:index + 2]
            started = time.perf_counter()
            targets = tracker.update(np.c_[boxes[lo:hi].copy(), scores[lo:hi]],
                                     (1, 1), (1, 1), np.array(embedding[lo:hi], copy=True),
                                     proposal_indices=np.arange(lo, hi))
            for track in targets:
                if np.prod(track.tlwh[2:]) <= 100:
                    continue
                proposal = track.current_proposal_index
                if track.frame_id != frame or not lo <= proposal < hi:
                    raise ValueError("Observed output lost current proposal mapping")
                measured = boxes[proposal].copy()
                measured[2:] -= measured[:2]
                stream.write(f"{frame},{int(track.track_id)},{measured[0]:.3f},{measured[1]:.3f},"
                             f"{measured[2]:.3f},{measured[3]:.3f},{float(track.score):.6f},-1,-1,-1\n")
                rows += 1
            timings.append(time.perf_counter() - started)
            if (index + 1) % 256 == 0 and shutil.disk_usage(output).free < config["guards"]["minimum_free_bytes"]:
                raise OSError("40GiB reserve")
    close_native_cue_observer(tracker)
    reference = assets / "phase7" / f"native_identity_frozen_{video}_fifo_s0_v1" / f"{video}.txt"
    if not reference.exists() or prediction.read_bytes() != reference.read_bytes():
        raise ValueError(f"Frozen FIFO prediction parity failed for {video}")
    return dict(video=video, frames=len(frames), output_rows=rows,
                prediction_sha256=sha256(prediction), reference_sha256=sha256(reference),
                prediction_byte_equal=True, trace_sha256=sha256(trace), trace_path=str(trace),
                association_seconds=float(sum(timings)),
                latency_p50_p95_ms=(1000 * np.percentile(timings, [50, 95])).tolist(),
                feature_manifest_sha256=sha256(features / "manifest.json"),
                detection_sha256=sha256(dp), embedding_sha256=sha256(embedding_path))


def _add_counts(a, b):
    result = collections.Counter(a)
    result.update(b)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise ValueError("Native cue observation is CPU-only")
    config = json.loads(args.config.read_text())
    if config["state"] != "FROZEN" or args.output.exists() or args.output.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError("Frozen immutable external output required")
    assets = Path("/data3/liuyeqiang/IR-MFT_assets/cfp")
    if shutil.disk_usage(assets).free < config["guards"]["minimum_free_bytes"]:
        raise OSError("40GiB reserve")
    native = load_bot_reference("/data3/liuyeqiang/CFP-FishMOT_references/BoT-SORT")
    args.output.mkdir(parents=True)
    report = dict(status="RUNNING_GT_FREE_NATIVE_CUE_OBSERVATION", experiment_id=config["experiment_id"],
                  config=config, config_sha256=sha256(args.config), videos={},
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  source_snapshots=snapshot_sources(args.output / "sources", [
                      __file__, args.config, "cfp/native_cue_observation.py",
                      "cfp/native_local_identity.py", "cfp/predictive_association.py",
                      "cfp/association.py", "cfp/predictive.py", "cfp/recovery_supervision.py",
                      "cfp/lifecycle_crossover.py"]),
                  online_physical_labels_loaded=False, scope=config["scope"])
    write_json(args.output / "manifest.json", report)
    for video in config["videos"]:
        report["videos"][video] = _run_video(video, config, args.output, assets, native)
        write_json(args.output / "manifest.json", report)
        print(video, report["videos"][video], flush=True)
    if not all(value["prediction_byte_equal"] for value in report["videos"].values()):
        raise ValueError("All frozen FIFO prediction parities are required before labels")
    report["status"] = "ALL_SIX_GT_FREE_NATIVE_CUE_OBSERVATIONS_COMPLETE"
    report["all_six_prediction_byte_equal"] = True
    write_json(args.output / "manifest.json", report)

    # Physical labels are intentionally opened only after every GT-free output
    # has been fixed and byte-checked.
    aggregate = _new_counts()
    report["offline"] = dict(status="RUNNING_RETROSPECTIVE_CUE_ANALYSIS",
                              annotation_sha256=None, videos={})
    for video in config["videos"]:
        physical = physical_bank(video, config["partition"])
        report["offline"]["annotation_sha256"] = physical["annotation_sha256"]
        feature_path = assets / "phase7" / f"native_dino_frozen_{video}_k4_v1" / "embedding.npy"
        embedding = np.load(feature_path, mmap_mode="r")
        summary = _summarize_trace(Path(report["videos"][video]["trace_path"]), physical, embedding)
        report["offline"]["videos"][video] = summary
        aggregate = _add_counts(aggregate, summary["counts"])
        write_json(args.output / "manifest.json", report)
        print(video, summary["counts"], flush=True)
    report["offline"]["aggregate_counts"] = dict(aggregate)
    heldout = [config["inner_video"], config["outer_video"]]
    gate_checks = {}
    for video in heldout:
        c = report["offline"]["videos"][video]["counts"]
        gate_checks[video] = dict(activation=int(c["margin_gate_activation"]),
                                  correction=int(c["margin_gate_correction"]),
                                  harm=int(c["margin_gate_harm"]),
                                  correction_gt_harm=bool(c["margin_gate_correction"] > c["margin_gate_harm"]))
    c = report["offline"]["aggregate_counts"]
    gate_checks["aggregate_counts"] = dict(activation=int(c["margin_gate_activation"]),
                                             correction=int(c["margin_gate_correction"]),
                                             harm=int(c["margin_gate_harm"]),
                                             correction_gt_harm=bool(c["margin_gate_correction"] > c["margin_gate_harm"]))
    report["offline"]["predeclared_gate_checks"] = gate_checks
    gate_pass = all(item["activation"] > 0 and item["correction_gt_harm"] for item in gate_checks.values())
    report["offline"]["decision"] = ("RUN_FIXED_MARGIN_NATIVE_REPLAY" if gate_pass
                                       else "DO_NOT_RUN_MARGIN_GATE_NATIVE_REPLAY")
    report["offline"]["status"] = "COMPLETE_RETROSPECTIVE_NATIVE_CUE_DISAGREEMENT_AUDIT"
    report["status"] = "COMPLETE_RETROSPECTIVE_NATIVE_CUE_DISAGREEMENT_AUDIT"
    report["online_physical_labels_loaded"] = False
    report["offline_labels_loaded_after_all_parity"] = True
    write_json(args.output / "manifest.json", report)
    print(json.dumps({"status": report["status"], "decision": report["offline"]["decision"],
                      "gate": gate_checks}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
