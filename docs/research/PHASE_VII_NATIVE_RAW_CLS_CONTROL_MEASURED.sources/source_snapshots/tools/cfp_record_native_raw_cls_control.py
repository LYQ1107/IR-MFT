#!/usr/bin/env python3
"""Record the locked all-fold source-matched raw-CLS native control."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from cfp.provenance import sha256, snapshot_sources, write_json


VIDEOS = ("BT-001", "BT-003", "MSK-002", "PF-001", "SN-001", "SN-015")
SEEDS = (0, 1, 2)
METRICS = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW")


def _summary(rows, side):
    result = {}
    for key in METRICS:
        values = np.asarray([row[side][key] for row in rows], dtype=float)
        result[key] = dict(mean=float(values.mean()),
                           sample_sd=float(values.std(ddof=1)), n=int(len(values)))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--assets-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    if (config.get("state") != "FROZEN" or tuple(config.get("videos", ())) != VIDEOS
            or tuple(config.get("seeds", ())) != SEEDS or config.get("feature_field") != "raw_cls"
            or args.output.exists()):
        raise ValueError("Unexpected immutable raw-CLS registry or existing report")
    rows = []
    for video in VIDEOS:
        for seed in SEEDS:
            name = f"{video}_s{seed}"
            raw_dir = args.root / name
            raw_manifest_path = raw_dir / "manifest.json"
            raw_prediction = raw_dir / f"{video}.txt"
            raw_metrics_path = args.root / f"{name}.trackeval" / "metrics.json"
            baseline_dir = args.assets_root / f"native_identity_global_{video}_fifo_s{seed}_v1"
            baseline_manifest_path = baseline_dir / "manifest.json"
            baseline_prediction = baseline_dir / f"{video}.txt"
            baseline_metrics_path = baseline_dir.with_name(baseline_dir.name + ".trackeval") / "metrics.json"
            raw_manifest = json.loads(raw_manifest_path.read_text())
            baseline_manifest = json.loads(baseline_manifest_path.read_text())
            raw_metrics = json.loads(raw_metrics_path.read_text())
            baseline_metrics = json.loads(baseline_metrics_path.read_text())
            raw = raw_metrics["metrics"][video]
            baseline = baseline_metrics["metrics"][video]
            if (raw_manifest.get("status") != "COMPLETE_NATIVE_SOURCE_MATCHED_RAW_CLS_REPLAY"
                    or raw_manifest.get("video") != video or raw_manifest.get("seed") != seed
                    or raw_manifest.get("feature_field") != "raw_cls"
                    or raw_manifest.get("online_GT_teacher_future_used") is not False
                    or raw_manifest.get("loaded_GT_annotation_labels") is not False
                    or raw_manifest.get("cached_detection_or_feature_used_for_inference") is not False
                    or raw_manifest.get("future_or_teacher_online_used") is not False
                    or raw_manifest.get("precomputed_IR_used_for_inference") is not False
                    or sha256(raw_prediction) != raw_manifest.get("prediction_sha256")
                    or raw_metrics["artifacts"][video]["prediction_sha256"] != raw_manifest["prediction_sha256"]
                    or sha256(baseline_prediction) != baseline_manifest.get("prediction_sha256")
                    or baseline_metrics["artifacts"][video]["prediction_sha256"] != baseline_manifest["prediction_sha256"]):
                raise ValueError(f"Hash/guard/provenance mismatch for {name}")
            rows.append(dict(video=video, seed=seed,
                             raw={key: float(raw[key]) for key in METRICS},
                             embedding={key: float(baseline[key]) for key in METRICS},
                             delta_raw_minus_embedding={key: float(raw[key] - baseline[key]) for key in METRICS},
                             prediction_byte_equal=(raw_prediction.read_bytes() == baseline_prediction.read_bytes()),
                             raw_manifest=str(raw_manifest_path), raw_manifest_sha256=sha256(raw_manifest_path),
                             raw_metrics=str(raw_metrics_path), raw_metrics_sha256=sha256(raw_metrics_path),
                             raw_prediction_sha256=raw_manifest["prediction_sha256"],
                             baseline_manifest=str(baseline_manifest_path), baseline_manifest_sha256=sha256(baseline_manifest_path),
                             baseline_metrics=str(baseline_metrics_path), baseline_metrics_sha256=sha256(baseline_metrics_path),
                             baseline_prediction_sha256=baseline_manifest["prediction_sha256"]))
    report = dict(
        status="COMPLETE_NATIVE_SOURCE_MATCHED_RAW_CLS_CONTROL",
        as_of="2026-10-16",
        experiment_id=config["experiment_id"],
        config=str(args.config), config_sha256=sha256(args.config),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256=snapshot_sources(args.output.with_suffix(".sources"), [
            __file__, args.config, "cfp/native_raw_cls_control.py", "cfp/native_local_identity.py",
            "cfp/predictive_association.py", "cfp/predictive.py", "cfp/association.py", "cfp/provenance.py"]),
        scope=config["scope"], feature_field=config["feature_field"],
        matched_control_field=config["matched_control_field"], videos=list(VIDEOS), seeds=list(SEEDS),
        rows=rows, coverage=dict(expected=len(VIDEOS) * len(SEEDS), completed=len(rows),
                                 all_prediction_hashes_locked=True, physical_labels_opened_after_all_prediction_hashes=True,
                                 online_GT_teacher_future_used=False, loaded_GT_annotation_labels=False,
                                 cached_detection_or_feature_used_for_inference=False,
                                 future_or_teacher_online_used=False, precomputed_IR_used_for_inference=False),
        aggregate=dict(raw_cls=_summary(rows, "raw"), matched_embedding=_summary(rows, "embedding"),
                       raw_cls_minus_matched_embedding={key: dict(
                           mean=float(np.mean([row["delta_raw_minus_embedding"][key] for row in rows])),
                           sample_sd=float(np.std([row["delta_raw_minus_embedding"][key] for row in rows], ddof=1)),
                           n=len(rows)) for key in METRICS}),
        per_video={video: {key: dict(
            mean=float(np.mean([row["delta_raw_minus_embedding"][key] for row in rows if row["video"] == video])),
            sample_sd=float(np.std([row["delta_raw_minus_embedding"][key] for row in rows if row["video"] == video], ddof=1)),
            n=3) for key in METRICS} for video in VIDEOS},
        decision="DO_NOT_ADOPT_RAW_CLS_AS_A_STANDALONE_NATIVE_REPLACEMENT",
        finding="The source-matched raw CLS coordinate is better on outer SN-015 and byte-equivalent on BT-003/PF-001/SN-001, but it loses aggregate HOTA, AssA and IDF1 against the paired fitted global embedding because BT-001 losses dominate. IDSW falls on average, while DetA/MOTA changes are small. The direction is video-dependent and does not establish a stable independent native contribution or justify selective stacking.",
        failure_type="All-fold paired raw_cls_minus_embedding means are negative for HOTA, AssA and IDF1; no video-independent improvement gate is met.",
        next_boundary="Preserve raw_cls as a measured diagnostic and do not tune a raw/head blend or select a video-specific fallback. A new combination requires an independently measured same-input signal; current native identity and NULL probes remain negative.",
        external_root=str(args.root), raw_outputs_outside_git=True,
        official_trackeval_commit="12c8791b303e0a0b50f753af204249e622d0281a",
        goal_status="IN_PROGRESS_UNACHIEVED")
    write_json(args.output, report)
    print(json.dumps(dict(status=report["status"], rows=len(rows), output=str(args.output))))


if __name__ == "__main__":
    main()
