#!/usr/bin/env python3
"""Reduce the external local4 complementarity audit to a compact report."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Compact report must be written once")
    manifest_path = args.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(args.config.read_text())
    rows = manifest.get("rows", {})
    if (manifest.get("status") != "COMPLETE_LOCAL4_COMPLEMENTARITY_AUDIT"
            or not manifest.get("prediction_hashes_locked")
            or not manifest.get("offline_labels_opened_after_all_prediction_hashes")
            or len(rows) != len(config["videos"]) * len(config["seeds"])):
        raise ValueError("Incomplete or unlocked complementarity audit")
    if any(row.get("online_GT_teacher_future_used") is not False
           or row.get("loaded_GT_annotation_labels") is not False
           or row.get("cached_detection_or_feature_used_for_inference") is not False
           or row.get("future_or_teacher_online_used") is not False
           or row.get("precomputed_IR_used_for_inference") is not False
           or row.get("prediction_byte_equal") is not True
           for row in rows.values()):
        raise ValueError("Online guard or parity failure in complementarity audit")
    report = {
        "status": manifest["status"],
        "as_of": "2026-10-16",
        "experiment_id": manifest["experiment_id"],
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": snapshot_sources(args.output.with_suffix(".sources"), [
            __file__, "tools/cfp_audit_native_local4_complementarity.py", args.config,
            "cfp/native_local4_complementarity.py", "cfp/predictive_association.py",
            "cfp/predictive.py", "cfp/association.py", "cfp/native_local_identity.py",
            "cfp/provenance.py", "cfp/recovery_supervision.py"]),
        "scope": config["scope"],
        "coverage": {
            "expected_replays": len(config["videos"]) * len(config["seeds"]),
            "completed_replays": len(rows),
            "all_prediction_hashes_locked": True,
            "physical_labels_opened_after_all_prediction_hashes": True,
            "online_guards_clean": True,
        },
        "per_replay": {
            name: {
                "video": row["video"], "seed": row["seed"],
                "prediction_sha256": row["prediction_sha256"],
                "trace_sha256": row["trace_sha256"],
                "prediction_byte_equal": row["prediction_byte_equal"],
                "frames": row["frames"], "output_rows": row["output_rows"],
            } for name, row in sorted(rows.items())
        },
        "aggregate_counts": manifest["offline"]["aggregate_counts"],
        "per_video_counts": manifest["offline"]["per_video"],
        "per_seed_counts": manifest["offline"]["per_seed"],
        "promotion_gate": manifest["offline"]["promotion_gate"],
        "decision": manifest["offline"]["decision"],
        "finding": (
            "The fixed visibility-weighted local4 cue changes the combined top-1 decision only when it has causal strip support; the per-video counts below are the registered same-graph complementarity evidence. The strict every-video promotion gate decides whether a local4 combination replay is allowed. This is a same-input diagnostic boundary, not a project-wide NO_GO."
        ),
        "next_boundary": (
            "Do not run local4 margin stacking or a video-specific fallback unless the predeclared every-video gate passes; otherwise require a new independently measured signal."
        ),
        "local4_outputs_outside_git": True,
        "goal_status": "IN_PROGRESS_UNACHIEVED",
    }
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "decision": report["decision"],
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
