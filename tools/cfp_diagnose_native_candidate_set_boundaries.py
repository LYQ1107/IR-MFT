#!/usr/bin/env python3
"""Audit candidate-set/unknown boundaries from already accepted native traces."""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import shutil

from cfp.native_candidate_set_audit import new_counts, summarize_trace
from cfp.provenance import sha256, snapshot_sources, write_json
from cfp.recovery_supervision import physical_bank


def _add_counts(left, right):
    result = collections.Counter(left)
    result.update(right)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("Frozen immutable external output required")
    config = json.loads(args.config.read_text())
    assets = Path("/data3/liuyeqiang/IR-MFT_assets/cfp")
    if shutil.disk_usage(assets).free < config["guards"]["minimum_free_bytes"]:
        raise OSError("40GiB reserve")
    source_report = json.loads((Path.cwd() / config["source_report"]).read_text())
    if source_report["status"] != "COMPLETE_RETROSPECTIVE_NATIVE_CUE_DISAGREEMENT_AUDIT":
        raise ValueError("Accepted native cue report required")

    # Verify every trace and prediction first. No physical labels are opened in
    # this phase, so the later label load cannot alter the acceptance boundary.
    for video in config["videos"]:
        parity = source_report["parity"]["videos"][video]
        video_root = Path(config["external_observation_root"]) / f"native_cue_{video}_v1"
        trace = video_root / f"{video}.cue.jsonl.gz"
        prediction = video_root / f"{video}.txt"
        reference = assets / "phase7" / f"native_identity_frozen_{video}_fifo_s0_v1" / f"{video}.txt"
        if sha256(trace) != parity["trace_sha256"] or sha256(prediction) != parity["prediction_sha256"]:
            raise ValueError(f"Accepted native artifact hash mismatch for {video}")
        if not reference.exists() or prediction.read_bytes() != reference.read_bytes():
            raise ValueError(f"Accepted native prediction parity mismatch for {video}")

    args.output.mkdir(parents=True)
    manifest = {
        "status": "RUNNING_RETROSPECTIVE_NATIVE_CANDIDATE_SET_BOUNDARIES",
        "experiment_id": config["experiment_id"],
        "config": config,
        "config_sha256": sha256(args.config),
        "source_report_sha256": sha256(Path.cwd() / config["source_report"]),
        "online_physical_labels_loaded": False,
        "trace_and_prediction_hashes_verified": True,
        "source_snapshots": snapshot_sources(args.output / "sources", [
            __file__, args.config, "cfp/native_candidate_set_audit.py",
            "cfp/recovery_supervision.py", "cfp/supervision.py"]),
        "videos": {},
    }
    write_json(args.output / "manifest.json", manifest)

    # Physical labels are intentionally loaded only after all six trace and
    # prediction hashes have passed.
    aggregate = new_counts()
    manifest["offline_labels_loaded_after_all_hashes"] = True
    for video in config["videos"]:
        physical = physical_bank(video, "train")
        trace = Path(config["external_observation_root"]) / f"native_cue_{video}_v1" / f"{video}.cue.jsonl.gz"
        summary = summarize_trace(trace, physical, config["raw_geometry_legal_max"])
        manifest["videos"][video] = summary
        aggregate = _add_counts(aggregate, summary["counts"])
        write_json(args.output / "manifest.json", manifest)
    manifest["aggregate_counts"] = dict(aggregate)
    per_video_support = {
        video: bool(manifest["videos"][video]["counts"]["competition_rows"] > 0)
        for video in config["videos"]
    }
    manifest["support_gate"] = {
        "all_videos_have_known_competition_rows": all(per_video_support.values()),
        "per_video": per_video_support,
        "clean_positive_unknown_competition_rows": int(aggregate["clean_positive_unknown_competition_rows"]),
        "clean_positive_other_identity_competition_rows": int(aggregate["clean_positive_other_identity_competition_rows"]),
    }
    manifest["decision"] = "DO_NOT_RUN_SET_COMPETITION_NATIVE_REPLAY"
    manifest["status"] = "COMPLETE_RETROSPECTIVE_NATIVE_CANDIDATE_SET_BOUNDARIES"
    manifest["online_physical_labels_loaded"] = False
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "decision": manifest["decision"],
                      "aggregate_counts": manifest["aggregate_counts"],
                      "support_gate": manifest["support_gate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
