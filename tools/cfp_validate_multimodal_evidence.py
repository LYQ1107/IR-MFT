#!/usr/bin/env python3
"""Audit the frozen RGB--IR evidence without allocating new experiment assets.

This validator is deliberately read-only with respect to the asset store.  It
recomputes selected three-seed means/deltas from the registered campaign
summaries, checks every referenced prediction/evaluation manifest, and audits
the nine frozen CMUE/PLIM composition controls.  It does not train, replay, or
write under ``IR-MFT_assets``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = Path("/data3/liuyeqiang/IR-MFT_assets/cfp")
METRICS = ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing JSON evidence: {path}")
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected object JSON: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-7) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def metrics(value: dict[str, Any]) -> dict[str, float | int]:
    return {name: value[name] for name in METRICS}


def command_arg(command: list[str], name: str) -> Path:
    try:
        return Path(command[command.index(name) + 1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"Missing {name} in registered command") from exc


def audit_replay(output: Path, evaluation: Path, expected: dict[str, Any]) -> dict[str, Any]:
    """Check one existing prediction/evaluation pair and return small provenance."""
    manifest_path = output / "manifest.json"
    evaluation_path = evaluation / "metrics.json"
    manifest = read_json(manifest_path)
    measured = read_json(evaluation_path)
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError(f"Incomplete prediction manifest: {manifest_path}")
    actual = measured.get("metrics", {}).get("COMBINED_SEQ")
    if not isinstance(actual, dict):
        raise RuntimeError(f"Missing COMBINED_SEQ metrics: {evaluation_path}")
    for name in METRICS:
        if not close(actual[name], expected[name]):
            raise RuntimeError(
                f"Metric mismatch for {output.name} {name}: "
                f"{actual[name]} != {expected[name]}"
            )
    artifact_hashes: dict[str, str] = {}
    artifacts = measured.get("artifacts", {})
    for sequence, info in manifest.get("videos", {}).items():
        prediction = output / f"{sequence}.txt"
        if not prediction.is_file():
            raise RuntimeError(f"Missing prediction file: {prediction}")
        prediction_hash = sha256(prediction)
        if prediction_hash != info.get("prediction_sha256"):
            raise RuntimeError(f"Prediction hash mismatch: {prediction}")
        artifact = artifacts.get(sequence, {})
        if artifact.get("prediction_sha256") != prediction_hash:
            raise RuntimeError(f"Evaluation artifact hash mismatch: {evaluation_path}")
        artifact_hashes[sequence] = prediction_hash
    return {
        "output": str(output),
        "evaluation": str(evaluation),
        "manifest_sha256": sha256(manifest_path),
        "evaluation_sha256": sha256(evaluation_path),
        "prediction_sha256": artifact_hashes,
        "metrics": metrics(actual),
    }


def audit_campaign_summary(summary_path: Path, assets_root: Path) -> tuple[int, list[dict[str, Any]]]:
    """Audit all runs in one immutable campaign summary."""
    summary = read_json(summary_path)
    if summary.get("status") != "COMPLETE":
        raise RuntimeError(f"Campaign is not complete: {summary_path}")
    results: list[dict[str, Any]] = []
    for run_id, run in sorted(summary.get("runs", {}).items()):
        commands = run.get("commands", [])
        if len(commands) < 2:
            raise RuntimeError(f"Missing replay/evaluation commands: {summary_path} {run_id}")
        output = command_arg(commands[0], "--output")
        evaluation = command_arg(commands[1], "--output")
        # Registered commands are absolute in the frozen summaries.  Resolve
        # relative commands against the repository only as a compatibility aid.
        if not output.is_absolute():
            output = ROOT / output
        if not evaluation.is_absolute():
            evaluation = ROOT / evaluation
        if not output.is_relative_to(assets_root) or not evaluation.is_relative_to(assets_root):
            raise RuntimeError(f"Campaign output escaped asset root: {run_id}")
        audited = audit_replay(output, evaluation, run["metrics"]["COMBINED_SEQ"])
        declared_manifest = run.get("prediction_manifest_sha256")
        if declared_manifest and declared_manifest != audited["manifest_sha256"]:
            raise RuntimeError(f"Declared manifest hash mismatch: {run_id}")
        audited["run_id"] = run_id
        audited["representation"] = run.get("job", {}).get("identity_representation")
        results.append(audited)
    return len(results), results


def audit_modality_campaigns(assets_root: Path, measured: dict[str, Any]) -> dict[str, Any]:
    campaign_files = {
        "ir": [assets_root / "phase3" / f"plim_modality_ir_seed{seed}_campaign_summary.json" for seed in range(3)],
        "rgb_ir": [assets_root / "phase3" / f"plim_modality_rgb_ir_seed{seed}_campaign_summary.json" for seed in range(3)],
        "two_rgb": [assets_root / "phase3" / f"plim_modality_two_rgb_seed{seed}_campaign_summary.json" for seed in range(3)],
    }
    selected: dict[str, Any] = {}
    all_runs: list[dict[str, Any]] = []
    campaign_audit: dict[str, Any] = {}
    for representation, paths in campaign_files.items():
        campaign_audit[representation] = []
        for path in paths:
            count, runs = audit_campaign_summary(path, assets_root)
            if count != 6:
                raise RuntimeError(f"Expected six runs in {path}, found {count}")
            campaign_audit[representation].append({"summary": str(path), "sha256": sha256(path), "runs": runs})
            all_runs.extend(runs)
        source_row = measured["rows"][representation]
        # The selected rows are fixed at weight .3 and are recomputed from the
        # three registered seed summaries, rather than copied from the report.
        prefix = {
            "ir": "plim_mod_dev_ir_gru_predictive_anchor",
            "rgb_ir": "plim_mod_dev_rgb_ir_gru_predictive_anchor",
            "two_rgb": "plim_mod_dev_two_rgb_gru_predictive_anchor",
        }[representation]
        chosen = [r for r in all_runs if r["run_id"].startswith(prefix + "_seed") and "_w0.3" in r["run_id"]]
        if len(chosen) != 3:
            raise RuntimeError(f"Expected three selected runs for {representation}, found {len(chosen)}")
        row_key = next(key for key in source_row if key.startswith(prefix))
        declared = source_row[row_key]
        recomputed_mean: dict[str, float] = {}
        recomputed_std: dict[str, float] = {}
        for name in METRICS:
            values = [float(run["metrics"][name]) for run in chosen]
            recomputed_mean[name] = statistics.mean(values)
            recomputed_std[name] = statistics.stdev(values)
            if not close(recomputed_mean[name], declared["mean"][name]):
                raise RuntimeError(f"Mean mismatch for {representation} {name}")
            # The source omits standard deviation for DetA/MOTA; compare all
            # declared fields and keep the recomputed values for the report.
            if name in declared.get("std", {}) and not close(recomputed_std[name], declared["std"][name]):
                raise RuntimeError(f"Sample std mismatch for {representation} {name}")
        selected[representation] = {
            "run_ids": [run["run_id"] for run in chosen],
            "mean": recomputed_mean,
            "std_sample": recomputed_std,
        }
    return {"campaigns": campaign_audit, "selected": selected, "run_count": len(all_runs)}


def audit_cmue(assets_root: Path) -> dict[str, Any]:
    measured_rounds = read_json(ROOT / "docs/research/phase2_measured_rounds_20261012.json")
    third_measured = read_json(ROOT / "docs/research/phase2_third_measured_results_20261012.json")
    rounds: dict[str, Any] = {}
    for round_name, round_info in measured_rounds["cmue_rounds"].items():
        summary_path = Path(round_info["summary_path"])
        if not summary_path.is_absolute():
            summary_path = ROOT / summary_path
        if sha256(summary_path) != round_info["summary_sha256"]:
            raise RuntimeError(f"CMUE source summary hash mismatch: {summary_path}")
        count, runs = audit_campaign_summary(summary_path, assets_root)
        if count != 9:
            raise RuntimeError(f"Expected nine CMUE runs in {summary_path}, found {count}")
        selected = [run for run in runs if "_w0.3" in run["run_id"]]
        rounds[round_name] = {
            "summary": str(summary_path),
            "summary_sha256": sha256(summary_path),
            "run_count": count,
            "selected_w0.3": {
                run["run_id"]: run["metrics"] for run in selected
            },
        }
    third_summary = assets_root / "phase2" / "cmue_third_seed0_campaign_summary.json"
    if sha256(third_summary) != third_measured["source_summary_sha256"]:
        raise RuntimeError("CMUE third-round source summary hash mismatch")
    third_count, third_runs = audit_campaign_summary(third_summary, assets_root)
    if third_count != 9:
        raise RuntimeError(f"Expected nine third-round CMUE runs, found {third_count}")
    rounds["ranking_sign_v3"] = {
        "summary": str(third_summary),
        "summary_sha256": sha256(third_summary),
        "run_count": third_count,
        "selected_w0.3": {run["run_id"]: run["metrics"] for run in third_runs if "_w0.3" in run["run_id"]},
    }
    return rounds


def audit_joint(assets_root: Path) -> dict[str, Any]:
    source = read_json(ROOT / "docs/research/phase4_joint_measured_seed0_20261012.json")
    rows = source["rows"]
    audited: dict[str, Any] = {}
    for run_id, expected in rows.items():
        output = assets_root / "phase3" / run_id
        evaluation = assets_root / "phase3" / f"{run_id}_evaluation"
        audited[run_id] = audit_replay(output, evaluation, expected)
    only = audited["joint_v1_dev_plim_only_seed0"]
    cmue = audited["joint_v1_dev_plim_cmue_seed0"]
    for sequence in only["prediction_sha256"]:
        left = Path(only["output"]) / f"{sequence}.txt"
        right = Path(cmue["output"]) / f"{sequence}.txt"
        if left.read_bytes() != right.read_bytes():
            raise RuntimeError(f"PLIM-only and CMUE joint outputs differ: {sequence}")
    cmue_asset = read_json(assets_root / "phase3" / "joint_v1_dev_plim_cmue_seed0" / "manifest.json")
    for decision in cmue_asset["videos"].values():
        utility = decision.get("utility_counts", {})
        if utility.get("accepted_rows") != 0 or utility.get("weight_sum") != 0.0:
            raise RuntimeError("CMUE joint audit did not reject all IR rows")
    return {
        "run_count": len(audited),
        "runs": audited,
        "cmue_rejected_all_ir": True,
        "plim_only_cmue_byte_equal": True,
    }


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assets_root = args.assets_root.resolve()
    free_gib = shutil.disk_usage(assets_root).free / 2**30
    if free_gib < 30.0:
        raise RuntimeError(f"Storage reserve below 30 GiB: {free_gib:.2f}")

    modality_source = read_json(ROOT / "docs/research/phase3_modality_prediction_measured_20261012.json")
    modality = audit_modality_campaigns(assets_root, modality_source)
    cmue = audit_cmue(assets_root)
    joint = audit_joint(assets_root)

    source_files = [
        ROOT / "cfp/utility_association.py",
        ROOT / "cfp/joint_association.py",
        ROOT / "cfp/representations.py",
        ROOT / "cfp/candidate_graph.py",
        ROOT / "tools/cfp_replay_utility.py",
        ROOT / "tools/cfp_replay_predictor.py",
        ROOT / "docs/research/CMUE_EXPERIMENTS.md",
        ROOT / "docs/research/PLIM_EXPERIMENTS.md",
        ROOT / "docs/research/FINAL_RESEARCH_REPORT.md",
    ]
    output = {
        "status": "COMPLETE_READ_ONLY_MULTIMODAL_AUDIT",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "protocol": {
            "partition": "dev",
            "sequences": ["BT-005", "SN-013"],
            "frames": 4834,
            "same_v3_detection_cache": True,
            "no_new_asset_generation": True,
            "free_gib_after_audit": round(shutil.disk_usage(assets_root).free / 2**30, 3),
        },
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in source_files},
        "modality_prediction": modality,
        "cmue": cmue,
        "joint_composition": joint,
        "deltas": {
            "rgb_ir_predictive_minus_two_rgb_predictive": {
                name: modality["selected"]["rgb_ir"]["mean"][name] - modality["selected"]["two_rgb"]["mean"][name]
                for name in METRICS
            },
            "rgb_ir_predictive_minus_ir_predictive": {
                name: modality["selected"]["rgb_ir"]["mean"][name] - modality["selected"]["ir"]["mean"][name]
                for name in METRICS
            },
            "cmue_round1_minus_constant_ir_control": {
                "HOTA": 50.97685469415867 - 50.9528,
                "AssA": 36.32727681677821 - 36.2939,
                "IDF1": 61.21307387934305 - 61.2089,
                "IDSW": 86 - 85,
                "control_source": "docs/research/CMUE_EXPERIMENTS.md",
            },
        },
        "classification": {
            "validated_multimodal_directions": [
                "CMUE conditional synthetic-IR utility selection",
                "PLIM paired RGB+IR predictive latent identity association",
            ],
            "multimodal_control": "two-RGB capacity control",
            "unvalidated_original_direction": "local interaction-aware RGB-IR association (H3)",
            "not_multimodal": "Phase VI A/B/C recoverability, risk/memory protection, and candidate-residual modules read RGB-only current-frame coordinates",
            "modality_boundary": "IR is causally constructed from RGB; results support conditional representation/optimization claims, not independent thermal-sensor information.",
        },
        "conclusion": {
            "stable_independent_multimodal_gain": False,
            "reason": "RGB+IR predictive is only +0.2123 HOTA over two-RGB predictive, essentially matches the lower-capacity ordinary single-RGB GRU, while CMUE is near a ranking-free constant-IR control and rejects all IR rows in the final joint audit.",
            "recommended_claim": "The implementation demonstrates a testable conditional RGB/synthetic-IR fusion mechanism and recoverable negative evidence; it does not yet validate an IR-specific improvement over strong RGB/capacity controls.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError(f"Output is immutable and already exists: {args.output}")
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": output["status"], "output": str(args.output), "modality_runs": modality["run_count"], "cmue_rounds": len(cmue), "joint_runs": joint["run_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
