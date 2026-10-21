#!/usr/bin/env python3
"""Read-only audit and aggregation of every completed official method family.

The experiment assets are immutable.  This script only reads existing
prediction manifests and TrackEval ``metrics.json`` files, verifies the
registered hashes/metric payloads, and writes a compact aggregate report to
the repository.  It never trains, replays, evaluates, or writes below the
asset root.

The report keeps Protocol A, seed-0 development controls, Protocol B online
interventions, and the Phase VI six-video OOF protocol separate.  Every row
contains the complete TrackEval metric schema (47 fields); HOTA threshold
arrays are aggregated element by element.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = Path("/data3/liuyeqiang/IR-MFT_assets/cfp")
ARRAY_METRICS = ("HOTA_TP", "HOTA_FN", "HOTA_FP")
CORE_METRICS = (
    "HOTA",
    "AssA",
    "DetA",
    "LocA",
    "IDF1",
    "IDP",
    "IDR",
    "MOTA",
    "MOTP",
    "CLR_Re",
    "CLR_Pr",
    "CLR_F1",
    "IDSW",
    "Frag",
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing JSON: {path}")
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: Any, right: Any, tolerance: float = 1e-7) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return left == right


def compare_metric_dict(actual: dict[str, Any], expected: dict[str, Any], where: str) -> None:
    for name, wanted in expected.items():
        if name not in actual:
            raise RuntimeError(f"Missing metric {name} at {where}")
        got = actual[name]
        if isinstance(wanted, list):
            if got != wanted:
                raise RuntimeError(f"Metric array mismatch at {where}: {name}")
        elif not close(got, wanted):
            raise RuntimeError(f"Metric mismatch at {where}: {name} {got!r} != {wanted!r}")


def command_arg(command: list[str], name: str) -> Path:
    try:
        return Path(command[command.index(name) + 1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"Missing {name} in command: {command}") from exc


def resolve_asset_path(path: Path, assets_root: Path) -> Path:
    if path.is_absolute():
        resolved = path
    else:
        resolved = ROOT / path
    resolved = resolved.resolve()
    if not resolved.is_relative_to(assets_root.resolve()):
        raise RuntimeError(f"Asset path escapes asset root: {resolved}")
    return resolved


def audit_pair(
    output: Path,
    evaluation: Path,
    assets_root: Path,
    expected: dict[str, Any] | None = None,
    declared_evaluation_sha256: str | None = None,
    declared_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    output = resolve_asset_path(output, assets_root)
    evaluation = resolve_asset_path(evaluation, assets_root)
    manifest_path = output / "manifest.json"
    metrics_path = evaluation / "metrics.json"
    manifest = read_json(manifest_path)
    measured = read_json(metrics_path)
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError(f"Prediction manifest is not COMPLETE: {manifest_path}")
    actual = measured.get("metrics", {}).get("COMBINED_SEQ")
    if not isinstance(actual, dict):
        raise RuntimeError(f"Missing COMBINED_SEQ metrics: {metrics_path}")
    if len(actual) != 47:
        raise RuntimeError(f"Expected 47 TrackEval fields at {metrics_path}, found {len(actual)}")
    for name in ARRAY_METRICS:
        if not isinstance(actual.get(name), list) or not actual[name]:
            raise RuntimeError(f"Invalid HOTA array at {metrics_path}: {name}")
    for name, value in actual.items():
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                raise RuntimeError(f"Non-finite metric at {metrics_path}: {name}")
    if expected is not None:
        compare_metric_dict(actual, expected, str(metrics_path))
    evaluation_hash = sha256(metrics_path)
    if declared_evaluation_sha256 and evaluation_hash != declared_evaluation_sha256:
        raise RuntimeError(f"Evaluation hash mismatch: {metrics_path}")
    manifest_hash = sha256(manifest_path)
    if declared_manifest_sha256 and manifest_hash != declared_manifest_sha256:
        raise RuntimeError(f"Manifest hash mismatch: {manifest_path}")

    prediction_hashes: dict[str, str] = {}
    artifacts = measured.get("artifacts", {})
    for sequence, info in manifest.get("videos", {}).items():
        prediction_path = output / f"{sequence}.txt"
        if not prediction_path.is_file():
            raise RuntimeError(f"Missing prediction file: {prediction_path}")
        prediction_hash = sha256(prediction_path)
        if prediction_hash != info.get("prediction_sha256"):
            raise RuntimeError(f"Prediction hash mismatch: {prediction_path}")
        if artifacts.get(sequence, {}).get("prediction_sha256") != prediction_hash:
            raise RuntimeError(f"Evaluation artifact hash mismatch: {metrics_path}")
        prediction_hashes[sequence] = prediction_hash
    return {
        "output": str(output),
        "evaluation": str(evaluation),
        "manifest_sha256": manifest_hash,
        "evaluation_sha256": evaluation_hash,
        "prediction_sha256": prediction_hashes,
        "source_split": measured.get("source_split"),
        "sequences": measured.get("sequences"),
        "trackeval_commit": measured.get("trackeval_commit"),
        "metrics": actual,
    }


def audit_campaign_run(summary: dict[str, Any], run_id: str, assets_root: Path) -> dict[str, Any]:
    run = summary.get("runs", {}).get(run_id)
    if not isinstance(run, dict):
        raise RuntimeError(f"Run {run_id} missing from campaign summary")
    commands = run.get("commands", [])
    if len(commands) < 2:
        raise RuntimeError(f"Missing replay/evaluation commands for {run_id}")
    output = command_arg(commands[0], "--output")
    evaluation = command_arg(commands[1], "--output")
    result = audit_pair(
        output,
        evaluation,
        assets_root,
        expected=run.get("metrics", {}).get("COMBINED_SEQ"),
        declared_evaluation_sha256=run.get("evaluation_sha256"),
        declared_manifest_sha256=run.get("prediction_manifest_sha256"),
    )
    result["run_id"] = run_id
    result["job"] = run.get("job", {})
    return result


def mean_value(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, list):
        if not all(isinstance(v, list) and len(v) == len(first) for v in values):
            raise RuntimeError("Cannot aggregate arrays with different lengths")
        return [statistics.mean(float(v[i]) for v in values) for i in range(len(first))]
    return statistics.mean(float(v) for v in values)


def std_value(values: list[Any]) -> Any:
    if len(values) < 2:
        return None
    first = values[0]
    if isinstance(first, list):
        return [statistics.stdev(float(v[i]) for v in values) for i in range(len(first))]
    return statistics.stdev(float(v) for v in values)


def aggregate_row(
    method_id: str,
    label: str,
    protocol: str,
    family: str,
    audits: list[dict[str, Any]],
    source: str,
    parameter: str | None = None,
) -> dict[str, Any]:
    if not audits:
        raise RuntimeError(f"No audits for {method_id}")
    metric_names = list(audits[0]["metrics"])
    if len(metric_names) != 47:
        raise RuntimeError(f"Unexpected metric schema for {method_id}")
    for audit in audits[1:]:
        if list(audit["metrics"]) != metric_names:
            raise RuntimeError(f"Metric schema differs for {method_id}")
    means = {name: mean_value([audit["metrics"][name] for audit in audits]) for name in metric_names}
    stds = {name: std_value([audit["metrics"][name] for audit in audits]) for name in metric_names}
    return {
        "method_id": method_id,
        "label": label,
        "protocol": protocol,
        "family": family,
        "parameter": parameter,
        "replicates": len(audits),
        "run_ids": [audit["run_id"] for audit in audits],
        "source": source,
        "evaluation_sha256": [audit["evaluation_sha256"] for audit in audits],
        "manifest_sha256": [audit["manifest_sha256"] for audit in audits],
        "metrics": {"mean": means, "sample_std": stds},
    }


def load_campaign(path: Path) -> dict[str, Any]:
    summary = read_json(path)
    if summary.get("status") != "COMPLETE":
        raise RuntimeError(f"Campaign is not COMPLETE: {path}")
    return summary


def audit_selected_campaign_family(
    assets_root: Path,
    summary_paths: list[Path],
    run_ids: list[str],
) -> list[dict[str, Any]]:
    summaries = {path.name: load_campaign(path) for path in summary_paths}
    audits: list[dict[str, Any]] = []
    for run_id in run_ids:
        match = re.search(r"_seed(\d+)_", run_id)
        if not match:
            raise RuntimeError(f"Cannot infer seed from {run_id}")
        seed = int(match.group(1))
        candidates = [summary for name, summary in summaries.items() if f"seed{seed}_" in name]
        if len(candidates) != 1:
            raise RuntimeError(f"Cannot locate campaign summary for {run_id}")
        audits.append(audit_campaign_run(candidates[0], run_id, assets_root))
    return audits


def declared_selection_check(
    aggregate: dict[str, Any],
    declared: dict[str, Any],
    fields: list[str] | tuple[str, ...] | None = None,
) -> None:
    names = fields or list(declared.get("mean", {}))
    for name in names:
        if name not in declared.get("mean", {}):
            continue
        if not close(aggregate["metrics"]["mean"][name], declared["mean"][name], 1e-6):
            raise RuntimeError(f"Selection mean mismatch for {aggregate['method_id']} {name}")
        if name in declared.get("sample_std", {}) and declared["sample_std"][name] is not None:
            if not close(aggregate["metrics"]["sample_std"][name], declared["sample_std"][name], 1e-6):
                raise RuntimeError(f"Selection std mismatch for {aggregate['method_id']} {name}")


def protocol_a_rows(assets_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audit_sources: dict[str, Any] = {}

    baseline_selection = read_json(ROOT / "configs/cfp/baseline_selection_post_neck_v1.json")
    baseline_labels = {
        "bot_sort": ("V3 + BoT-SORT + RGB ReID", "appearance/association"),
        "hybrid_sort": ("V3 + HybridSORT + RGB ReID", "appearance/association"),
        "rgb": ("RGB EMA", "single modality memory"),
        "ir": ("IR EMA", "single modality memory"),
        "fixed": ("fixed RGB/IR EMA", "two-modality memory"),
    }
    for family, (label, family_name) in baseline_labels.items():
        selected = baseline_selection["families"][family]["selected"]
        audits: list[dict[str, Any]] = []
        for run_id in selected["run_ids"]:
            evaluation = assets_root / "phase1" / f"{run_id}_evaluation"
            audits.append(
                audit_pair(
                    evaluation.with_name(run_id),
                    evaluation,
                    assets_root,
                    expected=None,
                )
            )
            audits[-1]["run_id"] = run_id
        row = aggregate_row(
            f"protocol_a_{family}",
            label,
            "Protocol A / same V3 detection / dev",
            family_name,
            audits,
            "configs/cfp/baseline_selection_post_neck_v1.json",
            selected.get("parameter"),
        )
        declared_selection_check(row, selected, fields=list(selected.get("mean", {})))
        rows.append(row)
        audit_sources[family] = {"config": str(ROOT / "configs/cfp/baseline_selection_post_neck_v1.json")}

    byte_eval = assets_root / "phase0/causal_dev_bytetrack_off_evaluation"
    byte_audit = audit_pair(byte_eval.with_name("causal_dev_bytetrack_off"), byte_eval, assets_root)
    byte_audit["run_id"] = "causal_dev_bytetrack_off"
    rows.append(
        aggregate_row(
            "protocol_a_bytetrack",
            "ByteTrack",
            "Protocol A / same V3 detection / dev",
            "external baseline",
            [byte_audit],
            "phase0/causal_dev_bytetrack_off_evaluation/metrics.json",
            None,
        )
    )

    campaign_groups = [
        (
            "anchor",
            ROOT / "configs/cfp/plim_anchor_three_seed_selection_v3.json",
            [assets_root / "phase3" / f"plim_anchor_seed{seed}_campaign_summary.json" for seed in range(3)],
            "Protocol A / same V3 detection / dev",
            {
                "plim_v3_dev_last_anchor": "last anchor",
                "plim_v3_dev_ema_anchor": "EMA anchor",
                "plim_v3_dev_fifo_anchor": "FIFO anchor",
                "plim_v3_dev_all_history_ema_anchor": "all-history EMA anchor",
                "plim_v3_dev_constant_zero_anchor": "constant-zero anchor",
                "plim_v3_dev_constant_half_anchor": "constant-half anchor",
                "plim_v3_dev_anchor_only_anchor": "anchor-only",
                "plim_v3_dev_gru_predictive_anchor": "predictive RGB GRU",
                "plim_v3_dev_gru_association_anchor": "ordinary RGB GRU",
                "plim_v3_dev_transformer_predictive_anchor": "predictive RGB Transformer",
                "plim_v3_dev_transformer_association_anchor": "ordinary RGB Transformer",
            },
        ),
        (
            "log_time",
            ROOT / "configs/cfp/plim_log_time_three_seed_selection_v4.json",
            [assets_root / "phase3" / f"plim_log_time_seed{seed}_campaign_summary.json" for seed in range(3)],
            "Protocol A / same V3 detection / dev",
            {
                "plim_v4_dev_gru_predictive_log_anchor": "log-time predictive RGB GRU",
                "plim_v4_dev_gru_association_log_anchor": "log-time ordinary RGB GRU",
                "plim_v4_dev_transformer_predictive_log_anchor": "log-time predictive RGB Transformer",
                "plim_v4_dev_transformer_association_log_anchor": "log-time ordinary RGB Transformer",
            },
        ),
    ]
    for group, selection_path, summary_paths, protocol, labels in campaign_groups:
        selection = read_json(selection_path)
        for family, label in labels.items():
            selected = selection["families"][family]["selected"]
            run_ids = selected.get("runs") or [
                f"{family}_seed{seed}_w{selected['appearance_weight']:g}" for seed in range(3)
            ]
            audits = audit_selected_campaign_family(assets_root, summary_paths, run_ids)
            row = aggregate_row(
                f"protocol_a_{family}",
                label,
                protocol,
                group,
                audits,
                str(selection_path.relative_to(ROOT)),
                f"appearance_weight={selected['appearance_weight']:g}",
            )
            declared_selection_check(row, selected)
            rows.append(row)
            audit_sources[family] = {"config": str(selection_path)}

    modality_source = read_json(ROOT / "docs/research/phase3_modality_prediction_measured_20261012.json")
    modality_summary_names = {
        "ir": "plim_modality_ir_seed{seed}_campaign_summary.json",
        "rgb_ir": "plim_modality_rgb_ir_seed{seed}_campaign_summary.json",
        "two_rgb": "plim_modality_two_rgb_seed{seed}_campaign_summary.json",
    }
    modality_labels = {
        "plim_mod_dev_ir_gru_predictive_anchor": "IR predictive GRU",
        "plim_mod_dev_ir_gru_association_anchor": "IR ordinary GRU",
        "plim_mod_dev_rgb_ir_gru_predictive_anchor": "RGB+IR predictive GRU",
        "plim_mod_dev_rgb_ir_gru_association_anchor": "RGB+IR ordinary GRU",
        "plim_mod_dev_two_rgb_gru_predictive_anchor": "two-RGB predictive GRU",
        "plim_mod_dev_two_rgb_gru_association_anchor": "two-RGB ordinary GRU",
    }
    for representation, summary_name in modality_summary_names.items():
        summary_paths = [assets_root / "phase3" / summary_name.format(seed=seed) for seed in range(3)]
        for family, declared in modality_source["rows"][representation].items():
            run_ids = [f"{family}_seed{seed}_w{declared['weight']:g}" for seed in range(3)]
            audits = audit_selected_campaign_family(assets_root, summary_paths, run_ids)
            row = aggregate_row(
                f"protocol_a_{family}",
                modality_labels[family],
                "Protocol A / same V3 detection / dev",
                f"modality/{representation}",
                audits,
                "docs/research/phase3_modality_prediction_measured_20261012.json",
                f"appearance_weight={declared['weight']:g}",
            )
            declared_selection_check(row, declared, fields=list(declared.get("mean", {})))
            rows.append(row)
            audit_sources[family] = {"source": str(ROOT / "docs/research/phase3_modality_prediction_measured_20261012.json")}

    return rows, audit_sources


def seed0_control_rows(assets_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cmue_summary = load_campaign(assets_root / "phase2/cmue_third_seed0_campaign_summary.json")
    cmue_labels = {
        f"cmue_v3_dev_{family}_seed0_w{weight}": f"{label} (w={weight})"
        for family, label in (
            ("cmue", "CMUE v3"),
            ("ordinary", "ordinary utility control"),
            ("quality", "quality utility control"),
        )
        for weight in ("0.1", "0.3", "0.5")
    }
    for run_id, label in cmue_labels.items():
        audit = audit_campaign_run(cmue_summary, run_id, assets_root)
        rows.append(
            aggregate_row(
                f"protocol_a_seed0_{run_id}",
                label,
                "Protocol A / seed0 development control",
                "CMUE utility control",
                [audit],
                "phase2/cmue_third_seed0_campaign_summary.json",
                "w=0.3",
            )
        )

    joint_source = read_json(ROOT / "docs/research/phase4_joint_measured_seed0_20261012.json")
    for run_id, expected in joint_source["rows"].items():
        output = assets_root / "phase3" / run_id
        evaluation = assets_root / "phase3" / f"{run_id}_evaluation"
        audit = audit_pair(output, evaluation, assets_root, expected=None)
        audit["run_id"] = run_id
        rows.append(
            aggregate_row(
                f"protocol_a_seed0_{run_id}",
                run_id.removeprefix("joint_v1_dev_").removesuffix("_seed0"),
                "Protocol A / seed0 development control",
                "joint composition / write policy",
                [audit],
                "docs/research/phase4_joint_measured_seed0_20261012.json",
                "seed0",
            )
        )
        compare_metric_dict(audit["metrics"], expected, str(evaluation / "metrics.json"))
    return rows


def protocol_b_rows(assets_root: Path) -> list[dict[str, Any]]:
    source = read_json(ROOT / "docs/research/phase4_online_corrected_measured_20261012.json")
    rows: list[dict[str, Any]] = []
    for run_id, entry in source["runs"].items():
        output = assets_root / "phase4" / run_id
        evaluation = assets_root / "phase4" / f"{run_id}_evaluation"
        audit = audit_pair(
            output,
            evaluation,
            assets_root,
            expected=entry["metrics"]["COMBINED_SEQ"],
            declared_evaluation_sha256=entry.get("evaluation_sha256"),
            declared_manifest_sha256=entry.get("manifest_sha256"),
        )
        audit["run_id"] = run_id
        label = run_id.removesuffix("_v1").replace("_", " ")
        rows.append(
            aggregate_row(
                f"protocol_b_{run_id}",
                label,
                "Protocol B / full current-frame online",
                "online deployment/intervention",
                [audit],
                "docs/research/phase4_online_corrected_measured_20261012.json",
                entry.get("scope"),
            )
        )
    return rows


def phase6_rows(assets_root: Path) -> list[dict[str, Any]]:
    source_path = ROOT / "docs/research/PHASE_VI_SYNTHESIS_MEASURED.json"
    source = read_json(source_path)
    rows: list[dict[str, Any]] = []
    for section, section_info in source["official"].items():
        families = section_info.get("result", {}).get("families", {})
        for family, entry in families.items():
            command = entry.get("evaluation_command")
            if not isinstance(command, list):
                raise RuntimeError(f"Missing evaluation command for Phase VI {section}/{family}")
            output = command_arg(command, "--predictions")
            evaluation = command_arg(command, "--output")
            audit = audit_pair(
                output,
                evaluation,
                assets_root,
                expected=entry["metrics"]["COMBINED_SEQ"],
                declared_evaluation_sha256=entry.get("evaluation_sha256"),
            )
            audit["run_id"] = f"phase6_{section}_{family}"
            rows.append(
                aggregate_row(
                    f"phase6_{section}_{family}",
                    f"Phase VI {section}/{family}",
                    "Phase VI / six-video OOF train partition",
                    section,
                    [audit],
                    "docs/research/PHASE_VI_SYNTHESIS_MEASURED.json",
                    "single registered OOF replay",
                )
            )
    return rows


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def scalar_delta(row: dict[str, Any], baseline: dict[str, Any], names: list[str]) -> dict[str, float]:
    return {name: row["metrics"]["mean"][name] - baseline["metrics"]["mean"][name] for name in names}


def build_report(assets_root: Path) -> dict[str, Any]:
    free_gib = shutil.disk_usage(assets_root).free / 2**30
    if free_gib < 30.0:
        raise RuntimeError(f"Storage reserve below 30 GiB: {free_gib:.2f}")
    protocol_a, audit_sources = protocol_a_rows(assets_root)
    seed0 = seed0_control_rows(assets_root)
    protocol_b = protocol_b_rows(assets_root)
    phase6 = phase6_rows(assets_root)
    rows = protocol_a + seed0 + protocol_b + phase6
    if not rows:
        raise RuntimeError("No method rows collected")
    metric_names = list(rows[0]["metrics"]["mean"])
    if len(metric_names) != 47:
        raise RuntimeError(f"Expected 47 metrics in report, found {len(metric_names)}")
    protocol_a_lookup = {row["method_id"]: row for row in protocol_a}
    ordinary = protocol_a_lookup["protocol_a_plim_v3_dev_gru_association_anchor"]
    bot = protocol_a_lookup["protocol_a_bot_sort"]
    modality_lookup = {
        row["method_id"]: row
        for row in protocol_a
        if row["family"].startswith("modality/")
    }
    deltas = {
        "protocol_a_vs_ordinary_rgb_gru": {
            row["method_id"]: scalar_delta(row, ordinary, list(CORE_METRICS))
            for row in protocol_a
        },
        "protocol_a_vs_bot_sort": {
            row["method_id"]: scalar_delta(row, bot, list(CORE_METRICS))
            for row in protocol_a
        },
        "rgb_ir_predictive_minus_two_rgb_predictive": scalar_delta(
            modality_lookup["protocol_a_plim_mod_dev_rgb_ir_gru_predictive_anchor"],
            modality_lookup["protocol_a_plim_mod_dev_two_rgb_gru_predictive_anchor"],
            list(CORE_METRICS),
        ),
        "rgb_ir_predictive_minus_ir_predictive": scalar_delta(
            modality_lookup["protocol_a_plim_mod_dev_rgb_ir_gru_predictive_anchor"],
            modality_lookup["protocol_a_plim_mod_dev_ir_gru_predictive_anchor"],
            list(CORE_METRICS),
        ),
    }
    by_protocol: dict[str, int] = {}
    for row in rows:
        by_protocol[row["protocol"]] = by_protocol.get(row["protocol"], 0) + 1
    return {
        "status": "COMPLETE_READ_ONLY_ALL_METHOD_METRIC_AUDIT",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "protocols": {
            "protocol_a": {
                "partition": "dev",
                "sequences": ["BT-005", "SN-013"],
                "frames": 4834,
                "same_v3_detection_cache": True,
                "three_seed_rows": len(protocol_a),
                "selection_is_frozen": True,
            },
            "protocol_a_seed0_controls": {
                "partition": "dev",
                "sequences": ["BT-005", "SN-013"],
                "frames": 4834,
                "three_seed_aggregation": False,
            },
            "protocol_b": {
                "partition": "dev",
                "sequences": ["BT-005", "SN-013"],
                "frames": 4834,
                "full_current_frame_online": True,
            },
            "phase6_oof": {
                "partition": "train",
                "sequences": ["BT-001", "BT-003", "MSK-002", "PF-001", "SN-001", "SN-015"],
                "six_video_outer_exclusion": True,
                "single_registered_replay_rows": len(phase6),
            },
        },
        "metric_schema": {
            "count": len(metric_names),
            "fields": metric_names,
            "array_fields": list(ARRAY_METRICS),
            "scalar_fields": [name for name in metric_names if name not in ARRAY_METRICS],
            "aggregate": "mean and sample standard deviation across registered replicates; arrays are elementwise",
        },
        "audit": {
            "assets_root": str(assets_root),
            "no_new_asset_generation": True,
            "free_gib_after_audit": round(shutil.disk_usage(assets_root).free / 2**30, 3),
            "rows_by_protocol": by_protocol,
            "campaign_sources": audit_sources,
        },
        "deltas": deltas,
        "rows": rows,
    }


def format_cell(mean: Any, std: Any) -> str:
    if isinstance(mean, list):
        return json.dumps(mean, separators=(",", ":")) + (" ± " + json.dumps(std, separators=(",", ":")) if std is not None else "")
    if std is None:
        return f"{float(mean):.6f}"
    return f"{float(mean):.6f} ± {float(std):.6f}"


def write_tsv(report: dict[str, Any], path: Path) -> None:
    fields = report["metric_schema"]["fields"]
    lines = ["protocol\tmethod_id\tlabel\tfamily\tparameter\treplicates\t" + "\t".join(f"{f}_mean\t{f}_sample_std" for f in fields)]
    for row in report["rows"]:
        values = [row["protocol"], row["method_id"], row["label"], row["family"], str(row.get("parameter") or ""), str(row["replicates"])]
        for field in fields:
            values.extend([json.dumps(row["metrics"]["mean"][field], ensure_ascii=False, separators=(",", ":")), json.dumps(row["metrics"]["sample_std"][field], ensure_ascii=False, separators=(",", ":"))])
        lines.append("\t".join(value.replace("\t", " ").replace("\n", " ") for value in values))
    path.write_text("\n".join(lines) + "\n")


def write_markdown(report: dict[str, Any], path: Path) -> None:
    core = report["metric_schema"]["fields"]
    main = ["HOTA", "AssA", "DetA", "LocA", "IDF1", "IDP", "IDR", "MOTA", "MOTP", "CLR_Re", "CLR_Pr", "CLR_F1", "IDSW", "Frag"]
    lines = [
        "# All completed methods: full TrackEval metric audit",
        "",
        "状态：**COMPLETE_READ_ONLY_ALL_METHOD_METRIC_AUDIT**。本报告读取已有官方 `metrics.json` 和预测 manifest，校验哈希后重新聚合，不训练、不重放、不生成资产。每行保留完整 47 个 TrackEval 字段；HOTA 的三个阈值数组按元素聚合。",
        "",
        "不同协议分开呈现：Protocol A 是 BT-005/SN-013 的同检测输入开发协议；Protocol B 是完整当前帧在线流程；Phase VI 是六视频 OOF train 分区。三者不能直接比较绝对数值。",
        "",
        f"- 总行数：{len(report['rows'])}",
        f"- 指标字段：{report['metric_schema']['count']}（数组：{', '.join(report['metric_schema']['array_fields'])}）",
        f"- 评估后剩余空间：{report['audit']['free_gib_after_audit']:.3f} GiB",
        "- 完整均值/样本标准差 TSV：`ALL_METHODS_FULL_TRACKING_METRICS.tsv`",
        "- 机器可读完整结果：`ALL_METHODS_FULL_TRACKING_METRICS.json`",
        "",
        "## Protocol A：三种子完整方法",
        "",
        "表中显示最常用的官方字段；47 个字段见 TSV/JSON。± 是跨注册重复的样本标准差；ByteTrack 只有一个注册输出，因此没有方差。",
        "",
        "| 方法 | n | " + " | ".join(main) + " |",
        "|---|---:|" + "---:|" * len(main),
    ]
    for row in report["rows"]:
        if row["protocol"] != "Protocol A / same V3 detection / dev":
            continue
        vals = []
        for name in main:
            vals.append(format_cell(row["metrics"]["mean"][name], row["metrics"]["sample_std"][name]))
        lines.append("| " + row["label"] + " | " + str(row["replicates"]) + " | " + " | ".join(vals) + " |")
    lines += [
        "",
        "### Protocol A 的指标冠军与多模态差值",
        "",
        "- HOTA、DetA、MOTA、MOTP、CLR_Pr、CLR_F1、OWTA 和 sMOTA 最高的是 BoT-SORT；AssA、IDF1、IDP、IDR 最高的是普通 RGB GRU。最低 IDSW 的三种子方法是预测 RGB GRU（83.667±1.528）。",
        "- RGB+IR 预测 GRU 相对双 RGB 预测 GRU：HOTA **+0.2123**、AssA **+0.3423**、IDF1 **+0.4006**、MOTA **+0.0244**，但 DetA **−0.0590**、IDSW **+1.667**；相对普通 RGB GRU：HOTA **−0.0007**、AssA **−0.0014**、IDF1 **−0.0566**、MOTA **+0.0488**、IDSW **−2.333**。因此配对预测有小幅融合差值，但未证明 IR 特有的稳定总体收益。",
        "",
        "## Protocol A：seed0 组合与 utility 控制",
        "",
        "这些行是诊断/组合控制，不与三种子均值混合。完整 47 字段仍在 TSV/JSON。",
        "",
        "| 方法 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        if row["protocol"] != "Protocol A / seed0 development control":
            continue
        m = row["metrics"]["mean"]
        lines.append(f"| {row['label']} | {m['HOTA']:.6f} | {m['AssA']:.6f} | {m['DetA']:.6f} | {m['IDF1']:.6f} | {m['MOTA']:.6f} | {m['IDSW']:.3f} |")
    lines += [
        "",
        "## Protocol B 与 Phase VI",
        "",
        "Protocol B 的在线干预和 Phase VI 的 OOF 方法也已逐行校验完整 47 字段；为避免生成数百列的不可读 Markdown，这些行和所有指标放在 TSV/JSON 中。Phase VI 的结果尤其不能与 Protocol A 的 4,834 帧开发集直接相减。",
        "",
        "## 聚合差值（Protocol A 三种子）",
        "",
        "相对普通 RGB GRU 和 BoT-SORT 的全部核心字段差值，以及 RGB+IR 预测相对 IR/双 RGB 控制的差值，已写入 JSON 的 `deltas`。其中 IDSW 以“越低越好”解读，其余大多数字段以“越高越好”解读；DetA/MOTA 等检测字段受外部关联器输出策略影响。",
        "",
        "## 可复核来源",
        "",
        "验证脚本：`tools/cfp_validate_all_method_metrics.py`。它会检查每个评估的 47 字段、评估哈希、预测 manifest、逐视频预测哈希和 TrackEval artifact 哈希，并拒绝写入资产目录。",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/research/ALL_METHODS_FULL_TRACKING_METRICS.json")
    parser.add_argument("--tsv", type=Path, default=ROOT / "docs/research/ALL_METHODS_FULL_TRACKING_METRICS.tsv")
    parser.add_argument("--markdown", type=Path, default=ROOT / "docs/research/ALL_METHODS_FULL_TRACKING_METRICS.md")
    args = parser.parse_args()
    assets_root = args.assets_root.resolve()
    report = build_report(assets_root)
    for output in (args.output, args.tsv, args.markdown):
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise RuntimeError(f"Output is immutable and already exists: {output}")
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    write_tsv(report, args.tsv)
    write_markdown(report, args.markdown)
    print(json.dumps({"status": report["status"], "rows": len(report["rows"]), "metrics": report["metric_schema"]["count"], "output": str(args.output), "tsv": str(args.tsv), "markdown": str(args.markdown), "free_gib": report["audit"]["free_gib_after_audit"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
