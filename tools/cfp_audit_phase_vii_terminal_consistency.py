#!/usr/bin/env python3
"""Cross-check the terminal Phase VII evidence/document boundary.

This audit is deliberately bookkeeping-only.  It verifies hashes, coverage,
test-scope metadata and delivery pointers; it does not infer a positive
scientific result from any measured report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(rel: str) -> tuple[Path, Any]:
    path = ROOT / rel
    return path, json.loads(path.read_text())


def nested_status(rows: list[dict[str, Any]], row_id: str) -> str | None:
    for row in rows:
        if row.get("id") == row_id:
            return row.get("status")
    return None


def remote_head(branch: str) -> tuple[str | None, str | None]:
    try:
        proc = subprocess.run(
            ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    line = proc.stdout.strip().splitlines()
    if not line:
        return None, "git ls-remote returned no matching ref"
    return line[0].split()[0], None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="docs/research/PHASE_VII_TERMINAL_CONSISTENCY_AUDIT_MEASURED.json",
    )
    args = parser.parse_args()

    state_path, state = load("docs/research/PHASE_VII_STATE.json")
    registry_path, registry = load("docs/research/PHASE_VII_EXPERIMENT_REGISTRY.json")
    task_path, task = load("docs/research/PHASE_VII_TASK_BOOK_REQUIREMENT_AUDIT_MEASURED.json")
    _, dinov2 = load("docs/research/PHASE_VII_DINOV2_ALLFOLD_RETRIEVAL_AUDIT_MEASURED.json")
    _, native = load("docs/research/PHASE_VII_VISIBLE_TOKEN_ALLFOLD_NATIVE_REPLAYS_MEASURED.json")
    _, strict_raw = load("docs/research/PHASE_VII_VISIBLE_TOKEN_ALLFOLD_STRICT_RAW_MEASURED.json")
    _, local4 = load("docs/research/PHASE_VII_NATIVE_LOCAL4_COMPLEMENTARITY_MEASURED.json")
    goal_path = ROOT / "docs/research/PHASE_VII_FINAL_GOAL.md"
    ledger_path = ROOT / "docs/research/PHASE_VII_HYPOTHESIS_LEDGER.md"
    report_path = ROOT / "docs/research/PHASE_VII_FINAL_RESEARCH_REPORT.md"

    reqs = task["requirements"]
    req_status = {row["id"]: row["status"] for row in reqs}
    required_documents = task["required_documents"]
    required_document_checks = []
    for entry in required_documents:
        path = ROOT / "docs/research" / entry["path"]
        actual = sha256(path) if path.exists() else None
        required_document_checks.append(
            {
                "path": entry["path"],
                "exists": path.exists(),
                "expected_sha256": entry.get("sha256"),
                "actual_sha256": actual,
                "hash_matches": actual == entry.get("sha256"),
            }
        )

    current_validation_state = state["validation_current_authoritative"]
    current_validation_registry = registry["validation_current_authoritative"]
    branch = state.get("branch", "research/evidence-fishmot-v1")
    remote, remote_error = remote_head(branch)
    local_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()

    checks = {
        "task_book_requirement_rows_17": len(reqs) == 17,
        "task_book_summary_15_scoped_2_open": task["summary"] == {
            "complete_or_scoped": 15,
            "not_justified_or_unestablished": 2,
            "total": 17,
        },
        "required_documents_13_present_and_hashed": len(required_documents) == 13
        and all(row["exists"] and row["hash_matches"] for row in required_document_checks),
        "goal_and_ledger_have_authoritative_309_test_text": "309" in goal_path.read_text()
        and "309" in ledger_path.read_text()
        and not re.search(r"\b(?:293|301|305)\s+(?:Phase|passed)\b", goal_path.read_text())
        and not re.search(r"\b(?:293|301|305)\s+(?:Phase|passed)\b", ledger_path.read_text()),
        "current_validation_state_309_pass_0_fail_24_warnings": (
            current_validation_state["phase_vii_passed"] == 309
            and current_validation_state["phase_vii_failed"] == 0
            and current_validation_state["warnings"] == 24
        ),
        "current_validation_registry_matches_state": current_validation_registry[
            "phase_vii_passed"
        ]
        == current_validation_state["phase_vii_passed"]
        and current_validation_registry["phase_vii_failed"]
        == current_validation_state["phase_vii_failed"]
        and current_validation_registry["warnings"] == current_validation_state["warnings"],
        "native_allfold_36_features_and_36_replays": (
            native["coverage"]["feature_tasks"] == 36
            and native["coverage"]["replay_tasks"] == 36
            and native["coverage"]["all_feature_guards_pass"]
            and native["coverage"]["all_replay_guards_pass"]
        ),
        "strict_raw_allfold_36_runs": strict_raw["coverage"]["actual_runs"] == 36,
        "dinov2_allfold_144_crop_controls": (
            dinov2["status"] == "COMPLETE_DINOV2_ALLFOLD_K4_K8_THREE_SEED_CROP_CONTROLS"
            and dinov2["coverage"]["runs_total"] == 144
        ),
        "local4_complementarity_gate_failed_without_replay": (
            local4["status"] == "COMPLETE_LOCAL4_COMPLEMENTARITY_AUDIT"
            and local4["decision"] == "DO_NOT_RUN_LOCAL4_COMBINATION_REPLAY"
            and local4["aggregate_counts"]["rows"] == 428220
            and local4["aggregate_counts"]["local4_correction"] == 0
            and local4["aggregate_counts"]["local4_harm"] == 0
            and local4["aggregate_counts"]["margin_correction"] == 0
            and local4["aggregate_counts"]["margin_harm"] == 0
            and local4["promotion_gate"]["pass_"] is False
        ),
        "p11_open_without_current_component_justification": req_status.get(
            "P11_selective_combinations"
        )
        == "NOT_JUSTIFIED_BY_CURRENT_COMPONENT_EVIDENCE",
        "p17_open_without_stable_independent_contribution": req_status.get(
            "P17_final_scientific_contribution"
        )
        == "NOT_ESTABLISHED",
        "goal_status_active_in_all_authoritative_records": (
            task["goal_status"] == "IN_PROGRESS_UNACHIEVED"
            and state["goal_status"] == "IN_PROGRESS_UNACHIEVED"
            and registry["goal_status"] == "IN_PROGRESS_UNACHIEVED"
        ),
        "no_active_phase_vii_processes_recorded": state.get("active_jobs") == [],
        "remote_branch_resolves": remote is not None,
        "remote_content_pointer_matches_task_audit": remote is not None
        and remote != task["remote_branch_last_verified"]
        and len(task["remote_branch_last_verified"]) == 40,
    }

    payload = {
        "status": "COMPLETE_TERMINAL_CONSISTENCY_CROSSCHECK_GOAL_STILL_ACTIVE"
        if all(checks.values())
        else "INCOMPLETE_TERMINAL_CONSISTENCY_CROSSCHECK",
        "as_of": date.today().isoformat(),
        "scope": "Read-only document/hash/coverage/delivery cross-check. It does not establish a positive method, P11 combination value, P17 contribution, external generalization or project completion.",
        "checks": checks,
        "required_document_checks": required_document_checks,
        "requirement_statuses": req_status,
        "authoritative_hashes": {
            "state": sha256(state_path),
            "registry": sha256(registry_path),
            "task_book_audit": sha256(task_path),
            "goal": sha256(goal_path),
            "hypothesis_ledger": sha256(ledger_path),
            "final_research_report": sha256(report_path),
        },
        "delivery": {
            "branch": branch,
            "local_head_at_run": local_head,
            "remote_branch_head": remote,
            "remote_error": remote_error,
            "task_audit_content_pointer": task["remote_branch_last_verified"],
            "remote_branch_pointer_is_metadata_after_content": remote
            != task["remote_branch_last_verified"],
        },
        "coverage": {
            "requirements_total": len(reqs),
            "required_documents_total": len(required_documents),
            "native_feature_tasks": native["coverage"]["feature_tasks"],
            "native_replay_tasks": native["coverage"]["replay_tasks"],
            "strict_raw_runs": strict_raw["coverage"]["actual_runs"],
            "dinov2_crop_runs": dinov2["coverage"]["runs_total"],
            "local4_clean_history_rows": local4["aggregate_counts"]["rows"],
            "local4_prediction_replays": local4["coverage"]["expected_replays"],
        },
        "open_boundaries": task["open_boundaries"],
    }

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"path": str(output), "status": payload["status"], "checks": checks}, indent=2))
    return 0 if payload["status"].startswith("COMPLETE_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
