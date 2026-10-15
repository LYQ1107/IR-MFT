#!/usr/bin/env python3
"""Audit the completed DINOv2 all-fold crop-retrieval controls.

The K4 and K8 reports already contain the measured runs.  This read-only
auditor checks their declared fold/mode/seed coverage, matching guards and
the compact aggregate values before the reports are marked terminal.  It
does not load images, checkpoints or labels and it cannot create MOT output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


VIDEOS = ("BT-001", "BT-003", "MSK-002", "PF-001", "SN-001", "SN-015")
MODES = ("global", "simple", "global_capacity", "patch_attention")
SEEDS = (0, 1, 2)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_report(path: Path, expected_k: int) -> dict:
    data = json.loads(path.read_text())
    assert data["completed_stage"] == f"COMPLETE_ALL_SIX_FOLDS_THREE_SEED_DINOV2_K{expected_k}"
    assert data["local_count"] == expected_k
    assert data["completed_fits"] == 72
    assert data["same_actual_sampling_coverage_and_matched_capacity"] is True
    assert set(data["summary"]) == set(VIDEOS)
    assert len(data["runs"]) == 72

    seen = set()
    for name, run in data["runs"].items():
        guard = run["guard"]
        assert guard["outer"] in VIDEOS
        assert guard["mode"] in MODES
        assert guard["seed"] in SEEDS
        assert guard["local_count"] == expected_k
        assert set(guard["fit"]) == set(VIDEOS) - {guard["outer"], guard["inner"]}
        assert guard["outer"] != guard["inner"]
        assert run["frozen_prefix_unchanged"] is True
        assert run["checkpoint_sha256"]
        assert run["diagnostic_sha256"]
        expected_name = f"dinov2_fish_{guard['mode']}_{guard['outer']}_k{expected_k}_s{guard['seed']}_v2"
        assert name == expected_name
        key = (guard["outer"], guard["mode"], guard["seed"])
        assert key not in seen
        seen.add(key)
    assert len(seen) == 72

    return {
        "report": str(path),
        "report_sha256": digest(path),
        "k": expected_k,
        "status_before_audit": data["status"],
        "completed_fits": data["completed_fits"],
        "outer_videos": list(VIDEOS),
        "modes": list(MODES),
        "seeds": list(SEEDS),
        "runs": len(data["runs"]),
        "macro_actual_detector_gap25": data["macro_actual_detector_gap25"],
        "macro_frozen_cls_actual_detector_gap25": data["macro_frozen_cls_actual_detector_gap25"],
        "source": data["source"],
        "scope": data["scope"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k4", type=Path, required=True)
    parser.add_argument("--k8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    k4 = check_report(args.k4, 4)
    k8 = check_report(args.k8, 8)
    assert k4["source"] == k8["source"]
    report = {
        "status": "COMPLETE_DINOV2_ALLFOLD_K4_K8_THREE_SEED_CROP_CONTROLS",
        "reports": {"k4": k4, "k8": k8},
        "coverage": {
            "outer_videos": list(VIDEOS),
            "modes": list(MODES),
            "seeds": list(SEEDS),
            "runs_per_k": 72,
            "runs_total": 144,
            "all_fold_mode_seed_coverage": True,
            "same_input_and_matched_capacity_guard": True,
            "fit_excludes_outer_and_inner": True,
        },
        "decision": "CLOSE_PROTOCOL_A_DINOV2_ALLFOLD_CROP_DIAGNOSTIC",
        "scope": (
            "Read-only audit of the existing six leave-two-video-out DINOv2-S crop "
            "retrieval reports. It validates 144 fixed final-epoch runs across K4/K8, "
            "four matched modes and three seeds. It supplies Protocol A identity/crop "
            "evidence only; it does not claim native MOT, raw online FPS, external "
            "generalization or a final novelty contribution."
        ),
        "interpretation": (
            "The all-fold K4/K8 crop controls are complete and reproducible at the "
            "declared report level. Global/simple/capacity/patch modes remain matched "
            "diagnostics; their aggregate retrieval differences do not authorize a "
            "new native replay or mechanical final stacking."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "runs_total": 144}, sort_keys=True))


if __name__ == "__main__":
    main()
