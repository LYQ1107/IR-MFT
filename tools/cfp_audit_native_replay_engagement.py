#!/usr/bin/env python3
"""Audit that the all-fold native FIFO replay actually engages identity edges.

The accepted all-fold replay manifests predate a metadata fix: they expose the
``AppearanceTracker.decision_counts`` field, while the active
``PredictiveAssociationTracker`` records native FIFO edge use in
``prediction_counts``.  This audit runs the exact replay source in a temporary
one-line instrumented copy, preserves the immutable official outputs, and
requires byte-identical predictions plus nonzero native edge engagement.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfp.provenance import sha256, snapshot_sources, write_json


VIDEOS = ("BT-001", "BT-003", "MSK-002", "PF-001", "SN-001", "SN-015")
MODES = ("masked_visible_tokens", "visible_target")
SEEDS = (0, 1, 2)


def instrument_replay(repo, path):
    source_path = repo / "tools/cfp_replay_native_current_identity.py"
    source = source_path.read_text()
    marker = "identity_counts=tracker.decision_counts,model_counts=getattr(tracker,'local_counts',None),"
    replacement = "identity_counts=tracker.decision_counts, prediction_counts=tracker.prediction_counts, model_counts=getattr(tracker,'local_counts',None),"
    if source.count(marker) != 1:
        raise ValueError("Native replay metadata marker changed; audit must stop")
    path.write_text(source.replace(marker, replacement))
    return source_path, sha256(source_path), sha256(path)


def run_one(task, replay, repo, env):
    output = Path(task["audit_output"])
    log = Path(task["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(replay),
               "--config", task["config"], "--features", task["feature"],
               "--detections", task["detections"], "--output", str(output),
               "--video", task["video"], "--memory", "fifo"]
    with log.open("w") as stream:
        stream.write("$ " + " ".join(map(str, command)) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=str(repo), env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"instrumented replay failed: {task['video']} {task['mode']} s{task['seed']}")
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"missing instrumented manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETE_NATIVE_CURRENT_IDENTITY_REPLAY":
        raise RuntimeError(f"incomplete instrumented replay: {manifest_path}")
    official = json.loads((Path(task["official_output"]) / "manifest.json").read_text())
    if manifest.get("prediction_sha256") != official.get("prediction_sha256"):
        raise ValueError(f"Prediction bytes changed in engagement audit: {task['video']} {task['mode']} s{task['seed']}")
    counts = manifest.get("prediction_counts")
    if not isinstance(counts, dict) or counts.get("predicted_rows", 0) <= 0 or counts.get("modified_edges", 0) <= 0:
        raise ValueError(f"Native identity path did not engage: {task['video']} {task['mode']} s{task['seed']}: {counts}")
    return dict(video=task["video"], mode=task["mode"], seed=task["seed"],
                official_manifest=task["official_output"] + "/manifest.json",
                audit_manifest=str(manifest_path),
                official_prediction_sha256=official.get("prediction_sha256"),
                audit_prediction_sha256=manifest.get("prediction_sha256"),
                official_identity_counts=official.get("identity_counts"),
                prediction_counts=counts,
                frames=manifest.get("frames"), output_rows=manifest.get("output_rows"),
                online_GT_teacher_future_used=manifest.get("online_GT_teacher_future_used"),
                actual_native_history_source=manifest.get("actual_native_history_source"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--feature-root", type=Path, required=True)
    ap.add_argument("--official-replay-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.workers < 1 or args.workers > 12:
        raise ValueError("workers must be in [1,12]")
    if args.output_root.exists():
        raise ValueError("Audit output must be a new immutable directory")
    config = args.repo / "configs/cfp/phase_vii_native_current_identity_replay_v1.json"
    detections = Path("/data3/liuyeqiang/IR-MFT_assets/cfp/phase0/causal_v3_train_detections")
    args.output_root.mkdir(parents=True)
    args.logs.mkdir(parents=True, exist_ok=True)
    replay = args.output_root / "instrumented_cfp_replay_native_current_identity.py"
    canonical, canonical_sha, instrumented_sha = instrument_replay(args.repo, replay)
    tasks = []
    for video in VIDEOS:
        for mode in MODES:
            for seed in SEEDS:
                official = args.official_replay_root / f"native_visible_token_{mode}_{video}_s{seed}_v1"
                feature = args.feature_root / "features" / video / f"{mode}_s{seed}"
                official_manifest = official / "manifest.json"
                feature_manifest = feature / "manifest.json"
                if not official_manifest.exists() or not feature_manifest.exists():
                    raise FileNotFoundError(f"Missing accepted all-fold artifact for {video} {mode} s{seed}")
                if json.loads(feature_manifest.read_text()).get("status") != "COMPLETE_NATIVE_CURRENT_VISIBLE_TOKEN_FEATURES":
                    raise ValueError(f"Feature bank is not complete: {feature_manifest}")
                if json.loads(official_manifest.read_text()).get("status") != "COMPLETE_NATIVE_CURRENT_IDENTITY_REPLAY":
                    raise ValueError(f"Official replay is not complete: {official_manifest}")
                output = args.output_root / f"native_visible_token_{mode}_{video}_s{seed}_v1"
                tasks.append(dict(video=video, mode=mode, seed=seed, config=str(config),
                                  feature=str(feature), detections=str(detections),
                                  official_output=str(official), audit_output=str(output),
                                  log=str(args.logs / f"{video}_{mode}_s{seed}.log")))
    env = dict(os.environ, PYTHONPATH=str(args.repo), CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    started = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, task, replay, args.repo, env) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
            print(json.dumps(rows[-1], sort_keys=True), flush=True)
    rows.sort(key=lambda row: (row["video"], row["mode"], row["seed"]))
    all_sha_equal = all(row["official_prediction_sha256"] == row["audit_prediction_sha256"] for row in rows)
    positive = all(row["prediction_counts"]["predicted_rows"] > 0 and row["prediction_counts"]["modified_edges"] > 0 for row in rows)
    report = dict(
        status="COMPLETE_ALLFOLD_NATIVE_REPLAY_ENGAGEMENT_AUDIT_36",
        created_unix=started,
        finished_unix=time.time(),
        canonical_replay_source=str(canonical),
        canonical_replay_sha256=canonical_sha,
        instrumented_replay_sha256=instrumented_sha,
        config=str(config),
        feature_root=str(args.feature_root),
        official_replay_root=str(args.official_replay_root),
        output_root=str(args.output_root),
        rows=rows,
        task_count=len(rows),
        all_prediction_hashes_equal=all_sha_equal,
        all_native_prediction_paths_engaged=positive,
        interpretation=("The accepted replay manifests' identity_counts field is an inherited "
                        "AppearanceTracker counter and remains zero for the PredictiveAssociationTracker. "
                        "The active native FIFO association path is recorded by prediction_counts; all 36 "
                        "instrumented replays must have positive predicted_rows and modified_edges while "
                        "remaining byte-identical to the accepted predictions."),
        scope=("Read-only metadata audit over six videos x two visible-token objectives x three seeds. "
               "No official output is overwritten; no GT, physical ID, teacher, future frame, or model is used. "
               "The replay timing remains association-only and is not an end-to-end FPS measurement."),
        sources=snapshot_sources(args.report.with_suffix(".sources"), [
            __file__, canonical, config, args.repo / "cfp/native_local_identity.py",
            args.repo / "cfp/predictive_association.py", args.repo / "cfp/predictive.py",
            args.repo / "cfp/association.py", args.repo / "cfp/provenance.py"]),
    )
    write_json(args.report, report)
    if len(rows) != 36 or not all_sha_equal or not positive:
        raise RuntimeError("Native engagement audit did not satisfy all 36 guards")
    print(json.dumps(dict(status=report["status"], tasks=len(rows), output_hashes_equal=all_sha_equal,
                          native_paths_engaged=positive)), flush=True)


if __name__ == "__main__":
    main()
