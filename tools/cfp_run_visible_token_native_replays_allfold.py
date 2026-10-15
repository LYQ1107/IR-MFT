#!/usr/bin/env python3
"""Replay and score completed visible-token native feature banks on all folds."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfp.provenance import sha256, snapshot_sources, write_json


VIDEOS = ("BT-001", "BT-003", "MSK-002", "PF-001", "SN-001", "SN-015")
MODES = ("masked_visible_tokens", "visible_target")
SEEDS = (0, 1, 2)


def run_checked(command, cwd, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as stream:
        stream.write("$ " + " ".join(map(str, command)) + "\n")
        stream.flush()
        return subprocess.run(command, cwd=str(cwd), env=env,
                              stdout=stream, stderr=subprocess.STDOUT).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--feature-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--logs", type=Path, required=True)
    args = ap.parse_args()
    config = args.repo / "configs/cfp/phase_vii_native_current_identity_replay_v1.json"
    replay = args.repo / "tools/cfp_replay_native_current_identity.py"
    evaluate = args.repo / "tools/cfp_evaluate.py"
    detections = Path("/data3/liuyeqiang/IR-MFT_assets/cfp/phase0/causal_v3_train_detections")
    data_root = Path("/data3/liuyeqiang/datasets/mft26-ir-causal-v1")
    derived_gt = Path("/data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt")
    trackeval = Path("/data3/liuyeqiang/CFP-FishMOT_references/TrackEval")
    protocol_template = args.repo / "configs/cfp/phase_vii_joint_evaluation_subset_v1.json"
    source_paths = [Path(__file__), config, replay, evaluate, protocol_template,
                    args.repo / "cfp/native_local_identity.py",
                    args.repo / "cfp/predictive_association.py",
                    args.repo / "cfp/predictive.py", args.repo / "cfp/association.py",
                    args.repo / "cfp/provenance.py"]
    guards = {str(p): sha256(p) for p in source_paths}
    tasks = []
    for video in VIDEOS:
        for mode in MODES:
            for seed in SEEDS:
                feature = args.feature_root / "features" / video / f"{mode}_s{seed}"
                output = args.output_root / f"native_visible_token_{mode}_{video}_s{seed}_v1"
                tasks.append(dict(video=video, mode=mode, seed=seed,
                                  feature=str(feature), output=str(output),
                                  evaluation=str(output) + ".trackeval",
                                  log=str(args.logs / f"{video}_{mode}_s{seed}.log"),
                                  status="PENDING", error=None))
    declared = [{k: t[k] for k in ("video", "mode", "seed", "feature", "output", "evaluation", "log")} for t in tasks]
    if args.state.exists():
        state = json.loads(args.state.read_text())
        if state.get("source_guards") != guards or state.get("tasks_declared") != declared:
            raise ValueError("Existing replay state has different source guards or task declaration")
        old = {(t["video"], t["mode"], t["seed"]): t for t in state.get("tasks", [])}
        for task in tasks:
            previous = old.get((task["video"], task["mode"], task["seed"]))
            if previous:
                for key in ("status", "error", "replay_manifest_sha256", "metrics_sha256", "metrics"):
                    if key in previous:
                        task[key] = previous[key]
    else:
        state = dict(status="PENDING", created_unix=time.time(), source_guards=guards,
                     tasks_declared=declared, tasks=[], scope=(
                         "All six outer videos x two fit-only visible-token objectives x three seeds. "
                         "Each completed current-only native feature bank is replayed through the unchanged "
                         "FIFO lifecycle and official TrackEval; no GT enters the tracker."))
        state["sources"] = snapshot_sources(args.state.with_suffix(".sources"), source_paths)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.logs.mkdir(parents=True, exist_ok=True)
    state["tasks"] = tasks
    write_json(args.state, state)
    env = dict(os.environ, PYTHONPATH=str(args.repo), CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    while True:
        waiting = 0
        for task in tasks:
            if task["status"] == "COMPLETE":
                continue
            feature = Path(task["feature"])
            output = Path(task["output"])
            evaluation = Path(task["evaluation"])
            feature_manifest = feature / "manifest.json"
            if not feature_manifest.exists():
                task["status"] = "WAITING_FEATURES"
                waiting += 1
                continue
            fm = json.loads(feature_manifest.read_text())
            if fm.get("status") != "COMPLETE_NATIVE_CURRENT_VISIBLE_TOKEN_FEATURES":
                task["status"] = "WAITING_FEATURES"
                waiting += 1
                continue
            task["status"] = "RUNNING_REPLAY"
            state["status"] = "RUNNING"
            state["active"] = task
            write_json(args.state, state)
            if not output.exists():
                command = [sys.executable, str(replay), "--config", str(config),
                           "--features", str(feature), "--detections", str(detections),
                           "--output", str(output), "--video", task["video"], "--memory", "fifo"]
                if run_checked(command, args.repo, Path(task["log"]), env):
                    task["status"] = "FAILED"
                    task["error"] = "native replay returned nonzero"
                    write_json(args.state, state)
                    raise RuntimeError(task["error"])
            replay_manifest = output / "manifest.json"
            rm = json.loads(replay_manifest.read_text())
            if rm.get("status") != "COMPLETE_NATIVE_CURRENT_IDENTITY_REPLAY":
                raise RuntimeError(f"Replay did not complete: {replay_manifest}")
            if not evaluation.exists():
                evaluation.mkdir(parents=True)
                protocol = json.loads(protocol_template.read_text())
                protocol.update(train=[task["video"]], dev=[], historical_test=[],
                                role="Offline official scoring of all-fold native visible-token replay; no metric selection")
                write_json(evaluation / "protocol.json", protocol)
                command = [sys.executable, str(evaluate), "--data-root", str(data_root),
                           "--derived-gt-root", str(derived_gt), "--protocol", str(evaluation / "protocol.json"),
                           "--partition", "train", "--predictions", str(output),
                           "--output", str(evaluation), "--trackeval-root", str(trackeval)]
                if run_checked(command, args.repo, evaluation / "evaluation.log", env):
                    raise RuntimeError(f"official scoring failed: {task['video']} {task['mode']} s{task['seed']}")
            metrics_path = evaluation / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            video_metrics = metrics["metrics"][task["video"]]
            artifact = metrics["artifacts"][task["video"]]
            if artifact["prediction_sha256"] != rm["prediction_sha256"]:
                raise ValueError("TrackEval prediction does not match native replay")
            task.update(status="COMPLETE", replay_manifest_sha256=sha256(replay_manifest),
                        metrics_sha256=sha256(metrics_path), metrics=video_metrics)
            state["tasks"] = tasks
            state["active"] = None
            write_json(args.state, state)
            print(json.dumps(dict(video=task["video"], mode=task["mode"], seed=task["seed"],
                                  HOTA=video_metrics["HOTA"], IDF1=video_metrics["IDF1"],
                                  IDSW=video_metrics["IDSW"])), flush=True)
        if all(task["status"] == "COMPLETE" for task in tasks):
            break
        state["status"] = "WAITING_FEATURES" if waiting else "RUNNING"
        state["active"] = None
        state["tasks"] = tasks
        write_json(args.state, state)
        time.sleep(30)
    state.update(status="COMPLETE_ALLFOLD_NATIVE_REPLAYS_36", finished_unix=time.time(), active=None)
    write_json(args.state, state)
    report = dict(status="COMPLETE_ALLFOLD_NATIVE_REPLAYS_36", config=str(config),
                  config_sha256=sha256(config), feature_root=str(args.feature_root),
                  tasks=tasks, source_guards=guards, state_sha256=sha256(args.state),
                  sources=snapshot_sources(args.report.with_suffix(".sources"), source_paths),
                  scope=state["scope"],
                  limitation="Feature extraction and replay are native current-frame evidence; raw decode/IR/V3/encoder end-to-end timing remains in the separate strict-raw campaign.")
    write_json(args.report, report)


if __name__ == "__main__":
    main()
