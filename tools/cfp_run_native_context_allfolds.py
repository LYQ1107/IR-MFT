#!/usr/bin/env python3
"""Replay outer-fold context-identity models on complete native videos."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from cfp.provenance import sha256, snapshot_sources, write_json


VIDEOS = ("BT-001", "BT-003", "MSK-002", "PF-001", "SN-001", "SN-015")
MODES = ("global", "local_temporal", "full")
SEEDS = (0, 1, 2)


def run_checked(command, *, cwd, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as stream:
        stream.write("$ " + " ".join(map(str, command)) + "\n")
        stream.flush()
        return subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args()
    config_path = args.repo / "configs/cfp/phase_vii_native_current_identity_replay_v1.json"
    replay_path = args.repo / "tools/cfp_replay_native_current_identity.py"
    evaluate_path = args.repo / "tools/cfp_evaluate.py"
    source_paths = [
        Path(__file__), replay_path, evaluate_path,
        args.repo / "cfp/native_local_identity.py",
        args.repo / "cfp/local_dense_jepa.py",
        args.repo / "cfp/local_dense_context_identity.py",
        args.repo / "cfp/predictive_association.py",
        args.repo / "cfp/association.py",
        args.repo / "cfp/predictive.py",
        args.repo / "cfp/provenance.py",
        config_path,
    ]
    guards = {str(path): sha256(path) for path in source_paths}
    tasks = [(video, mode, seed) for video in VIDEOS for mode in args.modes for seed in SEEDS]
    if args.state.exists():
        state = json.loads(args.state.read_text())
        if state.get("status") == "COMPLETE_CONTEXT_IDENTITY_NATIVE_ALLFOLDS":
            print(json.dumps({"status": state["status"], "completed": len(state["completed"])}))
            return
        if state.get("source_guards") != guards or state.get("tasks") != [list(task) for task in tasks]:
            raise ValueError("Existing state has different source guards or task order")
    else:
        state = dict(
            status="RUNNING",
            pid=os.getpid(),
            tasks=[list(task) for task in tasks],
            completed=[],
            active=None,
            source_guards=guards,
            sources=snapshot_sources(args.state.with_suffix(".sources"), source_paths),
            scope=(
                "Outer-fold K4 context-identity models on all six complete native videos, "
                "three seeds and the frozen-DINO current proposal bank. Modes are capacity "
                "global, ordinary local-temporal and context-V2 full. Each model was fit with "
                "the evaluated outer video excluded; no GT, teacher or future input is used online."
            ),
        )
        write_json(args.state, state)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.repo)
    env["CUDA_VISIBLE_DEVICES"] = ""
    config = json.loads(config_path.read_text())
    detections = Path("/data3/liuyeqiang/IR-MFT_assets/cfp/phase0/causal_v3_train_detections")
    trackeval_root = Path("/data3/liuyeqiang/CFP-FishMOT_references/TrackEval")
    data_root = Path("/data3/liuyeqiang/datasets/mft26-ir-causal-v1")
    derived_gt_root = Path("/data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt")
    feature_root_base = args.root
    completed_keys = {(x["video"], x["mode"], x["seed"]) for x in state.get("completed", [])}
    try:
        for video, mode, seed in tasks:
            if (video, mode, seed) in completed_keys:
                continue
            if any(sha256(path) != digest for path, digest in guards.items()):
                raise ValueError("Guarded source drift")
            model = args.root / f"local_dense_context_{mode}_{video}_k4_s{seed}_v2"
            features = feature_root_base / f"native_dino_frozen_{video}_k4_v1"
            output = args.root / f"native_context_allfold_{mode}_{video}_s{seed}_v1"
            evaluation = Path(str(output) + ".trackeval")
            replay_log = Path(str(output) + ".log")
            state.update(status="RUNNING", active=dict(video=video, mode=mode, seed=seed, model=str(model), output=str(output)))
            write_json(args.state, state)
            model_manifest = json.loads((model / "manifest.json").read_text())
            if model_manifest.get("status") != "COMPLETE" or model_manifest["guard"]["outer"] != video:
                raise ValueError(f"Invalid outer-fold model: {model}")
            if not output.exists():
                command = [
                    sys.executable, str(replay_path), "--config", str(config_path),
                    "--features", str(features), "--detections", str(detections),
                    "--output", str(output), "--video", video, "--model", str(model),
                    "--context-identity",
                ]
                if run_checked(command, cwd=args.repo, log_path=replay_log, env=env):
                    raise RuntimeError(f"native replay failed: {video} {mode} seed{seed}")
            manifest_path = output / "manifest.json"
            replay_manifest = json.loads(manifest_path.read_text())
            if replay_manifest.get("status") != "COMPLETE_NATIVE_CURRENT_IDENTITY_REPLAY":
                raise ValueError(f"Replay is not complete: {manifest_path}")
            if not evaluation.exists():
                evaluation.mkdir()
                protocol = json.loads((args.repo / "configs/cfp/phase_vii_joint_evaluation_subset_v1.json").read_text())
                protocol.update(train=[video], dev=[], historical_test=[], role="Offline complete native outer-fold scoring; no metric-based selection")
                write_json(evaluation / "protocol.json", protocol)
                command = [
                    sys.executable, str(evaluate_path), "--data-root", str(data_root),
                    "--derived-gt-root", str(derived_gt_root), "--protocol", str(evaluation / "protocol.json"),
                    "--partition", "train", "--predictions", str(output), "--output", str(evaluation),
                    "--trackeval-root", str(trackeval_root),
                ]
                if run_checked(command, cwd=args.repo, log_path=evaluation / "evaluation.log", env=env):
                    raise RuntimeError(f"official scoring failed: {video} {mode} seed{seed}")
            metrics_path = evaluation / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            video_metrics = metrics["metrics"][video]
            artifact = metrics["artifacts"][video]
            if artifact["prediction_sha256"] != replay_manifest["prediction_sha256"]:
                raise ValueError("Official metrics do not reference replay prediction")
            row = dict(
                video=video,
                mode=mode,
                seed=seed,
                model=str(model),
                model_manifest_sha256=sha256(model / "manifest.json"),
                model_checkpoint_sha256=model_manifest["checkpoint_sha256"],
                feature_manifest_sha256=replay_manifest["feature_manifest_sha256"],
                replay_manifest_sha256=sha256(manifest_path),
                prediction_sha256=replay_manifest["prediction_sha256"],
                official_metrics_sha256=sha256(metrics_path),
                metrics=video_metrics,
                association_seconds=replay_manifest["association_seconds"],
                output_rows=replay_manifest["output_rows"],
            )
            state["completed"].append(row)
            state["active"] = None
            write_json(args.state, state)
            completed_keys.add((video, mode, seed))
            print(json.dumps(dict(video=video, mode=mode, seed=seed, HOTA=video_metrics["HOTA"], completed=len(state["completed"]))), flush=True)
        report = dict(
            status="COMPLETE_CONTEXT_IDENTITY_NATIVE_ALLFOLDS",
            config=config,
            modes=list(args.modes),
            seeds=list(SEEDS),
            videos=list(VIDEOS),
            runs=state["completed"],
            state_sha256=sha256(args.state),
            sources=snapshot_sources(args.report.with_suffix(".sources"), source_paths),
            scope=state["scope"],
            limitation="Replay excludes raw detector/IR/encoder startup and is not a whole-raw FPS claim; outer models are independently fit with the evaluated video excluded.",
        )
        write_json(args.report, report)
        state.update(status="COMPLETE_CONTEXT_IDENTITY_NATIVE_ALLFOLDS", report=str(args.report), active=None)
        write_json(args.state, state)
    except Exception as error:
        state.update(status="HALTED_ACTUAL_FAILURE", error=str(error), active=state.get("active"))
        write_json(args.state, state)
        raise


if __name__ == "__main__":
    main()
