#!/usr/bin/env python3
"""Run and score the pre-registered all-fold learned local-attention native replay."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from cfp.provenance import sha256, snapshot_sources, write_json


VIDEOS = ("BT-001", "BT-003", "MSK-002", "PF-001", "SN-001", "SN-015")
SEEDS = (0, 1, 2)


def run_checked(command, *, cwd, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as stream:
        stream.write("$ " + " ".join(map(str, command)) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                stderr=subprocess.STDOUT)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="External phase7 asset root containing models/features")
    parser.add_argument("--output-root", type=Path,
                        help="Separate output filesystem with the 40GiB reserve")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    output_root = args.output_root or args.root
    output_root.mkdir(parents=True, exist_ok=True)

    config_path = args.repo / "configs/cfp/phase_vii_native_local_attention_allfold_v1.json"
    replay_config = args.repo / "configs/cfp/phase_vii_native_current_identity_replay_v1.json"
    evaluation_config = args.repo / "configs/cfp/phase_vii_joint_evaluation_subset_v1.json"
    replay_path = args.repo / "tools/cfp_replay_native_current_identity.py"
    evaluate_path = args.repo / "tools/cfp_evaluate.py"
    source_paths = [Path(__file__), config_path, replay_config, evaluation_config, replay_path,
                    evaluate_path, args.repo / "cfp/native_local_identity.py",
                    args.repo / "cfp/local_dense_jepa.py",
                    args.repo / "cfp/local_dense_context_identity.py",
                    args.repo / "cfp/predictive_association.py",
                    args.repo / "cfp/association.py", args.repo / "cfp/predictive.py",
                    args.repo / "cfp/provenance.py"]
    guards = {str(path): sha256(path) for path in source_paths}
    tasks = [(video, seed) for video in VIDEOS for seed in SEEDS]
    if args.state.exists():
        state = json.loads(args.state.read_text())
        if state.get("status") == "COMPLETE_NATIVE_LOCAL_ATTENTION_ALLFOLD":
            print(json.dumps({"status": state["status"], "completed": len(state["completed"])}))
            return
        if state.get("source_guards") != guards or state.get("tasks") != [list(x) for x in tasks]:
            raise ValueError("Existing state has different source guards or task order")
    else:
        state = dict(
            status="RUNNING", pid=os.getpid(), tasks=[list(x) for x in tasks],
            completed=[], active=None, source_guards=guards,
            sources=snapshot_sources(args.state.with_suffix(".sources"), source_paths),
            config_sha256=sha256(config_path), replay_config_sha256=sha256(replay_config),
            scope=json.loads(config_path.read_text())["scope"],
        )
        write_json(args.state, state)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.repo)
    env["CUDA_VISIBLE_DEVICES"] = ""
    data_root = Path("/data3/liuyeqiang/datasets/mft26-ir-causal-v1")
    derived_gt_root = Path("/data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt")
    detections = Path("/data3/liuyeqiang/IR-MFT_assets/cfp/phase0/causal_v3_train_detections")
    trackeval_root = Path("/data3/liuyeqiang/CFP-FishMOT_references/TrackEval")
    config = json.loads(replay_config.read_text())
    completed_keys = {(x["video"], x["seed"]) for x in state.get("completed", [])}

    try:
        for video, seed in tasks:
            if (video, seed) in completed_keys:
                continue
            if any(sha256(path) != digest for path, digest in guards.items()):
                raise ValueError("Guarded source drift")
            model = args.root / f"local_dense_context_local_attention_{video}_k4_s{seed}_v2"
            features = args.root / f"native_dino_frozen_{video}_k4_v1"
            output = output_root / f"native_local_attention_allfold_{video}_s{seed}_v1"
            evaluation = Path(str(output) + ".trackeval")
            log = Path(str(output) + ".log")
            state["active"] = dict(video=video, seed=seed, model=str(model), output=str(output))
            write_json(args.state, state)
            model_manifest = json.loads((model / "manifest.json").read_text())
            if model_manifest.get("status") != "COMPLETE" or model_manifest["guard"]["outer"] != video:
                raise ValueError(f"Invalid outer-fold model: {model}")
            if not output.exists():
                command = [sys.executable, str(replay_path), "--config", str(replay_config),
                           "--features", str(features), "--detections", str(detections),
                           "--output", str(output), "--video", video, "--memory", "fifo",
                           "--model", str(model), "--context-identity"]
                if run_checked(command, cwd=args.repo, log_path=log, env=env):
                    raise RuntimeError(f"native replay failed: {video} seed{seed}")
            replay_manifest = json.loads((output / "manifest.json").read_text())
            if replay_manifest.get("status") != "COMPLETE_NATIVE_CURRENT_IDENTITY_REPLAY":
                raise ValueError(f"Replay is not complete: {output / 'manifest.json'}")
            if not evaluation.exists():
                evaluation.mkdir()
                protocol = json.loads(evaluation_config.read_text())
                protocol.update(train=[video], dev=[], historical_test=[],
                                role="Offline official scoring of one declared native outer video")
                write_json(evaluation / "protocol.json", protocol)
                command = [sys.executable, str(evaluate_path), "--data-root", str(data_root),
                           "--derived-gt-root", str(derived_gt_root), "--protocol",
                           str(evaluation / "protocol.json"), "--partition", "train",
                           "--predictions", str(output), "--output", str(evaluation),
                           "--trackeval-root", str(trackeval_root)]
                if run_checked(command, cwd=args.repo, log_path=evaluation / "evaluation.log", env=env):
                    raise RuntimeError(f"official scoring failed: {video} seed{seed}")
            metrics = json.loads((evaluation / "metrics.json").read_text())
            artifact = metrics["artifacts"][video]
            if artifact["prediction_sha256"] != replay_manifest["prediction_sha256"]:
                raise ValueError("Official metrics do not reference replay prediction")
            row = dict(video=video, seed=seed, mode="local_attention", memory="fifo",
                       model=str(model), model_manifest_sha256=sha256(model / "manifest.json"),
                       model_checkpoint_sha256=model_manifest["checkpoint_sha256"],
                       feature_manifest_sha256=replay_manifest["feature_manifest_sha256"],
                       replay_manifest_sha256=sha256(output / "manifest.json"),
                       prediction_sha256=replay_manifest["prediction_sha256"],
                       official_metrics_sha256=sha256(evaluation / "metrics.json"),
                       metrics=metrics["metrics"][video],
                       association_seconds=replay_manifest["association_seconds"],
                       output_rows=replay_manifest["output_rows"])
            state["completed"].append(row)
            state["active"] = None
            write_json(args.state, state)
            completed_keys.add((video, seed))
            print(json.dumps(dict(video=video, seed=seed, HOTA=row["metrics"]["HOTA"],
                                  IDF1=row["metrics"]["IDF1"], completed=len(state["completed"]))),
                  flush=True)
        report = dict(status="COMPLETE_NATIVE_LOCAL_ATTENTION_ALLFOLD", config=json.loads(config_path.read_text()),
                      runs=state["completed"], state_sha256=sha256(args.state),
                      sources=snapshot_sources(args.report.with_suffix(".sources"), source_paths),
                      scope=state["scope"],
                      limitation="Cached native association replay excludes raw detector/IR/encoder startup and is not an end-to-end FPS claim.")
        write_json(args.report, report)
        state.update(status="COMPLETE_NATIVE_LOCAL_ATTENTION_ALLFOLD", report=str(args.report), active=None)
        write_json(args.state, state)
    except Exception as error:
        state.update(status="HALTED_ACTUAL_FAILURE", error=str(error), active=state.get("active"))
        write_json(args.state, state)
        raise


if __name__ == "__main__":
    main()
