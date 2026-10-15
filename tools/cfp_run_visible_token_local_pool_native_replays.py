#!/usr/bin/env python3
"""Pre-registered fit-free local-token native association control.

The accepted visible-token feature banks already contain current-frame CLS,
K4 local tokens and geometric visibility fractions.  This driver derives a
visibility-weighted K4 mean in a separate external feature root, then calls
the unchanged native FIFO replay and scoped official evaluator.  It never
modifies an accepted feature bank or replay output.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfp.provenance import sha256, snapshot_sources, write_json


VIDEOS = ("BT-001", "BT-003", "MSK-002", "PF-001", "SN-001", "SN-015")
MODES = ("masked_visible_tokens", "visible_target")
SEEDS = (0, 1, 2)


def derive_bank(source: Path, destination: Path):
    """Create an immutable derived manifest and pooled embedding array."""
    if destination.exists():
        manifest = destination / "manifest.json"
        if not manifest.exists():
            raise RuntimeError(f"Incomplete derived feature directory: {destination}")
        return json.loads(manifest.read_text())
    source_manifest = json.loads((source / "manifest.json").read_text())
    if source_manifest.get("status") != "COMPLETE_NATIVE_CURRENT_VISIBLE_TOKEN_FEATURES":
        raise ValueError(f"Unexpected source feature status: {source}")
    for name in ("frame_ids", "offsets", "local4", "visibility4", "valid", "raw_cls"):
        if not (source / f"{name}.npy").exists():
            raise FileNotFoundError(source / f"{name}.npy")
    destination.mkdir(parents=True)
    for name in ("frame_ids", "offsets", "local4", "visibility4", "valid", "raw_cls"):
        (destination / f"{name}.npy").symlink_to(source / f"{name}.npy")

    local4 = np.load(source / "local4.npy", mmap_mode="r")
    visibility = np.load(source / "visibility4.npy", mmap_mode="r")
    valid = np.load(source / "valid.npy", mmap_mode="r")
    if local4.ndim != 3 or local4.shape[1:] != (4, 384) or visibility.shape != local4.shape[:2]:
        raise ValueError("Expected K4 x 384 local features and aligned visibility")
    if len(valid) != len(local4):
        raise ValueError("Local feature and validity lengths differ")
    pooled = np.lib.format.open_memmap(destination / "embedding.npy", mode="w+", dtype=np.float32,
                                        shape=(len(local4), 384))
    for start in range(0, len(local4), 65536):
        end = min(len(local4), start + 65536)
        x = np.asarray(local4[start:end], dtype=np.float32)
        w = np.asarray(visibility[start:end], dtype=np.float32).clip(0.0, 1.0)
        denominator = w.sum(axis=1, keepdims=True)
        x = (x * w[..., None]).sum(axis=1) / np.maximum(denominator, 1e-6)
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(norm, 1e-12)
        x[~np.asarray(valid[start:end], dtype=bool)] = 0.0
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite pooled local feature")
        pooled[start:end] = x
    pooled.flush()
    del pooled
    derived = dict(source_manifest)
    derived.update(
        status="COMPLETE_NATIVE_CURRENT_VISIBLE_TOKEN_FEATURES",
        derived_from=str(source),
        derived_channel="visibility_weighted_local4_then_l2_normalize",
        derived_source_manifest_sha256=sha256(source / "manifest.json"),
        derived_source_local4_sha256=sha256(source / "local4.npy"),
        derived_source_visibility4_sha256=sha256(source / "visibility4.npy"),
        derived_source_valid_sha256=sha256(source / "valid.npy"),
        derived_pool_definition=(
            "For each proposal, weighted mean of the four real geometric local4 vectors "
            "with visibility4 fractions, divide by max(sum visibility4,1e-6), then L2-normalize; "
            "invalid proposals are zeroed."
        ),
    )
    artifacts = dict(derived.get("artifacts", {}))
    embedding_artifact = dict(artifacts.get("embedding", {}))
    embedding_artifact.update(path=str(destination / "embedding.npy"), sha256=sha256(destination / "embedding.npy"),
                              dtype="float32", shape=[int(len(local4)), 384])
    artifacts["embedding"] = embedding_artifact
    derived["artifacts"] = artifacts
    write_json(destination / "manifest.json", derived)
    return derived


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
    ap.add_argument("--derived-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--only", nargs=3, metavar=("VIDEO", "MODE", "SEED"), help="Run one pre-registered probe")
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("state") != "FROZEN" or config.get("pool") != "visibility_weighted_local4_then_l2_normalize":
        raise ValueError("Frozen local-pool configuration guard failed")
    tasks = []
    allowed = None
    if args.only:
        allowed = (args.only[0], args.only[1], int(args.only[2]))
        if allowed[0] not in VIDEOS or allowed[1] not in MODES or allowed[2] not in SEEDS:
            raise ValueError("Probe is outside the registered all-fold plan")
    for video in VIDEOS:
        for mode in MODES:
            for seed in SEEDS:
                if allowed is not None and (video, mode, seed) != allowed:
                    continue
                source = args.feature_root / "features" / video / f"{mode}_s{seed}"
                derived = args.derived_root / "features" / video / f"{mode}_s{seed}"
                output = args.output_root / f"local_pool_{mode}_{video}_s{seed}_v1"
                evaluation = Path(str(output) + ".trackeval")
                tasks.append(dict(video=video, mode=mode, seed=seed, source=str(source),
                                  derived=str(derived), output=str(output), evaluation=str(evaluation),
                                  log=str(args.logs / f"{video}_{mode}_s{seed}.log"), status="PENDING"))
    source_paths = [Path(__file__), args.config,
                    args.repo / "tools/cfp_replay_native_current_identity.py",
                    args.repo / "tools/cfp_evaluate.py",
                    args.repo / "configs/cfp/phase_vii_native_current_identity_replay_v1.json",
                    args.repo / "cfp/native_local_identity.py",
                    args.repo / "cfp/predictive_association.py",
                    args.repo / "cfp/predictive.py", args.repo / "cfp/association.py",
                    args.repo / "cfp/provenance.py"]
    guards = {str(p): sha256(p) for p in source_paths}
    args.derived_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.logs.mkdir(parents=True, exist_ok=True)
    state = dict(status="RUNNING", config=str(args.config), config_sha256=sha256(args.config),
                 feature_root=str(args.feature_root), derived_root=str(args.derived_root),
                 output_root=str(args.output_root), source_guards=guards, tasks=tasks,
                 scope=config["scope"], decision_rule=config["decision_rule"],
                 sources=snapshot_sources(args.state.with_suffix(".sources"), source_paths))
    write_json(args.state, state)
    env = dict(os.environ, PYTHONPATH=str(args.repo), CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    detections = Path("/data3/liuyeqiang/IR-MFT_assets/cfp/phase0/causal_v3_train_detections")
    data_root = Path("/data3/liuyeqiang/datasets/mft26-ir-causal-v1")
    derived_gt = Path("/data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt")
    trackeval = Path("/data3/liuyeqiang/CFP-FishMOT_references/TrackEval")
    replay_config = args.repo / "configs/cfp/phase_vii_native_current_identity_replay_v1.json"
    replay = args.repo / "tools/cfp_replay_native_current_identity.py"
    evaluate = args.repo / "tools/cfp_evaluate.py"
    protocol_template = args.repo / "configs/cfp/phase_vii_joint_evaluation_subset_v1.json"
    for task in tasks:
        source = Path(task["source"])
        derived = Path(task["derived"])
        output = Path(task["output"])
        evaluation = Path(task["evaluation"])
        task["status"] = "DERIVING_FEATURES"
        state["active"] = task
        write_json(args.state, state)
        derive_bank(source, derived)
        if not output.exists():
            task["status"] = "RUNNING_REPLAY"
            write_json(args.state, state)
            command = [sys.executable, str(replay), "--config", str(replay_config),
                       "--features", str(derived), "--detections", str(detections),
                       "--output", str(output), "--video", task["video"], "--memory", "fifo"]
            if run_checked(command, args.repo, Path(task["log"]), env):
                task["status"] = "FAILED"
                state["error"] = "native replay returned nonzero"
                write_json(args.state, state)
                raise RuntimeError(state["error"])
        replay_manifest = output / "manifest.json"
        rm = json.loads(replay_manifest.read_text())
        if rm.get("status") != "COMPLETE_NATIVE_CURRENT_IDENTITY_REPLAY":
            raise RuntimeError(f"Replay did not complete: {replay_manifest}")
        if not evaluation.exists():
            task["status"] = "RUNNING_TRACKEVAL"
            evaluation.mkdir(parents=True)
            protocol = json.loads(protocol_template.read_text())
            protocol.update(train=[task["video"]], dev=[], historical_test=[],
                            role="Offline official scoring of pre-registered visibility-weighted local-token control")
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
            raise ValueError("TrackEval prediction does not match local-pool replay")
        task.update(status="COMPLETE", replay_manifest_sha256=sha256(replay_manifest),
                    metrics_sha256=sha256(metrics_path), metrics=video_metrics,
                    derived_manifest_sha256=sha256(derived / "manifest.json"))
        state["tasks"] = tasks
        state["active"] = None
        write_json(args.state, state)
        print(json.dumps(dict(video=task["video"], mode=task["mode"], seed=task["seed"],
                              HOTA=video_metrics["HOTA"], IDF1=video_metrics["IDF1"],
                              IDSW=video_metrics["IDSW"])), flush=True)
    state.update(status="COMPLETE_LOCAL_POOL_NATIVE_REPLAYS", finished_unix=time.time(), active=None)
    write_json(args.state, state)
    report = dict(status="COMPLETE_LOCAL_POOL_NATIVE_REPLAYS", config=str(args.config),
                  config_sha256=sha256(args.config), feature_root=str(args.feature_root),
                  derived_root=str(args.derived_root), tasks=tasks, source_guards=guards,
                  state_sha256=sha256(args.state), sources=snapshot_sources(args.report.with_suffix(".sources"), source_paths),
                  scope=config["scope"], decision_rule=config["decision_rule"],
                  limitation="Fit-free local-token control only; feature production and native association replay are measured, not raw end-to-end FPS.")
    write_json(args.report, report)


if __name__ == "__main__":
    main()
