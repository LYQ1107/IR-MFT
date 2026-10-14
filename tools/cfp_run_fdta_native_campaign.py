#!/usr/bin/env python3
"""Run three FDTA-style adapter modes × three seeds on both held-out videos."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--existing-root", type=Path)
    args = parser.parse_args()
    if args.state.exists() or os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", None):
        raise ValueError("New CPU-only native campaign state required")
    args.root.mkdir(parents=True, exist_ok=True)
    config_path = Path("configs/cfp/phase_vii_fdta_adapter_v1.json")
    config = json.loads(config_path.read_text())
    modes, seeds, videos = config["modes"], config["seed_plan"], [config["inner"], config["outer"]]
    source_files = [__file__, "tools/cfp_replay_fdta_native.py", "tools/cfp_evaluate_single.py", "tools/cfp_evaluate.py", "cfp/fdta_fish_adapter.py", "cfp/fdta_native.py", "cfp/native_local_identity.py", "cfp/predictive_association.py", "cfp/predictive.py", "cfp/association.py", str(config_path)]
    guards = {path: sha256(path) for path in source_files}
    tasks = [(mode, seed, video) for mode in modes for seed in seeds for video in videos]
    state = dict(status="RUNNING", tasks=tasks, completed=[], active=None, source_guards=guards,
                 sources=snapshot_sources(args.state.with_suffix(".sources"), source_files),
                 scope="Actual frozen-DINO current-proposal causal native FIFO replay of all registered FDTA-style modes/seeds on both declared held-out videos; no GT/teacher/future input online. Offline adapter fit is separate and four-video-only.")
    write_json(args.state, state)
    try:
        for mode, seed, video in tasks:
            if any(sha256(path) != value for path, value in guards.items()):
                raise ValueError("FDTA native source guard changed")
            training = args.training_root / f"fdta_adapter_{mode}_s{seed}_v1"
            if not (training / "manifest.json").exists():
                raise ValueError("Missing completed FDTA fit: " + str(training))
            name = f"native_fdta_{mode}_{video}_s{seed}_v1"
            candidate = args.existing_root / name if args.existing_root else None
            output = candidate if candidate is not None and (candidate / "manifest.json").exists() else args.root / name
            command = [sys.executable, "tools/cfp_replay_fdta_native.py", "--root", str(args.root.parent),
                       "--detections", str(args.detections), "--training", str(training), "--video", video,
                       "--memory", "fifo", "--output", str(output)]
            log = Path(str(output) + ".log")
            state["active"] = dict(mode=mode, seed=seed, video=video, output=str(output), command=command, log=str(log)); write_json(args.state, state)
            if not (output / "manifest.json").exists():
                with log.open("x") as stream:
                    code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
                if code:
                    raise RuntimeError(f"FDTA native child failed with {code}: {mode} seed {seed} video {video}")
            manifest = json.loads((output / "manifest.json").read_text())
            if manifest["status"] != "COMPLETE_NATIVE_FDTA_ADAPTER_REPLAY":
                raise RuntimeError("Native FDTA output did not complete: " + str(output))
            trackeval = output.with_name(output.name + ".trackeval")
            if not (trackeval / "metrics.json").exists():
                eval_log = Path(str(trackeval) + ".log")
                eval_log.parent.mkdir(parents=True, exist_ok=True)
                with eval_log.open("w") as stream:
                    code = subprocess.run([sys.executable, "tools/cfp_evaluate_single.py", "--predictions", str(output), "--video", video, "--output", str(trackeval)], stdout=stream, stderr=subprocess.STDOUT).returncode
                if code:
                    raise RuntimeError(f"FDTA TrackEval failed with {code}: {mode} seed {seed} video {video}")
            metrics = json.loads((trackeval / "metrics.json").read_text())
            row = dict(mode=mode, seed=seed, video=video, output=str(output), replay_manifest_sha256=sha256(output / "manifest.json"),
                       prediction_sha256=manifest["prediction_sha256"], official_metrics_path=str(trackeval / "metrics.json"),
                       official_metrics_sha256=sha256(trackeval / "metrics.json"), metrics=metrics["metrics"][video])
            state["completed"].append(row); state["active"] = None; write_json(args.state, state)
        state["status"] = "COMPLETE_FDTA_NATIVE_18_REPLAYS"; write_json(args.state, state)
        print(json.dumps({"status": state["status"], "completed": len(state["completed"])}))
    except Exception as error:
        state.update(status="HALTED_ACTUAL_FAILURE", error=str(error)); write_json(args.state, state); raise


if __name__ == "__main__": main()
