#!/usr/bin/env python3
"""Persistent all-fold native current-frame visible-token feature campaign.

One process is allowed per approved GPU.  The feature banks and checkpoints
stay outside Git; this driver stores only small state and provenance records.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Make the driver directly executable from an absolute path while preserving
# the repository package coordinate used by every child process.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfp.provenance import sha256, snapshot_sources, write_json


VIDEOS = ("BT-001", "BT-003", "MSK-002", "PF-001", "SN-001", "SN-015")
MODES = ("masked_visible_tokens", "visible_target")
SEEDS = (0, 1, 2)
GPUS = ("0", "1", "2", "3")


def gpu_idle(gpu):
    values = subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-gpu=memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True).strip().split(",")
    memory, utilization = (int(v.strip()) for v in values)
    return memory <= 64 and utilization == 0


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--logs", type=Path, required=True)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    if config["state"] != "FROZEN":
        raise ValueError("Frozen native feature configuration required")
    source_paths = [
        Path(__file__), args.config,
        args.repo / "tools/cfp_extract_native_visible_token_features.py",
        args.repo / "cfp/visible_token_native_encoder.py",
        args.repo / "cfp/visible_token_fish_mae.py",
        args.repo / "cfp/dinov2_native_encoder.py",
        args.repo / "cfp/dinov2_identity.py",
        args.repo / "cfp/identity_data.py",
        args.repo / "cfp/local_identity_encoder.py",
        args.repo / "cfp/provenance.py",
    ]
    guards = {str(p): sha256(p) for p in source_paths}
    tasks = []
    for video in VIDEOS:
        fold = json.loads((args.repo / f"configs/cfp/phase_vii_visible_token_outer_{video}_v1.json").read_text())
        for mode in MODES:
            for seed in SEEDS:
                output = args.output_root / "features" / video / f"{mode}_s{seed}"
                tasks.append(dict(video=video, mode=mode, seed=seed,
                                  inner=fold["excluded_videos"][0], fit=fold["fit_videos"],
                                  training=str(Path(config["pretrain_root"]) / video / f"{mode}_s{seed}"),
                                  output=str(output), log=str(args.logs / f"{video}_{mode}_s{seed}.log"),
                                  status="PENDING", pid=None, gpu=None, returncode=None))
    if args.state.exists():
        state = json.loads(args.state.read_text())
        if state.get("source_guards") != guards or state.get("tasks_declared") != [
                {k: t[k] for k in ("video", "mode", "seed", "inner", "fit", "training", "output", "log")}
                for t in tasks]:
            raise ValueError("Existing state has different source guards or task declaration")
        previous = {(x["video"], x["mode"], x["seed"]): x for x in state.get("tasks", [])}
        for task in tasks:
            old = previous.get((task["video"], task["mode"], task["seed"]))
            if old:
                for key in ("status", "pid", "gpu", "returncode", "started_unix", "finished_unix", "error", "command"):
                    if key in old:
                        task[key] = old[key]
    else:
        state = dict(status="PENDING", created_unix=time.time(), tasks=[], active=[],
                     source_guards=guards, tasks_declared=[
                         {k: t[k] for k in ("video", "mode", "seed", "inner", "fit", "training", "output", "log")}
                         for t in tasks], sources=snapshot_sources(args.state.with_suffix(".sources"), source_paths),
                     scope=config["scope"])
    active = {}
    for task in tasks:
        manifest = Path(task["output"]) / "manifest.json"
        if manifest.exists():
            try:
                report = json.loads(manifest.read_text())
                if report.get("status") == "COMPLETE_NATIVE_CURRENT_VISIBLE_TOKEN_FEATURES":
                    task["status"] = "COMPLETE"
                    task["returncode"] = 0
                elif task.get("status") == "RUNNING" and not alive(task.get("pid")):
                    task["status"] = "PENDING"
                    task["pid"] = None
            except (OSError, json.JSONDecodeError):
                pass

    def save():
        state["tasks"] = tasks
        state["active"] = [dict(gpu=gpu, video=t["video"], mode=t["mode"], seed=t["seed"], pid=t["pid"])
                           for gpu, t in ((g, p._task) for g, p in active.items())]
        state["updated_unix"] = time.time()
        write_json(args.state, state)

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.logs.mkdir(parents=True, exist_ok=True)
    save()
    try:
        while True:
            for gpu, process in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                task = process._task
                task["returncode"] = code
                task["finished_unix"] = time.time()
                active.pop(gpu)
                if code != 0:
                    task["status"] = "FAILED"
                    task["error"] = f"child returncode {code}"
                    save()
                    raise RuntimeError(f"native feature task failed: {task['video']} {task['mode']} s{task['seed']}")
                report_path = Path(task["output"]) / "manifest.json"
                report = json.loads(report_path.read_text())
                if report.get("status") != "COMPLETE_NATIVE_CURRENT_VISIBLE_TOKEN_FEATURES":
                    task["status"] = "FAILED"
                    task["error"] = "child returned zero without complete manifest"
                    save()
                    raise RuntimeError(f"incomplete native feature task: {report_path}")
                task["status"] = "COMPLETE"
                save()
                print(json.dumps(dict(video=task["video"], mode=task["mode"], seed=task["seed"],
                                      status=task["status"], completed=sum(x["status"] == "COMPLETE" for x in tasks))), flush=True)
            if all(t["status"] == "COMPLETE" for t in tasks):
                state["status"] = "COMPLETE_ALLFOLD_NATIVE_FEATURES_36"
                state["finished_unix"] = time.time()
                save()
                return
            state["status"] = "RUNNING"
            save()
            for gpu in GPUS:
                if gpu in active:
                    continue
                task = next((t for t in tasks if t["status"] == "PENDING"), None)
                if task is None or not gpu_idle(gpu):
                    continue
                output = Path(task["output"])
                if output.exists() and (output / "manifest.json").exists():
                    report = json.loads((output / "manifest.json").read_text())
                    if report.get("status") != "COMPLETE_NATIVE_CURRENT_VISIBLE_TOKEN_FEATURES":
                        raise RuntimeError(f"Refusing to overwrite existing incomplete output: {output}")
                output.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "tools/cfp_extract_native_visible_token_features.py",
                           "--config", str(args.config), "--data-root", config["data_root"],
                           "--frame-index", config["frame_index"], "--detections", config["detections"],
                           "--reference", config["dino_reference"], "--weights", config["dino_weights"],
                           "--source-proof", config["dino_source_proof"], "--training", task["training"],
                           "--output", task["output"], "--video", task["video"], "--inner", task["inner"],
                           "--mode", task["mode"], "--seed", str(task["seed"]), "--fit", *task["fit"]]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(args.repo),
                           OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                           CUBLAS_WORKSPACE_CONFIG=":4096:8")
                stream = Path(task["log"]).open("a")
                process = subprocess.Popen(command, cwd=str(args.repo), env=env,
                                           stdout=stream, stderr=subprocess.STDOUT)
                process._task = task
                task.update(status="RUNNING", pid=process.pid, gpu=gpu, started_unix=time.time(), command=command)
                active[gpu] = process
                save()
            time.sleep(5)
    except BaseException as error:
        state["status"] = "HALTED_ACTUAL_FAILURE"
        state["error"] = repr(error)
        save()
        for process in active.values():
            try:
                process.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
        raise


if __name__ == "__main__":
    main()
