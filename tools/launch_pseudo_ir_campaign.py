#!/usr/bin/env python3
"""Run dependent per-GPU comparison jobs, preserving logs and failure states."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/run_pseudo_ir_comparison.py"
VISIBLE = "/data3/liuyeqiang/IR-MFT_assets/checkpoints/bytetrack_visible_last_epoch_ckpt.pth.tar"


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    os.replace(temporary, path)


def wait_json(path, predicate, description, state_path):
    while True:
        if path.is_file():
            value = json.loads(path.read_text())
            if predicate(value):
                return value
            raise RuntimeError(f"Invalid prerequisite for {description}: {path}")
        write(state_path, dict(status="WAITING", prerequisite=str(path), description=description,
                              updated_unix=time.time()))
        time.sleep(10)


def run_job(args, name, command):
    state_path = args.result_root / f"queue_gpu{args.gpu}.state.json"
    out = args.result_root / name
    completion = out / ("completion.json" if command[0] == "train" else "metrics.json")
    if completion.exists():
        done = json.loads(completion.read_text())
        if command[0] == "train" and done.get("status") != "COMPLETE":
            raise RuntimeError(f"Existing nonformal completion: {completion}")
        if command[0] == "infer" and done.get("protocol", {}).get("max_frames"):
            raise RuntimeError(f"Existing truncated evaluation: {completion}")
        return
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(SCRIPT)] + command + ["--output", str(out)]
    # Interrupted epoch-boundary runs resume the sole saved checkpoint.
    latest = out / "latest_ckpt.pth.tar"
    if command[3] == "train" and latest.is_file():
        checkpoint_index = command.index("--ckpt") + 1
        command[checkpoint_index] = str(latest)
        command += ["--resume"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    with (out / "process.log").open("a") as log:
        log.write("COMMAND " + json.dumps(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        write(state_path, dict(status="RUNNING", job=name, pid=process.pid, command=command,
                               physical_gpu=args.gpu, started_unix=time.time()))
        print(json.dumps(dict(job=name, pid=process.pid, physical_gpu=args.gpu)), flush=True)
        code = process.wait()
    if code:
        write(state_path, dict(status="FAILED", job=name, returncode=code,
                               log=str(out / "process.log"), updated_unix=time.time()))
        raise RuntimeError(f"Job failed: {name}; read {out / 'process.log'}")
    if not completion.is_file():
        raise RuntimeError(f"Job returned without its completion artifact: {name}")
    print("JOB_COMPLETE", name, flush=True)


def inference(args, arm, variant, checkpoint, extra=None):
    for split in ["dev", "test"]:
        run_job(args, f"eval_{arm}_{split}", ["infer", "--variant", variant,
            "--ckpt", str(checkpoint), "--split", split, "--batch-size", "4", "--workers", "4"] + (extra or []))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--gpu", type=int, choices=[2, 3], required=True)
    p.add_argument("--result-root", type=Path,
        default=Path("/data3/liuyeqiang/IR-MFT_assets/pseudo_ir_comparison_20261120"))
    args = p.parse_args()
    args.result_root.mkdir(parents=True, exist_ok=True)
    state = args.result_root / f"queue_gpu{args.gpu}.state.json"
    if args.gpu == 2:
        wait_json(args.result_root / "eval_rgb_original_dev/metrics.json",
                  lambda d:d.get("protocol", {}).get("frames") == 4834,
                  "manual original RGB Dev reference", state)
        # Reuse the completed Dev reference; recompute all Test frames now.
        inference(args, "rgb_original", "rgb", VISIBLE)
        stats = args.result_root / "old_legacy_train_stats.json"
        wait_json(stats, lambda d:d.get("complete") and d.get("frames") == 27251,
                  "full Train-only old auxiliary statistics", state)
        run_job(args, "train_old_legacy_s0", ["train", "--variant", "legacy", "--ckpt", VISIBLE,
            "--batch-size", "8", "--accumulate", "1", "--workers", "4", "--ir-stats", str(stats)])
        inference(args, "old_legacy", "legacy", args.result_root / "train_old_legacy_s0/latest_ckpt.pth.tar",
                  ["--ir-stats", str(stats)])
        # Train the RGB control while the NAS-backed new-input worker runs on
        # GPU3. This changes only scheduling, not the shared training budget.
        run_job(args, "train_rgb_matched_s0", ["train", "--variant", "rgb", "--ckpt", VISIBLE,
            "--batch-size", "8", "--accumulate", "1", "--workers", "4", "--calibration-epochs", "0"])
        inference(args, "rgb_matched", "rgb", args.result_root / "train_rgb_matched_s0/latest_ckpt.pth.tar")
    else:
        wait_json(args.result_root / "train_new_diffv2ir_s0/completion.json",
                  lambda d:d.get("status") == "COMPLETE" and d.get("epochs") == 6,
                  "manual new DiffV2IR V3 training", state)
        inference(args, "new_diffv2ir", "diffv2ir", args.result_root / "train_new_diffv2ir_s0/latest_ckpt.pth.tar")
    write(state, dict(status="COMPLETE", physical_gpu=args.gpu, updated_unix=time.time()))


if __name__ == "__main__":
    main()
