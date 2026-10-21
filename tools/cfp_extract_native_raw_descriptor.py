#!/usr/bin/env python3
"""Produce current-proposal-only fixed raw RGB descriptors for one video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from cfp.dinov2_native_encoder import current_proposal_crops
from cfp.provenance import sha256, snapshot_sources, write_json
from cfp.raw_identity_descriptor import raw_rgb_descriptor


def idle_gpu(physical: str) -> None:
    if physical not in ("0", "1"):
        raise ValueError("Raw descriptor pilot is restricted to idle physical GPU 0/1")
    memory, utilization = [int(value.strip()) for value in subprocess.check_output(
        ["nvidia-smi", "-i", physical, "--query-gpu=memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True).split(",")]
    if memory > 64 or utilization:
        raise ValueError("Selected GPU is not idle; other jobs are protected")


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("config", "data-root", "frame-index", "detections", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    physical = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if (config.get("state") != "FROZEN" or args.video not in config["videos"]
            or args.output.exists() or args.output.resolve().is_relative_to(Path.cwd().resolve())):
        raise ValueError("Frozen pilot video and immutable outside-Git output are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    idle_gpu(physical)
    frame_manifest = json.loads((args.frame_index / "manifest.json").read_text())
    detection_manifest = json.loads((args.detections / "manifest.json").read_text())
    frame_path = args.frame_index / (args.video + ".frames.json")
    detection_path = args.detections / (args.video + ".npz")
    if (frame_manifest.get("status") != "COMPLETE_INDEX_ONLY"
            or frame_manifest.get("data_root") != str(args.data_root)
            or frame_manifest["detection_manifest_sha256"] != sha256(args.detections / "manifest.json")
            or sha256(frame_path) != frame_manifest["videos"][args.video]["frame_paths_sha256"]
            or sha256(detection_path) != detection_manifest["caches"][args.video]["sha256"]):
        raise ValueError("Current frame index/proposal provenance differs")
    paths = json.loads(frame_path.read_text())
    with np.load(detection_path, allow_pickle=False) as archive:
        frames, offsets, boxes, scores = [archive[name].copy() for name in
                                          ("frame_ids", "offsets", "boxes", "scores")]
    if (not np.array_equal(frames, np.arange(1, len(frames) + 1))
            or len(paths) != len(frames) or len(offsets) != len(frames) + 1
            or offsets[0] != 0 or offsets[-1] != len(boxes)
            or np.any(np.diff(offsets) < 0) or not np.isfinite(boxes).all()
            or not np.isfinite(scores).all()):
        raise ValueError("Every current frame and proposal row must remain in source order")
    for row in paths:
        relative = Path(row["rgb_file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Frame paths must remain below the declared data root")
    reserve_mount = Path(config["guards"]["reserve_mount"])
    if (torch.cuda.is_available() is False
            or reserve_mount.stat().st_dev != args.output.parent.stat().st_dev
            and config["guards"].get("allow_cross_mount_output", True) is False):
        raise ValueError("GPU and declared output/reserve mounts are required")
    if reserve_mount.stat().st_dev == args.data_root.stat().st_dev:
        if reserve_mount.stat().st_dev != args.output.parent.stat().st_dev:
            # Outputs may live on the root filesystem, but the /data3 reserve
            # is checked explicitly below and is the scientific storage guard.
            pass
    if __import__("shutil").disk_usage(reserve_mount).free < int(config["guards"]["minimum_free_bytes"]):
        raise OSError("40GiB reserve on the scientific asset mount")
    idle_gpu(physical)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.2)
    args.output.mkdir(parents=True)
    report = dict(status="RUNNING", experiment_id=config["experiment_id"], config=config,
                  config_sha256=sha256(args.config), video=args.video, physical_gpu=physical,
                  frame_paths_sha256=sha256(frame_path), detection_sha256=sha256(detection_path),
                  detection_manifest_sha256=sha256(args.detections / "manifest.json"),
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  sources=snapshot_sources(args.output / "sources", [
                      __file__, args.config, "cfp/raw_identity_descriptor.py",
                      "cfp/dinov2_native_encoder.py", "cfp/identity_data.py",
                      "cfp/local_identity_encoder.py", "cfp/provenance.py"]),
                  frames=len(frames), proposals=len(boxes), completed_frames=0,
                  scope="Current RGB crop/proposal only; no GT, physical IDs, future images, teacher or IR.")
    write_json(args.output / "manifest.json", report)
    np.save(args.output / "frame_ids.npy", frames, allow_pickle=False)
    np.save(args.output / "offsets.npy", offsets, allow_pickle=False)
    descriptor = np.lib.format.open_memmap(args.output / "raw_descriptor.npy", mode="w+",
                                           dtype=np.float32, shape=(len(boxes), 384))
    valid = np.lib.format.open_memmap(args.output / "valid.npy", mode="w+",
                                      dtype=np.bool_, shape=(len(boxes),))
    digest = hashlib.sha256()
    frame_seconds = []
    started = time.monotonic()
    try:
        for frame_index, frame in enumerate(frames):
            frame_start = time.perf_counter()
            image = cv2.imread(str(args.data_root / "train" / paths[frame_index]["rgb_file"]), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Current RGB decode failed")
            lo, hi = offsets[frame_index:frame_index + 2]
            crops, masks, crop_valid = current_proposal_crops(image, boxes[lo:hi])
            with torch.inference_mode():
                values = raw_rgb_descriptor(crops.to("cuda"), masks.to("cuda"))
            descriptor[lo:hi] = values.float().cpu().numpy()
            valid[lo:hi] = crop_valid
            digest.update(int(frame).to_bytes(8, "little"))
            digest.update(hashlib.sha256(image.tobytes()).digest())
            digest.update(boxes[lo:hi].tobytes())
            frame_seconds.append(time.perf_counter() - frame_start)
            report["completed_frames"] = frame_index + 1
            if (frame_index + 1) % 128 == 0 or frame_index + 1 == len(frames):
                descriptor.flush(); valid.flush()
                write_json(args.output / "manifest.json", report)
                print(json.dumps(dict(video=args.video, frames=frame_index + 1,
                                      total=len(frames), elapsed=time.monotonic() - started)), flush=True)
                if __import__("shutil").disk_usage(reserve_mount).free < int(config["guards"]["minimum_free_bytes"]):
                    raise OSError("40GiB reserve on the scientific asset mount")
        descriptor.flush(); valid.flush()
        report.update(status="COMPLETE_NATIVE_RAW_DESCRIPTOR", current_input_digest=digest.hexdigest(),
                      elapsed_seconds=time.monotonic() - started,
                      current_frame_latency_p50_p95_ms=(1000 * np.percentile(frame_seconds, [50, 95])).tolist(),
                      valid_crop_proposals=int(valid.sum()),
                      artifacts={"raw_descriptor": {"path": str(args.output / "raw_descriptor.npy"),
                          "sha256": sha256(args.output / "raw_descriptor.npy"),
                          "shape": list(descriptor.shape), "dtype": str(descriptor.dtype)},
                                 "valid": {"path": str(args.output / "valid.npy"),
                          "sha256": sha256(args.output / "valid.npy"),
                          "shape": list(valid.shape), "dtype": str(valid.dtype)}},
                      frame_ids_sha256=sha256(args.output / "frame_ids.npy"),
                      offsets_sha256=sha256(args.output / "offsets.npy"))
        write_json(args.output / "manifest.json", report)
    except Exception as error:
        report.update(status="FAILED_ACTUAL_EXECUTION", error=str(error))
        write_json(args.output / "manifest.json", report)
        raise


if __name__ == "__main__":
    main()
