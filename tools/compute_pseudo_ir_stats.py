#!/usr/bin/env python3
"""Exact Train-only RGB-order legacy auxiliary statistics; no image copies."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import cv2
import numpy as np


def histogram(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.dtype != np.uint8:
        raise ValueError(f"Invalid legacy auxiliary image: {path}")
    return np.stack([np.bincount(image[:, :, c].ravel(), minlength=256) for c in [2, 1, 0]])


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data-root", default="/data3/liuyeqiang/datasets/mft26-ir")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    cv2.setNumThreads(1)
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "configs/cfp/protocol_v2.json").read_text())
    annotation = json.loads((Path(args.data_root) / "annotations/train.json").read_text())
    paths = []
    for image in annotation["images"]:
        name = Path(image["ir_file_name"])
        if name.parts[0] in protocol["train"]:
            paths.append(Path(args.data_root) / "train" / name)
    if len(paths) != protocol["frame_counts"]["train"]:
        raise ValueError("Train coverage mismatch")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} Train images: {missing[:3]}")
    counts = np.zeros((3, 256), np.int64)
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n, hist in enumerate(pool.map(histogram, paths), 1):
            counts += hist
            if n % 1000 == 0:
                print(json.dumps(dict(frames=n, total=len(paths), seconds=time.monotonic()-start)), flush=True)
    lut = np.arange(256, dtype=np.float64) / 255.0
    probability = counts / counts.sum(axis=1, keepdims=True)
    mean = probability @ lut
    std = np.sqrt(np.maximum(probability @ (lut**2) - mean**2, 0))
    result = dict(complete=True, split="train", frames=len(paths), pixels=int(counts[0].sum()),
                  mean_0_1=mean.tolist(), std_0_1=std.tolist(), channel_order="RGB",
                  representation="archived legacy infrared JPEG, decoded BGR and reordered RGB",
                  histogram=counts.tolist(), source_root=args.data_root,
                  seconds=time.monotonic()-start)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "histogram"}), flush=True)


if __name__ == "__main__":
    main()
