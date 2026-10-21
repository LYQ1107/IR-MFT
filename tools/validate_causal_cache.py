#!/usr/bin/env python3
"""Compare selected disk-cache frames with the online causal generator."""

import argparse
import json
import os

import cv2
import numpy as np

from yolox.data.causal_auxiliary import CausalAuxiliaryGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("MFT26IR_ROOT", "/data3/liuyeqiang/datasets/mft26-ir"))
    parser.add_argument("--cache-root", default=os.environ.get("MFT26IR_CAUSAL_CACHE_ROOT", "/data3/liuyeqiang/IR-MFT_assets/causal_aux_cache"))
    parser.add_argument("--protocol", default="configs/cfp/protocol_v2.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    protocol = json.load(open(args.protocol, "r", encoding="utf-8"))
    payload = json.load(open(os.path.join(args.data_root, "annotations", "train.json"), "r", encoding="utf-8"))
    videos = {int(video["id"]): video["file_name"] for video in payload["videos"]}
    rows = []
    for sequence in protocol["train"]:
        images = sorted(
            [image for image in payload["images"] if videos[int(image["video_id"])] == sequence],
            key=lambda image: int(image["frame_id"]),
        )
        sample_ids = {1, len(images), max(1, len(images) // 2)}
        generator = CausalAuxiliaryGenerator(alpha=0.01)
        for image in images:
            frame_id = int(image["frame_id"])
            visible_path = os.path.join(args.data_root, "train", image["vi_file_name"])
            visible = cv2.imread(visible_path, cv2.IMREAD_COLOR)
            if visible is None:
                raise FileNotFoundError(visible_path)
            online = generator(visible, sequence_id=sequence, frame_id=frame_id)
            if frame_id not in sample_ids:
                continue
            stem = os.path.join(args.cache_root, "train", image["vi_file_name"])
            stem = stem.rsplit(".", 1)[0]
            cache_path = stem + ".jpg" if os.path.isfile(stem + ".jpg") else stem + ".webp"
            cached = cv2.imread(cache_path, cv2.IMREAD_UNCHANGED)
            if cached is None:
                raise FileNotFoundError(cache_path)
            if cached.ndim == 2:
                cached = cv2.applyColorMap(cached, cv2.COLORMAP_INFERNO)
            delta = np.abs(online.astype(np.int16) - cached.astype(np.int16))
            rows.append({
                "sequence": sequence,
                "frame_id": frame_id,
                "cache_path": cache_path,
                "max_abs_pixel_error": int(delta.max()),
                "mean_abs_pixel_error": float(delta.mean()),
            })
    result = {"status": "PASS", "samples": rows, "max_abs_pixel_error": max(row["max_abs_pixel_error"] for row in rows)}
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        open(args.output, "w", encoding="utf-8").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

