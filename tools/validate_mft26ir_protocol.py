#!/usr/bin/env python3
"""Validate the frozen MFT26-IR split before any training/evaluation run."""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_annotation(path, expected_sequences):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    videos = {int(video["id"]): video["file_name"] for video in payload["videos"]}
    frames = Counter()
    frame_ids = defaultdict(list)
    for image in payload["images"]:
        sequence = videos[int(image["video_id"])]
        frames[sequence] += 1
        frame_ids[sequence].append(int(image["frame_id"]))
    observed = sorted(frames)
    expected = sorted(expected_sequences)
    if observed != expected:
        raise AssertionError(f"{path}: sequences {observed} != {expected}")
    for sequence, values in frame_ids.items():
        if values != list(range(1, len(values) + 1)):
            raise AssertionError(f"{path}: non-contiguous frame IDs in {sequence}")
    return {
        "path": os.path.abspath(path),
        "sha256": sha256(path),
        "sequences": observed,
        "frames_by_sequence": dict(sorted(frames.items())),
        "frames": int(sum(frames.values())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("MFT26IR_ROOT", "/data3/liuyeqiang/datasets/mft26-ir"))
    parser.add_argument("--protocol", default="configs/cfp/protocol_v2.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.protocol, "r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    train = inspect_annotation(
        os.path.join(args.data_root, "annotations", "train.json"),
        protocol["train"] + protocol["dev"],
    )
    test = inspect_annotation(
        os.path.join(args.data_root, "annotations", "test.json"),
        protocol["historical_test"],
    )
    result = {
        "status": "PASS",
        "protocol_id": protocol["protocol_id"],
        "train_annotation": train,
        "test_annotation": test,
        "expected_test_frames": 15985,
    }
    if test["frames"] != result["expected_test_frames"]:
        raise AssertionError(
            f"test frame count {test['frames']} != {result['expected_test_frames']}"
        )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

