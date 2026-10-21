#!/usr/bin/env python3
"""Audit prefix causality of the RGB-derived auxiliary fallback."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from yolox.data.causal_auxiliary import CausalAuxiliaryGenerator


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def frame_paths(root, split, sequence):
    folder = Path(root) / split / sequence / "visible"
    paths = sorted(folder.glob("*.jpg"), key=lambda p: int(p.stem))
    if not paths:
        raise FileNotFoundError(folder)
    return paths


def audit_sequence(root, split, sequence, prefix_length):
    paths = frame_paths(root, split, sequence)
    sample_paths = paths[: max(prefix_length + 3, 8)]
    frames = []
    for path in sample_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"unable to decode {path}")
        frames.append(frame)

    first = CausalAuxiliaryGenerator(alpha=0.01)
    outputs_a = [first(frame, sequence_id=sequence, frame_id=i + 1) for i, frame in enumerate(frames)]

    second = CausalAuxiliaryGenerator(alpha=0.01)
    outputs_b = []
    for index, frame in enumerate(frames):
        if index >= prefix_length:
            # Deliberately replace the suffix with a different current frame.
            frame = np.ascontiguousarray(255 - frame)
        outputs_b.append(second(frame, sequence_id=sequence, frame_id=index + 1))

    prefix_equal = all(
        np.array_equal(outputs_a[index], outputs_b[index])
        for index in range(prefix_length)
    )
    reset_a = CausalAuxiliaryGenerator(alpha=0.01)
    reset_b = CausalAuxiliaryGenerator(alpha=0.01)
    first_a = reset_a(frames[0], sequence_id=sequence, frame_id=1)
    first_b = reset_b(frames[0], sequence_id=sequence, frame_id=1)
    reset_equal = np.array_equal(first_a, first_b)
    return {
        "sequence": sequence,
        "split": split,
        "available_frames": len(paths),
        "audited_frames": len(frames),
        "prefix_length": prefix_length,
        "prefix_equal_after_future_perturbation": bool(prefix_equal),
        "sequence_reset_equal": bool(reset_equal),
        "first_output_sha256": sha256_bytes(outputs_a[0].tobytes()),
        "prefix_output_sha256": sha256_bytes(
            b"".join(outputs_a[index].tobytes() for index in range(prefix_length))
        ),
        "status": "PASS" if prefix_equal and reset_equal else "FAIL",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("MFT26IR_ROOT", "/data3/liuyeqiang/datasets/mft26-ir"))
    parser.add_argument("--protocol", default="configs/cfp/protocol_v2.json")
    parser.add_argument("--output", default="docs/fusion_research/CAUSAL_AUXILIARY_MODALITY_AUDIT.json")
    parser.add_argument("--prefix-length", type=int, default=4)
    args = parser.parse_args()

    protocol = json.load(open(args.protocol, encoding="utf-8"))
    rows = []
    for split_key, split in (("train", "train"), ("dev", "train"), ("historical_test", "test")):
        for sequence in protocol[split_key]:
            rows.append(audit_sequence(args.data_root, split, sequence, args.prefix_length))
    report = {
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "generator": {
            "name": "CausalAuxiliaryGenerator",
            "source": "yolox/data/causal_auxiliary.py",
            "rule": "current luminance minus EMA(prefix) background, output before update",
            "alpha": 0.01,
            "response_scale": 4.0,
            "physical_sensor": False,
            "future_frames": False,
            "sequence_state_reset": True,
        },
        "data_root": os.path.abspath(args.data_root),
        "protocol": os.path.abspath(args.protocol),
        "rows": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    md_path = os.path.splitext(args.output)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Causal auxiliary modality audit\n\n")
        handle.write(
            "The strict fallback is a causal RGB-derived auxiliary representation, "
            "not a physical thermal sensor. It computes the current luminance "
            "difference from a prefix EMA background and updates the background "
            "only after emitting the current output.\n\n"
        )
        handle.write(f"Overall status: **{report['status']}**.\n\n")
        handle.write("| Split | Sequence | Frames sampled | Prefix invariant | Reset invariant | Status |\n")
        handle.write("|---|---|---:|---|---|---|\n")
        for row in rows:
            handle.write(
                f"| {row['split']} | {row['sequence']} | {row['audited_frames']} | "
                f"{row['prefix_equal_after_future_perturbation']} | "
                f"{row['sequence_reset_equal']} | {row['status']} |\n"
            )
        handle.write(
            "\nThe archived historical `infrared` folders remain a `legacy` intervention. "
            "They are not silently relabelled as independent TIR measurements; the "
            "strict online candidate uses this fallback or a separately verified "
            "causal cache.\n"
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

