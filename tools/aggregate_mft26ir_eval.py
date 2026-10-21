#!/usr/bin/env python3
"""Merge sequence-sharded evaluator outputs and compute one protocol summary."""

import argparse
import json
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

from tools.eval_mft26ir import compute_motmetrics, run_trackeval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", action="append", required=True)
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--data-root", default=os.environ.get("MFT26IR_ROOT", "/data3/liuyeqiang/datasets/mft26-ir"))
    parser.add_argument("--gt-root", default=os.environ.get("MFT26IR_GT_ROOT", "/data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt"))
    parser.add_argument("--trackeval-root", default="/data3/liuyeqiang/IR-MFT_assets/vendor/VUP-PFTrack")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--fusion-mode", default=None)
    parser.add_argument("--auxiliary-mode", default=None)
    parser.add_argument("--skip-trackeval", action="store_true")
    args = parser.parse_args()

    sequences = [value.strip() for value in args.sequences.split(",") if value.strip()]
    if len(set(sequences)) != len(sequences):
        raise ValueError("duplicate sequences in aggregate request")
    os.makedirs(os.path.join(args.output, "track_results"), exist_ok=True)
    sources = {}
    for result_dir in args.result_dir:
        for sequence in sequences:
            source = os.path.join(result_dir, "track_results", f"{sequence}.txt")
            if os.path.isfile(source):
                if sequence in sources:
                    raise ValueError(f"sequence {sequence} appears in multiple shards")
                sources[sequence] = source
    missing = sorted(set(sequences) - set(sources))
    if missing:
        raise FileNotFoundError(f"missing sequence outputs: {missing}")
    for sequence, source in sources.items():
        shutil.copyfile(source, os.path.join(args.output, "track_results", f"{sequence}.txt"))

    data_split = "test" if args.split == "test" else "train"
    metrics = {
        "motmetrics": compute_motmetrics(
            os.path.join(args.output, "track_results"),
            args.gt_root,
            sequences,
            data_split=data_split,
        ),
        "protocol": {
            "split": args.split,
            "data_name": data_split,
            "sequences": sequences,
            "checkpoint": os.path.abspath(args.checkpoint) if args.checkpoint else None,
            "fusion_mode": args.fusion_mode,
            "auxiliary_mode": args.auxiliary_mode,
            "shards": [os.path.abspath(path) for path in args.result_dir],
        },
    }
    if not args.skip_trackeval:
        metrics["trackeval"] = run_trackeval(
            os.path.join(args.output, "track_results"),
            args.gt_root,
            args.output,
            sequences,
            args.trackeval_root,
            data_split=data_split,
        )
    with open(os.path.join(args.output, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
