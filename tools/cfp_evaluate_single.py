#!/usr/bin/env python3
"""Run the pinned TrackEval evaluator in a fresh process for one sequence."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from cfp_evaluate import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(Path("/data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt"), "train", [args.video],
                      args.predictions, args.output, Path("/data3/liuyeqiang/CFP-FishMOT_references/TrackEval"))
    print(json.dumps(result["metrics"][args.video]))


if __name__ == "__main__": main()
