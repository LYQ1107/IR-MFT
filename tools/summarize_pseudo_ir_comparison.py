#!/usr/bin/env python3
"""Validate matched coverage and summarize completed pseudo-IR comparisons."""
import argparse
import hashlib
import json
from pathlib import Path


ARMS = {
    "rgb_original": "Original RGB ByteTrack",
    "old_legacy": "V3 + old legacy IR",
    "new_diffv2ir": "V3 + new DiffV2IR",
    "rgb_matched": "RGB matched fine-tune + ByteTrack",
}
METRICS = ["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW"]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--result-root", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-markdown", required=True)
    p.add_argument("--require-complete", action="store_true")
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    protocol = json.loads((repo / "configs/pseudo_ir_comparison_20261120.json").read_text())
    result = dict(status="COMPLETE", protocol=protocol, evaluations=[], deltas={}, missing=[])
    table = ["| Method | Split | Frames | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    lookup = {}
    for arm, title in ARMS.items():
        for split in ["dev", "test"]:
            path = Path(args.result_root) / ("eval_" + arm + "_" + split) / "metrics.json"
            if not path.is_file():
                result["missing"].append(str(path))
                continue
            raw = path.read_bytes()
            d = json.loads(raw)
            actual = d["protocol"]
            if actual["frames"] != protocol["frames"][split]:
                raise ValueError(f"Incomplete frame coverage: {path}")
            if set(actual["sequences"]) != set(protocol[split + "_sequences"]):
                raise ValueError(f"Sequence coverage mismatch: {path}")
            for key in ["conf", "nms", "track_thresh", "track_buffer", "match_thresh"]:
                if actual[key] != protocol["evaluation"][key]:
                    raise ValueError(f"Evaluation configuration mismatch: {key}, {path}")
            metrics = {key: d["trackeval"][key] for key in METRICS}
            record = dict(arm=arm, method=title, split=split, frames=actual["frames"],
                          metrics=metrics, runtime=d["runtime"], protocol=actual,
                          metrics_path=str(path), metrics_sha256=hashlib.sha256(raw).hexdigest())
            result["evaluations"].append(record)
            lookup[(arm, split)] = record
            table.append("| " + " | ".join([title, split, str(actual["frames"])] +
                         [f"{metrics[key]:.3f}" for key in METRICS]) + " |")
    for split in ["dev", "test"]:
        new = lookup.get(("new_diffv2ir", split))
        if new is None:
            continue
        for old in ["rgb_original", "old_legacy", "rgb_matched"]:
            control = lookup.get((old, split))
            if control is not None:
                result["deltas"][split + "_new_minus_" + old] = {
                    key: new["metrics"][key] - control["metrics"][key] for key in METRICS}
    if result["missing"]:
        result["status"] = "INCOMPLETE"
    for name, value in [(args.output_json, json.dumps(result, indent=2, ensure_ascii=False)+"\n"),
                        (args.output_markdown, "\n".join(table)+"\n")]:
        dest = Path(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(value)
    print(json.dumps(dict(status=result["status"], completed=len(result["evaluations"]),
                         missing=len(result["missing"]), deltas=result["deltas"]), indent=2))
    if args.require_complete and result["missing"]:
        raise RuntimeError("Required comparison arms are incomplete")


if __name__ == "__main__":
    main()
