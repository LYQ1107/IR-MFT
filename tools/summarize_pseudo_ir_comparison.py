#!/usr/bin/env python3
"""Validate matched coverage and summarize completed pseudo-IR comparisons."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ARMS = {
    "rgb_original": "Original RGB ByteTrack",
    "old_legacy": "V3 + old legacy IR",
    "new_diffv2ir": "V3 + new DiffV2IR",
    "rgb_matched": "RGB matched fine-tune + ByteTrack",
}
METRICS = ["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW"]
TRAINING_ARMS = {
    "old_legacy": ("train_old_legacy_s0", "legacy", 3),
    "new_diffv2ir": ("train_new_diffv2ir_s0", "diffv2ir", 3),
    "rgb_matched": ("train_rgb_matched_s0", "rgb", 0),
}
PERCENT_FIELDS = {
    "HOTA": "HOTA___AUC", "DetA": "DetA___AUC", "AssA": "AssA___AUC",
    "DetRe": "DetRe___AUC", "DetPr": "DetPr___AUC", "LocA": "LocA___AUC",
    "IDF1": "IDF1", "IDR": "IDR", "IDP": "IDP", "MOTA": "MOTA",
    "CLR_Re": "CLR_Re", "CLR_Pr": "CLR_Pr",
}
COUNT_FIELDS = ["IDSW", "CLR_TP", "CLR_FN", "CLR_FP", "Frag", "IDTP", "IDFN",
                "IDFP", "Dets", "GT_Dets", "IDs", "GT_IDs", "CLR_Frames"]


def detailed_metrics(path, sequences, headline):
    """Use TrackEval's own sequence rows, without averaging sequence scores."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for row in rows:
        metrics = {key: 100.0 * float(row[column]) for key, column in PERCENT_FIELDS.items()}
        metrics.update({key: int(float(row[key])) for key in COUNT_FIELDS})
        if row["seq"] in sequences:
            output[row["seq"]] = metrics
        else:
            # TrackEval serializes headline percentages to five significant
            # digits. Its combined CSV row retains the full precision.
            for key in METRICS:
                if abs(metrics[key] - headline[key]) > 0.011:
                    raise ValueError(f"Combined TrackEval row disagrees with summary: {path}, {key}")
    if set(output) != set(sequences):
        raise ValueError(f"Detailed sequence coverage mismatch: {path}")
    return output


def audit_detection_coverage(folder, per_sequence):
    """Prove each full sequence was inferred once, in its original order."""
    output = {}
    for seq, metrics in per_sequence.items():
        path = folder / 'detections' / (seq + '.npz')
        with np.load(path, allow_pickle=False) as cache:
            frames, offsets = cache['frame_ids'], cache['offsets']
            boxes, scores = cache['boxes'], cache['scores']
        expected = metrics['CLR_Frames']
        if (frames.ndim != 1 or not np.issubdtype(frames.dtype, np.integer)
                or not np.array_equal(frames, np.arange(1, expected + 1))):
            raise ValueError(f'Missing, repeated or reordered sequence frames: {path}')
        if (offsets.shape != (expected + 1,) or offsets[0] != 0
                or np.any(np.diff(offsets) < 0) or offsets[-1] != len(boxes)
                or scores.shape != (len(boxes),) or boxes.shape != (len(boxes), 4)):
            raise ValueError(f'Invalid detection-to-frame offsets: {path}')
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError(f'Nonfinite saved detections: {path}')
        output[seq] = dict(frames=expected, detections=len(boxes),
                           ordered_contiguous_frame_ids=True,
                           frame_ids_sha256=hashlib.sha256(frames.astype('<i8').tobytes()).hexdigest(),
                           cache_path=str(path))
    return output


def audit_training(root, protocol):
    records, missing = {}, []
    original = root / "eval_rgb_original_dev/metrics.json"
    original_sha = json.loads(original.read_text())["protocol"]["source_sha256"] if original.exists() else None
    for arm, (directory, variant, calibration) in TRAINING_ARMS.items():
        folder = root / directory
        paths = [folder / name for name in ["train_manifest.json", "train_history.json", "completion.json"]]
        if any(not path.exists() for path in paths):
            missing.extend(str(path) for path in paths if not path.exists())
            continue
        manifest, history, complete = [json.loads(path.read_text()) for path in paths]
        expected = dict(variant=variant, seed=protocol["training"]["seed"],
                        train_size=protocol["training"]["size_hw"],
                        batch_size=protocol["training"]["batch_size_per_worker"],
                        accumulate=protocol["training"]["gradient_accumulation"],
                        calibration_epochs=calibration, finetune_epochs=3, train_frames=27251,
                        lr=protocol["training"]["V3_reference_lr_at_batch64"],
                        calibration_lr=protocol["training"]["calibration_reference_lr_at_batch64"],
                        max_frames=0, max_steps=0, no_checkpoint=False)
        for key, value in expected.items():
            if manifest[key] != value:
                raise ValueError(f"Training protocol mismatch: {arm}, {key}")
        if set(manifest["train_sequences"]) != set(protocol["train_sequences"]):
            raise ValueError(f"Training sequence mismatch: {arm}")
        if original_sha and manifest["source_sha256"] != original_sha:
            raise ValueError(f"Training initialization mismatch: {arm}")
        if str(manifest["physical_gpu"]) not in {"2", "3"}:
            raise ValueError(f"Unapproved training GPU: {arm}")
        if complete["status"] != "COMPLETE" or complete["epochs"] != calibration + 3:
            raise ValueError(f"Training incomplete: {arm}")
        if len(history) != complete["epochs"] or any(row["frames"] != 27251 for row in history):
            raise ValueError(f"Training epoch/frame coverage mismatch: {arm}")
        if [row["epoch"] for row in history] != list(range(1, calibration + 4)):
            raise ValueError(f"Training epoch sequence mismatch: {arm}")
        for index, row in enumerate(history):
            calibrating = index < calibration
            phase_index = index if calibrating else index - calibration
            reference = manifest["calibration_lr"] if calibrating else manifest["lr"]
            expected_lr = reference * manifest["batch_size"] * manifest["accumulate"] / 64.0 * (
                0.2 + 0.8 * (1 + math.cos(math.pi * phase_index / 3)) / 2)
            if not math.isclose(row["lr"], expected_lr, rel_tol=1e-10):
                raise ValueError(f"Training effective LR mismatch: {arm}, epoch {index + 1}")
            expected_phase = "calibration" if calibrating else "V3_finetune"
            if row["phase"] != expected_phase:
                raise ValueError(f"Training phase mismatch: {arm}, epoch {index + 1}")
        records[arm] = dict(manifest=manifest, history=history, completion=complete)
    return records, missing


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--result-root", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-markdown", required=True)
    p.add_argument("--require-complete", action="store_true")
    p.add_argument("--require-historical-reference", action="store_true")
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    protocol = json.loads((repo / "configs/pseudo_ir_comparison_20261120.json").read_text())
    root = Path(args.result_root)
    training, missing = audit_training(root, protocol)
    result = dict(status="COMPLETE", protocol=protocol, training=training,
                  evaluations=[], secondary_references=[], deltas={}, per_sequence_deltas={}, missing=missing)
    table = ["| Method | Split | Frames | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    lookup = {}
    arms = dict(ARMS, frozen_v3="Historical frozen V3 (different training budget)")
    for arm, title in arms.items():
        for split in ["dev", "test"]:
            secondary = arm == "frozen_v3"
            path = root / (("reference_" if secondary else "eval_") + arm + "_" + split) / "metrics.json"
            if not path.is_file():
                if not secondary or args.require_historical_reference:
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
            expected_variant = "legacy" if secondary else ("rgb" if arm == "rgb_original" else TRAINING_ARMS[arm][1])
            if actual["variant"] != expected_variant or actual["max_frames"] or actual["skip_evaluation"]:
                raise ValueError(f"Evaluation arm mismatch: {path}")
            if actual["input_size"] != protocol["evaluation"]["size_hw"]:
                raise ValueError(f"Evaluation image size mismatch: {path}")
            if str(actual["physical_gpu"]) not in {"2", "3"}:
                raise ValueError(f"Unapproved evaluation GPU: {path}")
            if secondary:
                reference = json.loads((root / "historical_v3_checkpoint_manifest.json").read_text())
                if actual["source_sha256"] != reference["sha256"]:
                    raise ValueError(f"Historical reference checkpoint changed: {path}")
                for key in ["ir_means", "ir_std"]:
                    if actual[key] != reference[key]:
                        raise ValueError(f"Historical reference normalization changed: {path}, {key}")
            if arm == "rgb_original" and split == "test" and ("rgb_original", "dev") in lookup:
                if actual["source_sha256"] != lookup[("rgb_original", "dev")]["protocol"]["source_sha256"]:
                    raise ValueError(f"Original RGB baseline checkpoint changed between splits: {path}")
            if arm in training:
                train = training[arm]
                if actual["source_sha256"] != train["completion"]["sha256"]:
                    raise ValueError(f"Evaluated checkpoint is not the final trained checkpoint: {path}")
                for key in ["ir_means", "ir_std"]:
                    if actual[key] != train["manifest"][key]:
                        raise ValueError(f"Training/evaluation normalization mismatch: {path}, {key}")
            metrics = {key: d["trackeval"][key] for key in METRICS}
            per_sequence = detailed_metrics(path.parent / "trackeval_output/pedestrian_detailed.csv",
                                             actual["sequences"], metrics)
            coverage = audit_detection_coverage(path.parent, per_sequence)
            if sum(seq['frames'] for seq in coverage.values()) != actual['frames']:
                raise ValueError(f'Per-frame and aggregate coverage disagree: {path}')
            record = dict(arm=arm, method=title, split=split, frames=actual["frames"],
                          metrics=metrics, trackeval_all_metrics=d["trackeval"],
                          per_sequence=per_sequence, detection_coverage=coverage,
                          runtime=d["runtime"], protocol=actual,
                          metrics_path=str(path), metrics_sha256=hashlib.sha256(raw).hexdigest())
            result["secondary_references" if secondary else "evaluations"].append(record)
            if not secondary:
                lookup[(arm, split)] = record
            if secondary and split == "dev":
                table += ["", "Secondary reference; its training history and budget differ from the matched arms:", "",
                          "| Method | Split | Frames | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |",
                          "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
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
                result["per_sequence_deltas"][split + "_new_minus_" + old] = {
                    seq: {key: new["per_sequence"][seq][key] - control["per_sequence"][seq][key]
                          for key in METRICS} for seq in protocol[split + "_sequences"]}
    table += ["", "Individual sequences (TrackEval percentages; IDSW is a count):", "",
              "| Method | Split | Sequence | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["evaluations"] + result["secondary_references"]:
        for seq, metrics in row["per_sequence"].items():
            table.append("| " + " | ".join([row["method"], row["split"], seq] +
                         [f"{metrics[key]:.3f}" for key in METRICS]) + " |")
    if result["missing"]:
        result["status"] = "INCOMPLETE"
    for name, value in [(args.output_json, json.dumps(result, indent=2, ensure_ascii=False)+"\n"),
                        (args.output_markdown, "\n".join(table)+"\n")]:
        dest = Path(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(value)
    print(json.dumps(dict(status=result["status"], completed=len(result["evaluations"]),
                         secondary_references=len(result["secondary_references"]),
                         missing=len(result["missing"]), deltas=result["deltas"]), indent=2))
    if args.require_complete and result["missing"]:
        raise RuntimeError("Required comparison arms are incomplete")


if __name__ == "__main__":
    main()
