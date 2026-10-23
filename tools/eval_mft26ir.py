#!/usr/bin/env python3
"""Run one fixed MFT26-IR test protocol and save MOT + TrackEval metrics."""

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import OrderedDict
from types import SimpleNamespace

# Running this file directly puts ``tools/`` ahead of the repository root.
# Pin the local YOLOX package before optional external reference packages
# (some of which ship a package with the same top-level name).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

import motmetrics as mm
import numpy as np
import pandas as pd
import torch

# The pinned local motmetrics fork predates NumPy 1.24 aliases.
if "float" not in np.__dict__:
    np.float = float
if "int" not in np.__dict__:
    np.int = int
if "bool" not in np.__dict__:
    np.bool = bool

from yolox.exp import get_exp
from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils import postprocess


DEFAULT_ROOT = "/data3/liuyeqiang/datasets/mft26-ir"
DEFAULT_GT_ROOT = "/data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt"
DEFAULT_TRACKEVAL_ROOT = "/data3/liuyeqiang/IR-MFT_assets/vendor/VUP-PFTrack"


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_file", default="exps/yolox_mft26ir_rgbt_v1.py")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", default="results/v1_avg")
    parser.add_argument(
        "--split",
        default="test",
        choices=("dev", "test"),
        help="evaluation split; dev uses train.json and train/*/gt, test uses test.json",
    )
    parser.add_argument(
        "--fusion-mode",
        default=None,
        choices=(
            "avg",
            "gated_residual",
            "dynamic_quality",
            "spatial_complementarity",
            "quality_spatial",
            "quality_spatial_redundancy",
            "temporal_complementarity",
        ),
        help="override the fusion mode declared by the experiment",
    )
    parser.add_argument(
        "--auxiliary-mode",
        default=None,
        choices=("legacy", "causal", "causal_cache", "zero", "shuffle", "repeat_rgb", "grayscale_rgb"),
        help="auxiliary input intervention; causal mode forces sequential loading",
    )
    parser.add_argument(
        "--temporal-state",
        default="auto",
        choices=("auto", "stateful", "disabled", "reset_each_frame"),
        help=(
            "temporal fusion state policy: auto/stateful keeps causal state "
            "within each sequence, disabled removes the state contribution, "
            "and reset_each_frame removes history while keeping the module"
        ),
    )
    parser.add_argument(
        "--tracker-type",
        default="bytetrack",
        choices=("bytetrack",),
        help="tracker implementation; the baseline tracker remains BYTETracker",
    )
    parser.add_argument("--collect-fusion-stats", action="store_true")
    parser.add_argument("--profile-runtime", action="store_true")
    parser.add_argument("--manifest-output", default=None)
    parser.add_argument(
        "--sequences",
        default=None,
        help="comma-separated test sequences to evaluate in this worker",
    )
    parser.add_argument("--data-root", default=os.environ.get("MFT26IR_ROOT", DEFAULT_ROOT))
    parser.add_argument(
        "--gt-root",
        default=os.environ.get("MFT26IR_GT_ROOT", DEFAULT_GT_ROOT),
        help="protocol-derived MOT GT root; duplicate-ID corrections are applied there",
    )
    parser.add_argument("--trackeval-root", default=DEFAULT_TRACKEVAL_ROOT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="CPU workers used to decode and preprocess paired frames",
    )
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--nms", type=float, default=0.7)
    parser.add_argument("--track-thresh", type=float, default=0.6)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.9)
    parser.add_argument("--min-box-area", type=float, default=100.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-trackeval", action="store_true")
    parser.add_argument(
        "--detections-output",
        default=None,
        help="optional directory for cached post-NMS detections used in tracker sweeps",
    )
    return parser


def write_sequence_results(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for frame_id, tlwh, track_id, score in rows:
            x, y, width, height = [float(value) for value in tlwh]
            handle.write(
                f"{frame_id},{track_id},{x:.3f},{y:.3f},{width:.3f},{height:.3f},"
                f"{float(score):.6f},-1,-1,-1\n"
            )


def load_mot_file(path):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        index = pd.MultiIndex.from_arrays([[], []], names=["FrameId", "Id"])
        return pd.DataFrame(
            columns=["X", "Y", "Width", "Height"], index=index
        )
    return mm.io.loadtxt(path, fmt="mot15-2D", min_confidence=-1)


def compute_motmetrics(result_dir, gt_root, sequences, data_split="test"):
    gts = OrderedDict()
    tracks = OrderedDict()
    for sequence in sequences:
        gt_path = os.path.join(gt_root, data_split, sequence, "gt", "gt.txt")
        result_path = os.path.join(result_dir, f"{sequence}.txt")
        gts[sequence] = mm.io.loadtxt(gt_path, fmt="mot15-2D", min_confidence=1)
        tracks[sequence] = load_mot_file(result_path)

    accumulators = []
    names = []
    for sequence in sequences:
        accumulators.append(
            mm.utils.compare_to_groundtruth(
                gts[sequence], tracks[sequence], "iou", distth=0.5
            )
        )
        names.append(sequence)

    metrics = [
        "idf1",
        "idp",
        "idr",
        "recall",
        "precision",
        "num_false_positives",
        "num_misses",
        "num_switches",
        "mota",
        "motp",
        "num_objects",
    ]
    summary = mm.metrics.create().compute_many(
        accumulators, names=names, metrics=metrics, generate_overall=True
    )

    serializable = {}
    for name, row in summary.iterrows():
        serializable[name] = {}
        for metric in metrics:
            value = row[metric]
            serializable[name][metric] = None if pd_is_nan(value) else float(value)
    return serializable


def pd_is_nan(value):
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def keep_batch(batch):
    """Keep variable-length MOT targets as a list in the evaluation loader."""
    return batch


def run_trackeval(result_dir, gt_root, output_dir, sequences, trackeval_root, data_split="test"):
    """Run the repository's already-used TrackEval fork for HOTA/DetA/AssA."""
    # The historical fork is customized for this MOT-RGBT layout and still
    # uses NumPy aliases removed in NumPy 1.24. Keep the compatibility local to
    # this process rather than modifying the shared evaluation toolkit.
    if "float" not in np.__dict__:
        np.float = float
    if "int" not in np.__dict__:
        np.int = int

    sys.path.insert(0, trackeval_root)
    from trackeval.eval import Evaluator
    from trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox
    from trackeval.metrics.clear import CLEAR
    from trackeval.metrics.hota import HOTA
    from trackeval.metrics.identity import Identity

    trackeval_dir = os.path.join(output_dir, "trackeval_format")
    os.makedirs(trackeval_dir, exist_ok=True)
    for sequence in sequences:
        source = os.path.join(result_dir, f"{sequence}.txt")
        if not os.path.isfile(source):
            raise FileNotFoundError(source)
        shutil.copyfile(source, os.path.join(trackeval_dir, f"{sequence}.txt"))

    trackeval_output = os.path.join(output_dir, "trackeval_output")
    os.makedirs(trackeval_output, exist_ok=True)
    config = {
        "TRACKERS_FOLDER": trackeval_dir,
        "TRACKERS_TO_EVAL": ["QRTM_V1"],
        "GT_FOLDER": os.path.join(gt_root, data_split),
        "OUTPUT_FOLDER": trackeval_output,
        "BENCHMARK": "MFT26IR",
        "CLASSES_TO_EVAL": ["pedestrian"],
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
        "SEQ_INFO": list(sequences),
        "PRINT_RESULTS": False,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "DO_PREPROC": False,
        "use_super_categories": False,
        "USE_PARALLEL": False,
    }
    evaluator = Evaluator(
        {
            "PRINT_RESULTS": False,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "DISPLAY_LESS_PROGRESS": True,
        }
    )
    dataset = MotChallenge2DBox(config)
    evaluator.evaluate(dataset, [HOTA(), CLEAR(), Identity()])

    summary_path = os.path.join(trackeval_output, "pedestrian_summary.txt")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(summary_path)
    with open(summary_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    headers = lines[0].split()
    values = lines[1].split()
    parsed = {}
    for key, value in zip(headers, values):
        try:
            parsed[key] = float(value)
        except ValueError:
            parsed[key] = value
    return parsed


def main():
    args = make_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    exp = get_exp(args.exp_file, None)
    if args.fusion_mode is not None:
        exp.fusion_mode = args.fusion_mode
    if args.auxiliary_mode is not None:
        exp.auxiliary_mode = args.auxiliary_mode
    exp.data_dir = args.data_root
    exp.test_conf = args.conf
    exp.nmsthre = args.nms
    model = exp.get_model().to(device)
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    if args.fp16:
        model.half()
    model.eval()
    if args.temporal_state == "disabled":
        configure_state = getattr(model, "set_fusion_stateful", None)
        if configure_state is not None:
            configure_state(False)
    elif args.temporal_state in {"stateful", "reset_each_frame"}:
        configure_state = getattr(model, "set_fusion_stateful", None)
        if configure_state is not None:
            configure_state(True, detach_state=True)

    rgb_means = getattr(exp, "rgb_means", (0.485, 0.456, 0.406))
    rgb_std = getattr(exp, "rgb_std", (0.229, 0.224, 0.225))
    ir_means = getattr(exp, "ir_means", rgb_means)
    ir_std = getattr(exp, "ir_std", rgb_std)
    eval_split = args.split
    annotation = exp.val_ann if eval_split == "test" else exp.train_ann
    data_name = "test" if eval_split == "test" else "train"
    dataset = exp._dataset(
        eval_split,
        annotation,
        __import__("yolox.data", fromlist=["RGBTValTransform"]).RGBTValTransform(
            rgb_means=rgb_means,
            std=rgb_std,
            ir_means=ir_means,
            ir_std=ir_std,
        ),
        data_name=data_name,
    )
    # ``train.json`` contains both Train and Dev videos.  The dataset has
    # already applied the protocol sequence filter, so derive the available
    # sequence list from its filtered annotations rather than the raw COCO
    # video table.
    all_sequences = []
    seen_sequences = set()
    for annotation_row in dataset.annotations:
        sequence = annotation_row[2].split("/", 1)[0]
        if sequence not in seen_sequences:
            all_sequences.append(sequence)
            seen_sequences.add(sequence)
    if not all_sequences:
        raise RuntimeError("No test sequences were found in the annotation JSON")

    if args.sequences:
        requested_sequences = [
            sequence.strip() for sequence in args.sequences.split(",") if sequence.strip()
        ]
        unknown_sequences = sorted(set(requested_sequences) - set(all_sequences))
        if unknown_sequences:
            raise ValueError(f"Unknown test sequences: {unknown_sequences}")
        sequences = [
            sequence for sequence in all_sequences if sequence in requested_sequences
        ]
    else:
        sequences = all_sequences
    if not sequences:
        raise RuntimeError("No test sequences selected")

    tracker_args = SimpleNamespace(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        mot20=False,
    )
    result_dir = os.path.join(args.output, "track_results")
    os.makedirs(result_dir, exist_ok=True)
    rows_by_sequence = {sequence: [] for sequence in sequences}
    sequence_by_index = [
        dataset.annotations[index][2].split("/")[0]
        for index in range(len(dataset))
    ]
    selected_indices = [
        index
        for index, sequence in enumerate(sequence_by_index)
        if sequence in rows_by_sequence
    ]
    tracker = None
    current_sequence = None
    total_frames = 0
    detections_by_sequence = {sequence: [] for sequence in sequences}

    eval_dataset = torch.utils.data.Subset(dataset, selected_indices)
    eval_workers = 0 if exp.auxiliary_mode == "causal" else args.workers
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, eval_workers),
        pin_memory=True,
        collate_fn=keep_batch,
        persistent_workers=eval_workers > 0,
    )

    fusion_level_stats = []
    runtime_start = time.perf_counter()
    model_sequence = None
    with torch.no_grad():
        for batch_items in eval_loader:
            infos = [item[2] for item in batch_items]
            batch_sequence = infos[0][4].split("/")[0]
            if batch_sequence != model_sequence:
                reset = getattr(model, "reset_fusion_state", None)
                if reset is not None:
                    reset()
                model_sequence = batch_sequence
            if args.temporal_state == "reset_each_frame":
                reset = getattr(model, "reset_fusion_state", None)
                if reset is not None:
                    reset()
            input_tensor = torch.from_numpy(
                np.stack([item[0] for item in batch_items], axis=0)
            ).to(device, non_blocking=True)
            if args.fp16:
                input_tensor = input_tensor.half()
            model_output = model(input_tensor, return_aux=args.collect_fusion_stats)
            if args.collect_fusion_stats:
                raw_output, fusion_aux = model_output
            else:
                raw_output = model_output
                fusion_aux = None
            outputs = postprocess(raw_output, 1, args.conf, args.nms)
            for info, output in zip(infos, outputs):
                sequence = info[4].split("/")[0]
                if sequence != current_sequence:
                    if sequence not in rows_by_sequence:
                        raise RuntimeError(f"Unexpected sequence in data: {sequence}")
                    tracker = BYTETracker(tracker_args, frame_rate=25)
                    current_sequence = sequence

                if fusion_aux is not None:
                    levels = fusion_aux.get("levels", [])
                    if levels and not fusion_level_stats:
                        fusion_level_stats = [
                            {"effective_weight": [], "redundancy": [], "gate_range": []}
                            for _ in levels
                        ]
                    for level_index, level_info in enumerate(levels):
                        if level_index >= len(fusion_level_stats):
                            continue
                        for key in ("effective_weight", "redundancy", "gate_range"):
                            value = level_info.get(key)
                            if torch.is_tensor(value):
                                fusion_level_stats[level_index][key].append(
                                    float(value.detach().float().mean().cpu())
                                )

                if args.detections_output is not None:
                    if output is None or len(output) == 0:
                        boxes = np.empty((0, 4), dtype=np.float32)
                        scores = np.empty((0,), dtype=np.float32)
                    else:
                        output_np = output.detach().float().cpu().numpy()
                        scale = min(
                            exp.test_size[0] / float(info[0]),
                            exp.test_size[1] / float(info[1]),
                        )
                        boxes = (output_np[:, :4] / scale).astype(np.float32)
                        scores = (output_np[:, 4] * output_np[:, 5]).astype(np.float32)
                    detections_by_sequence[sequence].append(
                        (int(info[2]), boxes, scores)
                    )

                if output is None:
                    output = torch.empty(
                        (0, 7), dtype=input_tensor.dtype, device=device
                    )
                online_targets = tracker.update(
                    output, (info[0], info[1]), exp.test_size
                )
                for target in online_targets:
                    tlwh = target.tlwh
                    if tlwh[2] * tlwh[3] <= args.min_box_area:
                        continue
                    rows_by_sequence[sequence].append(
                        (info[2], tlwh, target.track_id, target.score)
                    )
                total_frames += 1

    for sequence in sequences:
        write_sequence_results(
            os.path.join(result_dir, f"{sequence}.txt"), rows_by_sequence[sequence]
        )

    if args.detections_output is not None:
        os.makedirs(args.detections_output, exist_ok=True)
        for sequence in sequences:
            entries = detections_by_sequence[sequence]
            frame_ids = np.asarray([entry[0] for entry in entries], dtype=np.int32)
            offsets = np.zeros((len(entries) + 1,), dtype=np.int64)
            if entries:
                counts = np.asarray([len(entry[2]) for entry in entries], dtype=np.int64)
                offsets[1:] = np.cumsum(counts)
                boxes = np.concatenate([entry[1] for entry in entries], axis=0)
                scores = np.concatenate([entry[2] for entry in entries], axis=0)
            else:
                boxes = np.empty((0, 4), dtype=np.float32)
                scores = np.empty((0,), dtype=np.float32)
            np.savez_compressed(
                os.path.join(args.detections_output, f"{sequence}.npz"),
                frame_ids=frame_ids,
                offsets=offsets,
                boxes=boxes,
                scores=scores,
            )

    metrics = {
        "motmetrics": compute_motmetrics(
            args.output + "/track_results", args.gt_root, sequences, data_split=data_name
        )
    }
    if not args.skip_trackeval:
        metrics["trackeval"] = run_trackeval(
            result_dir,
            args.gt_root,
            args.output,
            sequences,
            args.trackeval_root,
            data_split=data_name,
        )
    metrics["protocol"] = {
        "sequences": sequences,
        "frames": total_frames,
        "checkpoint": os.path.abspath(args.ckpt),
        "gt_root": os.path.abspath(args.gt_root),
        "conf": args.conf,
        "nms": args.nms,
        "track_thresh": args.track_thresh,
        "track_buffer": args.track_buffer,
        "match_thresh": args.match_thresh,
        "min_box_area": args.min_box_area,
        "fusion_mode": getattr(exp, "fusion_mode", None),
        "auxiliary_mode": getattr(exp, "auxiliary_mode", "legacy"),
        "temporal_state": args.temporal_state,
        "tracker_type": args.tracker_type,
        "workers": eval_workers,
        "split": eval_split,
        "data_name": data_name,
    }
    if args.profile_runtime:
        elapsed = max(time.perf_counter() - runtime_start, 1e-9)
        metrics["runtime"] = {
            "elapsed_seconds": elapsed,
            "frames_per_second": float(total_frames / elapsed),
            "frames": total_frames,
        }
    if fusion_level_stats:
        stats = []
        for level in fusion_level_stats:
            stats.append(
                {
                    key: {
                        "mean": float(np.mean(values)) if values else None,
                        "std": float(np.std(values)) if values else None,
                        "min": float(np.min(values)) if values else None,
                        "max": float(np.max(values)) if values else None,
                        "count": len(values),
                    }
                    for key, values in level.items()
                }
            )
        metrics["fusion_stats"] = {"levels": stats}
    metrics_path = os.path.join(args.output, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    if args.manifest_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.manifest_output)), exist_ok=True)
        with open(args.manifest_output, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("saved:", os.path.abspath(metrics_path))


if __name__ == "__main__":
    main()
