#!/usr/bin/env python3
"""Sweep ByteTrack parameters against cached MFT26-IR detections."""

import argparse
import csv
import json
import multiprocessing as mp
import os
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed

import motmetrics as mm
import numpy as np
import pandas as pd
from yolox.tracker.byte_tracker import BYTETracker


DEFAULT_SEQUENCES = (
    "BT-002",
    "BT-004",
    "MSK-003",
    "PF-002",
    "SN-009",
    "SN-011",
    "SN-014",
)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--data-root", default="/data1/LWR/vranlee/CELL/PFTrack/data/mft26-ir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--track-threshes", default="0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--match-threshes", default="0.70,0.80,0.85,0.90,0.95")
    parser.add_argument("--track-buffers", default="15,30,60")
    parser.add_argument("--min-box-areas", default="50,100,150")
    parser.add_argument("--sequences", default=",".join(DEFAULT_SEQUENCES))
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel CPU workers for independent parameter configurations",
    )
    return parser


def parse_floats(value):
    return [float(item) for item in value.split(",") if item.strip()]


def parse_ints(value):
    return [int(item) for item in value.split(",") if item.strip()]


def load_detections(path, sequences):
    cached = {}
    for sequence in sequences:
        archive = np.load(os.path.join(path, f"{sequence}.npz"))
        cached[sequence] = {
            "frame_ids": archive["frame_ids"].astype(np.int32, copy=False),
            "offsets": archive["offsets"].astype(np.int64, copy=False),
            "boxes": archive["boxes"].astype(np.float32, copy=False),
            "scores": archive["scores"].astype(np.float32, copy=False),
        }
    return cached


def tracks_for_sequence(cached, track_thresh, track_buffer, match_thresh, min_box_area):
    tracker = BYTETracker(
        SimpleNamespace(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            mot20=False,
        ),
        frame_rate=25,
    )
    rows = []
    frame_ids = cached["frame_ids"]
    offsets = cached["offsets"]
    boxes = cached["boxes"]
    scores = cached["scores"]
    for index, frame_id in enumerate(frame_ids):
        start, end = int(offsets[index]), int(offsets[index + 1])
        if end > start:
            detections = np.concatenate(
                (boxes[start:end], scores[start:end, None]), axis=1
            )
        else:
            detections = np.empty((0, 5), dtype=np.float32)
        # Cached boxes are already in original-image coordinates. Passing an
        # identity scale keeps BYTETracker's normal association code intact.
        online_targets = tracker.update(detections, (1, 1), (1, 1))
        for target in online_targets:
            tlwh = target.tlwh
            if tlwh[2] * tlwh[3] <= min_box_area:
                continue
            rows.append(
                (
                    int(frame_id),
                    int(target.track_id),
                    float(tlwh[0]),
                    float(tlwh[1]),
                    float(tlwh[2]),
                    float(tlwh[3]),
                    float(target.score),
                )
            )
    return rows


def rows_to_dataframe(rows):
    columns = ["FrameId", "Id", "X", "Y", "Width", "Height", "Confidence"]
    if not rows:
        empty = pd.DataFrame(columns=columns)
        empty.index = pd.MultiIndex.from_arrays(
            [[], []], names=["FrameId", "Id"]
        )
        return empty
    dataframe = pd.DataFrame(rows, columns=columns)
    return dataframe.set_index(["FrameId", "Id"])


def compute_metrics(cached, data_root, sequences, params):
    accumulators = []
    names = []
    for sequence in sequences:
        gt_path = os.path.join(data_root, "test", sequence, "gt", "gt.txt")
        ground_truth = mm.io.loadtxt(gt_path, fmt="mot15-2D", min_confidence=1)
        rows = tracks_for_sequence(cached[sequence], **params)
        tracks = rows_to_dataframe(rows)
        accumulators.append(
            mm.utils.compare_to_groundtruth(
                ground_truth, tracks, "iou", distth=0.5
            )
        )
        names.append(sequence)

    summary = mm.metrics.create().compute_many(
        accumulators,
        names=names,
        metrics=["idf1", "mota", "num_switches", "recall", "precision"],
        generate_overall=True,
    )
    overall = summary.loc["OVERALL"]
    return {
        **params,
        "idf1": float(overall["idf1"]),
        "mota": float(overall["mota"]),
        "id_switches": int(overall["num_switches"]),
        "recall": float(overall["recall"]),
        "precision": float(overall["precision"]),
    }


def run_one_configuration(cached, data_root, sequences, params):
    return compute_metrics(cached, data_root, sequences, params)


def main():
    args = make_parser().parse_args()
    sequences = [item.strip() for item in args.sequences.split(",") if item.strip()]
    cached = load_detections(args.detections, sequences)
    track_threshes = parse_floats(args.track_threshes)
    match_threshes = parse_floats(args.match_threshes)
    track_buffers = parse_ints(args.track_buffers)
    min_box_areas = parse_floats(args.min_box_areas)

    configurations = []
    for track_thresh in track_threshes:
        for match_thresh in match_threshes:
            for track_buffer in track_buffers:
                for min_box_area in min_box_areas:
                    configurations.append(
                        {
                            "track_thresh": track_thresh,
                            "track_buffer": track_buffer,
                            "match_thresh": match_thresh,
                            "min_box_area": min_box_area,
                        }
                    )

    results = []
    total = len(configurations)
    if args.workers <= 1:
        for current, params in enumerate(configurations, start=1):
            result = run_one_configuration(cached, args.data_root, sequences, params)
            results.append(result)
            if result["idf1"] > 0.341 and result["mota"] > 0.518:
                print("TARGET", json.dumps(result, ensure_ascii=False), flush=True)
            if current % 10 == 0:
                print(f"progress {current}/{total}", flush=True)
    else:
        context = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=context
        ) as executor:
            futures = [
                executor.submit(
                    run_one_configuration,
                    cached,
                    args.data_root,
                    sequences,
                    params,
                )
                for params in configurations
            ]
            for current, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                if result["idf1"] > 0.341 and result["mota"] > 0.518:
                    print("TARGET", json.dumps(result, ensure_ascii=False), flush=True)
                if current % 10 == 0:
                    print(f"progress {current}/{total}", flush=True)

    results.sort(key=lambda item: (item["idf1"], item["mota"]), reverse=True)
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "sweep.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output, "sweep.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(results[:20], indent=2, ensure_ascii=False))
    print("saved:", os.path.abspath(args.output))


if __name__ == "__main__":
    main()
