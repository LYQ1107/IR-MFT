#!/usr/bin/env python3
"""Match caches geometrically to distinguish inference drift from ordering."""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from cfp.provenance import sha256, write_json


def compare(old_path, new_path):
    old, new = np.load(old_path), np.load(new_path)
    if not np.array_equal(old['frame_ids'], new['frame_ids']):
        raise ValueError('Frame indexing differs')
    matches, exact_box, exact_score, unmatched_old, unmatched_new = 0, 0, 0, 0, 0
    box_errors, score_errors, changed_frames = [], [], []
    for index, frame in enumerate(old['frame_ids']):
        a0, a1 = old['offsets'][index:index + 2]
        b0, b1 = new['offsets'][index:index + 2]
        a, b = old['boxes'][a0:a1], new['boxes'][b0:b1]
        sa, sb = old['scores'][a0:a1], new['scores'][b0:b1]
        if len(a) == len(b) and np.array_equal(a, b) and np.array_equal(sa, sb):
            matches += len(a)
            exact_box += len(a)
            exact_score += len(a)
            box_errors.extend([0.] * len(a))
            score_errors.extend([0.] * len(a))
            continue
        changed_frames.append(int(frame))
        wh = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]))
        inter = wh.prod(axis=-1)
        area_a, area_b = np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=-1), np.maximum(0, b[:, 2:] - b[:, :2]).prod(axis=-1)
        iou = inter / np.maximum(area_a[:, None] + area_b - inter, 1e-8)
        rows, cols = linear_sum_assignment(1 - iou)
        good = iou[rows, cols] >= .5
        rows, cols = rows[good], cols[good]
        matches += len(rows)
        unmatched_old += len(a) - len(rows)
        unmatched_new += len(b) - len(rows)
        be = np.abs(a[rows] - b[cols]).max(axis=-1)
        se = np.abs(sa[rows] - sb[cols])
        exact_box += int(np.sum(be == 0))
        exact_score += int(np.sum(se == 0))
        box_errors.extend(be.tolist())
        score_errors.extend(se.tolist())
    return {'old_sha256': sha256(old_path), 'new_sha256': sha256(new_path),
            'frames': len(old['frame_ids']), 'old_detections': len(old['scores']), 'new_detections': len(new['scores']),
            'matched_iou_at_least_0_5': matches, 'unmatched_old': unmatched_old, 'unmatched_new': unmatched_new,
            'matched_boxes_exact': exact_box, 'matched_scores_exact': exact_score,
            'box_max_coordinate_abs_diff_px_p50_p95_max': np.quantile(box_errors, [.5, .95, 1]).tolist(),
            'score_abs_diff_p50_p95_max': np.quantile(score_errors, [.5, .95, 1]).tolist(),
            'changed_frames': changed_frames}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--new', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    records = {}
    for old in sorted(args.old.glob('*.npz')):
        new = args.new / old.name
        records[old.stem] = compare(old, new)
        print(old.stem, {k: v for k, v in records[old.stem].items() if k != 'changed_frames'}, flush=True)
    write_json(args.output, records)


if __name__ == '__main__':
    main()
