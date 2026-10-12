#!/usr/bin/env python3
"""Render train/dev offline event evidence with original RGB, GT and outputs."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from cfp.provenance import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--baseline-label', default='Original ByteTrack')
    parser.add_argument('--control-label', default='Identity-blind geometry control')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    comparison = json.loads(args.comparison.read_text())
    if comparison['status'] != 'COMPLETE' or 'no test labels' not in comparison['scope']:
        raise ValueError('Use a completed train/dev event comparison')
    annotation = json.loads((args.data_root / 'annotations/train.json').read_text())
    names = {video['id']: video['file_name'] for video in annotation['videos']}
    images = {(names[image['video_id']], int(image['frame_id'])): image for image in annotation['images']}
    args.output.mkdir(parents=True)
    reports = []
    for sequence, diagnostic in comparison['videos'].items():
        gt = np.loadtxt(args.data_root / 'train' / sequence / 'gt/gt.txt', delimiter=',', ndmin=2)
        predictions = [np.loadtxt(root / (sequence + '.txt'), delimiter=',', ndmin=2) for root in (args.baseline, args.control)]
        for event in diagnostic['baseline_only_switch_events']:
            t, identity = event['frame'], event['gt_id']
            nearby_ids = {entry['gt_id'] for entry in diagnostic['baseline_only_switch_events'] if entry['frame'] == t}
            frames = [max(1, t - 25), t, min(max(frame for seq, frame in images if seq == sequence), t + 25)]
            target = gt[(gt[:, 1] == identity) & np.isin(gt[:, 0], frames), 2:6]
            x1, y1 = target[:, :2].min(axis=0) - 150
            x2, y2 = (target[:, :2] + target[:, 2:]).max(axis=0) + 150
            first = cv2.imread(str(args.data_root / 'train' / images[(sequence, frames[0])]['vi_file_name']))
            if first is None:
                raise ValueError('Missing original RGB frame')
            x1, y1 = int(max(0, x1)), int(max(0, y1))
            x2, y2 = int(min(first.shape[1], x2)), int(min(first.shape[0], y2))
            figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
            image_hashes = {}
            for col, frame in enumerate(frames):
                path = args.data_root / 'train' / images[(sequence, frame)]['vi_file_name']
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(path)
                image_hashes[str(frame)] = sha256(path)
                for row, prediction in enumerate(predictions):
                    axis = axes[row, col]
                    axis.imshow(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB), extent=(x1, x2, y2, y1))
                    relevant_gt = gt[(gt[:, 0] == frame) & np.isin(gt[:, 1], list(nearby_ids)), 2:6]
                    for records, color, prefix in [(gt, 'lime', 'G'), (prediction, 'deepskyblue', 'P')]:
                        for entry in records[records[:, 0] == frame]:
                            x, y, w, h = entry[2:6]
                            if prefix == 'G' and int(entry[1]) not in nearby_ids:
                                continue
                            if prefix == 'P':
                                intersections = np.maximum(0, np.minimum(relevant_gt[:, :2] + relevant_gt[:, 2:], [x + w, y + h]) - np.maximum(relevant_gt[:, :2], [x, y])).prod(axis=1)
                                unions = relevant_gt[:, 2:].prod(axis=1) + w * h - intersections
                                if len(intersections) == 0 or np.max(intersections / np.maximum(unions, 1e-8)) < .15:
                                    continue
                            if x + w < x1 or x > x2 or y + h < y1 or y > y2:
                                continue
                            if prefix == 'G' and (entry[6] == 0 or (len(entry) > 7 and entry[7] != 1)):
                                continue
                            box_color = 'gold' if prefix == 'G' and int(entry[1]) != identity else color
                            axis.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=box_color, linewidth=1.8))
                            tx, ty = np.clip(x, x1 + 2, x2 - 40), np.clip(y + (0 if prefix == 'G' else h), y1 + 12, y2 - 5)
                            axis.text(tx, ty, f'{prefix}{int(entry[1])}', color=box_color,
                                      fontsize=8, bbox=dict(facecolor='black', alpha=.65, edgecolor='none', pad=1))
                    axis.set_xlim(x1, x2)
                    axis.set_ylim(y2, y1)
                    axis.set_title(f'{args.baseline_label if row == 0 else args.control_label}, frame {frame}', fontsize=10)
                    axis.axis('off')
            figure.suptitle(f'{sequence}: official CLEAR switch at frame {t}, GT {identity}\nGreen = target GT; yellow = competing GT; blue = overlapping predictions; offline dev diagnosis', fontsize=11)
            output = args.output / f'{sequence}_frame{t}_gt{identity}.png'
            figure.savefig(output, dpi=160)
            plt.close(figure)
            reports.append(dict(sequence=sequence, event=event, frames=frames, figure=str(output),
                                figure_sha256=sha256(output), source_image_sha256=image_hashes,
                                baseline_sha256=sha256(args.baseline / (sequence + '.txt')),
                                control_sha256=sha256(args.control / (sequence + '.txt'))))
    write_json(args.output / 'manifest.json', dict(status='COMPLETE', source_sha256=sha256(__file__),
               comparison_sha256=sha256(args.comparison), baseline_label=args.baseline_label,control_label=args.control_label,examples=reports,
               scope='Offline train/dev qualitative diagnostics; occurrence difference is not automatically identity recovery or a method gain'))


if __name__ == '__main__':
    main()
