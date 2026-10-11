#!/usr/bin/env python3
"""Replay frozen proposals with exact-off or declared appearance baselines."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from cfp.association import AppearanceTracker
from cfp.provenance import config_hash, sha256, write_json
from yolox.tracker.basetrack import BaseTrack


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--detections', type=Path, required=True)
    p.add_argument('--features-root', type=Path)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--partition', choices=['train', 'dev', 'historical_test'], required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--mode', choices=['off', 'rgb', 'ir', 'fixed'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--appearance-weight', type=float)
    p.add_argument('--final-freeze', type=Path)
    args = p.parse_args()
    if args.partition == 'historical_test' and args.mode != 'off':
        if args.final_freeze is None or json.loads(args.final_freeze.read_text())['state'] != 'FROZEN':
            raise ValueError('Enabled historical test evaluation requires final frozen configuration')
    if args.output.exists():
        raise FileExistsError('Replay outputs are immutable')
    config = json.loads(args.config.read_text())
    if config['state'] != 'FROZEN':
        raise ValueError('Undeclared association baseline')
    if args.appearance_weight is not None:
        if args.appearance_weight not in config['appearance_weight_declared_candidates']:
            raise ValueError('Appearance weight outside the declared dev search')
        config['appearance_weight'] = args.appearance_weight
    if args.mode != 'off' and args.features_root is None:
        raise ValueError('Enabled appearance requires aligned detector-crop features')
    protocol = json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True)
    manifest = dict(status='RUNNING', config=config, config_sha256=config_hash(config), mode=args.mode,
                    protocol_sha256=sha256(args.protocol), partition=args.partition,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    source_sha256=sha256(__file__), tracker_sha256=sha256('cfp/association.py'),
                    original_tracker_sha256=sha256('yolox/tracker/byte_tracker.py'))
    write_json(args.output / 'manifest.json', manifest)
    reports = {}
    BaseTrack._count = 0
    for sequence in protocol[args.partition]:
        path = args.detections / (sequence + '.npz')
        cached = np.load(path)
        frames, offsets, boxes, scores = [cached[key] for key in ['frame_ids', 'offsets', 'boxes', 'scores']]
        if not np.array_equal(frames, np.arange(1, len(frames) + 1)) or len(offsets) != len(frames) + 1:
            raise ValueError('Invalid frame indexing')
        if offsets[0] != 0 or offsets[-1] != len(boxes) or len(boxes) != len(scores) or np.any(np.diff(offsets) < 0):
            raise ValueError('Invalid detection indexing')
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError('Nonfinite proposals')
        rgb, ir, feature_manifest = None, None, None
        if args.mode != 'off':
            feature_path = args.features_root / (sequence + '.npz')
            feature_manifest = json.loads(feature_path.with_suffix('.provenance.json').read_text())
            if feature_manifest['status'] != 'COMPLETE' or feature_manifest['detection_sha256'] != sha256(path):
                raise ValueError('Feature/proposal provenance mismatch')
            if feature_manifest['features_sha256'] != sha256(feature_path):
                raise ValueError('Feature cache changed')
            feature = np.load(feature_path)
            rgb = feature['rgb']
            ir = feature['ir'] if 'ir' in feature.files else None
            if len(rgb) != len(boxes) or (args.mode in ('fixed', 'ir') and ir is None):
                raise ValueError('Missing or unaligned appearance features')
            if args.mode in ('fixed', 'ir') and feature_manifest['input_regime'] != 'causal':
                raise ValueError('Constructed-modality comparisons require causal inputs')
            # Each video has its own persistent output ID namespace and state.
            BaseTrack._count = 0
        tracker = AppearanceTracker(SimpleNamespace(track_thresh=config['track_thresh'], track_buffer=config['track_buffer'],
                                    match_thresh=config['match_thresh'], mot20=False), frame_rate=config['frame_rate'], mode=args.mode,
                                    appearance_weight=config['appearance_weight'], ir_weight=config['ir_weight'], ema=config['ema'],
                                    proximity_iou=config['proximity_iou'], appearance_max_distance=config['appearance_max_distance'])
        latencies, rows = [], []
        for i, frame in enumerate(frames):
            lo, hi = offsets[i:i + 2]
            detections = np.concatenate([boxes[lo:hi].copy(), scores[lo:hi, None]], axis=1)
            start = time.perf_counter()
            targets = tracker.update(detections, (1, 1), (1, 1), None if rgb is None else rgb[lo:hi], None if ir is None else ir[lo:hi])
            latencies.append(time.perf_counter() - start)
            for target in targets:
                x, y, w, h = target.tlwh
                if w * h <= config['min_box_area']:
                    continue
                rows.append(f'{frame},{target.track_id},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{float(target.score):.6f},-1,-1,-1\n')
        result_path = args.output / (sequence + '.txt')
        result_path.write_text(''.join(rows))
        reports[sequence] = dict(frames=len(frames), detection_sha256=sha256(path), prediction_sha256=sha256(result_path),
                                 feature_manifest=feature_manifest, association_seconds=sum(latencies),
                                 association_latency_p50_p95_ms=(np.quantile(latencies, [.5, .95]) * 1000).tolist(),
                                 decision_counts=tracker.decision_counts)
        print(sequence, 'frames', len(frames), 'association seconds', round(sum(latencies), 3), flush=True)
    manifest.update(status='COMPLETE', videos=reports, latency_scope='Association only; no encoder, generator, detector or original decode')
    write_json(args.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
