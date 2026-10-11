#!/usr/bin/env python3
"""Replay fixed caches with the untouched historical BYTETracker."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from cfp.provenance import sha256, write_json
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.basetrack import BaseTrack


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--detections', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--partition', choices=['train', 'dev', 'historical_test'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for seq in protocol[args.partition]:
        cache_path = args.detections / (seq + '.npz')
        cache = np.load(cache_path)
        frames, offsets, boxes, scores = [cache[k] for k in ('frame_ids', 'offsets', 'boxes', 'scores')]
        if not np.array_equal(frames, np.arange(1, len(frames) + 1)) or len(offsets) != len(frames) + 1:
            raise ValueError('Invalid frame/offset indexing')
        if offsets[0] != 0 or offsets[-1] != len(boxes) or len(boxes) != len(scores) or np.any(np.diff(offsets) < 0):
            raise ValueError('Invalid cache size/offsets')
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError('Nonfinite cache')
        # Original evaluator keeps a globally increasing ID counter across videos.
        # Reset only at process entry for parity; sequence identity namespace is separate.
        tracker = BYTETracker(SimpleNamespace(track_thresh=.58, track_buffer=45, match_thresh=.97, mot20=False), frame_rate=25)
        times, rows = [], []
        for index, frame in enumerate(frames):
            lo, hi = offsets[index:index + 2]
            b, s = boxes[lo:hi].copy(), scores[lo:hi].copy()
            dets = np.concatenate((b, s[:, None]), axis=1)
            started = time.perf_counter()
            targets = tracker.update(dets, (1, 1), (1, 1))
            times.append(time.perf_counter() - started)
            for target in targets:
                x, y, w, h = target.tlwh
                if w * h <= 100:
                    continue
                rows.append(f'{frame},{target.track_id},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{float(target.score):.6f},-1,-1,-1\n')
        out = args.output / (seq + '.txt')
        out.write_text(''.join(rows))
        reports[seq] = {'cache_sha256': sha256(cache_path), 'frames': len(frames),
                        'detections': len(boxes), 'prediction_rows': len(rows),
                        'prediction_sha256': sha256(out), 'association_seconds': sum(times),
                        'latency_p50_ms': float(np.quantile(times, .5) * 1000),
                        'latency_p95_ms': float(np.quantile(times, .95) * 1000)}
        print(seq, reports[seq], flush=True)
    write_json(args.output / 'replay.json', {'baseline': 'original unchanged BYTETracker',
                                            'protocol_sha256': sha256(args.protocol), 'videos': reports})


if __name__ == '__main__':
    # Fresh interpreter starts BaseTrack._count at zero, as original inference does.
    BaseTrack._count = 0
    main()
