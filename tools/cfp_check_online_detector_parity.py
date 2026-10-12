#!/usr/bin/env python3
"""Compare current-frame detector calls to frozen batch8 proposals, no GT reads."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from cfp.online import CurrentFrameFrontend
from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--detections', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--scripts-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int, default=128)
    args = p.parse_args()
    if args.output.exists() or args.frames <= 0:
        raise ValueError('Immutable nonempty parity diagnostic required')
    dm = json.loads((args.detections / 'manifest.json').read_text())
    if dm['status'] != 'COMPLETE' or dm['checkpoint_sha256'] != sha256(args.checkpoint):
        raise ValueError('Frozen detector reference changed')
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    report = dict(status='RUNNING', detection_manifest_sha256=sha256(args.detections / 'manifest.json'),
                  source_snapshots=snapshot_sources(args.output, [__file__, 'cfp/online.py']), batches={},
                  scope='Current paired pixels from proven causal generation; detector numerical/batch-shape parity only, no GT/model/threshold selection or end-to-end speed.')
    write_json(args.output / 'manifest.json', report)
    for batch in [1, 8]:
        frontend = CurrentFrameFrontend(args.checkpoint, args.scripts_root, detector_batch=batch,
                                        input_regime='causal_precomputed', preprocessing='lookup_uint8')
        videos = {}
        for sequence in ['BT-005', 'SN-013']:
            dp = args.detections / (sequence + '.npz')
            if sha256(dp) != dm['caches'][sequence]['sha256']:
                raise ValueError('Frozen detector arrays changed')
            with np.load(dp) as a:
                offsets, raw, boxes, scores = [a[k] for k in ('offsets', 'raw_post_nms', 'boxes', 'scores')]
            mismatches, count_mismatch, max_delta = [], [], 0.
            current_rows = []
            for frame in range(1, min(args.frames, len(offsets) - 1) + 1):
                rgb = cv2.imread(str(args.data_root / 'train' / sequence / 'visible' / f'{frame:06d}.jpg'))
                ir = cv2.imread(str(args.data_root / 'train' / sequence / 'infrared' / f'{frame:06d}.jpg'))
                values = frontend.process(rgb, provided_ir=ir)
                lo, hi = offsets[frame - 1:frame + 1]
                actual = values['raw_post_nms']
                if not np.array_equal(actual, raw[lo:hi]):
                    mismatches.append(frame)
                    if actual.shape != raw[lo:hi].shape:
                        count_mismatch.append(frame)
                    elif len(actual):
                        max_delta = max(max_delta, float(np.max(np.abs(actual - raw[lo:hi]))))
                current_rows.append(dict(frame=frame, raw_equal=bool(np.array_equal(actual, raw[lo:hi])),
                                         boxes_equal=bool(np.array_equal(values['boxes'], boxes[lo:hi])),
                                         scores_equal=bool(np.array_equal(values['scores'], scores[lo:hi]))))
            videos[sequence] = dict(frames=len(current_rows), raw_mismatch_frames=mismatches,
                                    count_mismatch_frames=count_mismatch, max_raw_absolute_difference=max_delta,
                                    per_frame=current_rows)
            print('batch', batch, sequence, 'raw mismatches', len(mismatches), 'count mismatches', len(count_mismatch), 'max delta', max_delta, flush=True)
        report['batches'][str(batch)] = videos
        write_json(args.output / 'manifest.json', report)
        del frontend
        torch.cuda.empty_cache()
    report['status'] = 'COMPLETE'
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
