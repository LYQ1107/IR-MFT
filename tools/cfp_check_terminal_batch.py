#!/usr/bin/env python3
"""Diagnose the canonical cache's short final batch without altering online runs."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from cfp.online import CurrentFrameFrontend
from cfp.provenance import sha256, snapshot_sources, write_json
from yolox.utils import postprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    cache = root / 'phase0/causal_v3_dev_detections'
    metadata = json.loads((cache / 'manifest.json').read_text())
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if torch.__version__ != '2.0.0+cu117' or torch.backends.cudnn.version() != 8500:
        raise ValueError('Canonical runtime required')
    frontend = CurrentFrameFrontend(metadata['checkpoint_path'], '/data3/liuyeqiang/IR-MFT_assets/scripts',
                                    input_regime='causal_precomputed', preprocessing='lookup_uint8')
    report = dict(status='RUNNING', cache_manifest_sha256=sha256(cache / 'manifest.json'),
                  source_snapshots=snapshot_sources(args.output, [__file__, 'cfp/online.py', 'tools/cfp_cache_detections.py']),
                  total_cache_frames=metadata['total_frames'], configured_batch=8,
                  terminal_batch_size=metadata['total_frames'] % 8, frames={})
    with np.load(cache / 'SN-013.npz') as bank:
        offsets, reference = bank['offsets'], bank['raw_post_nms']
    for frame in (2997, 2998, 2999, 3000):
        base = Path('/data3/liuyeqiang/datasets/mft26-ir-causal-v1/train/SN-013')
        rgb, ir = [cv2.imread(str(base / branch / f'{frame:06d}.jpg')) for branch in ('visible', 'infrared')]
        image, _ = frontend.transform(rgb, ir, None, frontend.exp.test_size)
        expected = reference[offsets[frame - 1]:offsets[frame]]
        rows = {}
        for batch in (8, 2):
            tensor = torch.from_numpy(image)[None].repeat(batch, 1, 1, 1).cuda().half()
            with torch.inference_mode():
                outputs = postprocess(frontend.detector(tensor), 1, .1, .7)
            actual = np.empty((0, 7), np.float32) if outputs[0] is None else outputs[0].float().cpu().numpy()
            rows[str(batch)] = dict(raw_equal=bool(np.array_equal(actual, expected)), shape_equal=actual.shape == expected.shape,
                                    max_absolute_difference=float(np.max(np.abs(actual - expected))) if actual.shape == expected.shape else None)
        report['frames'][str(frame)] = rows
        print(frame, rows, flush=True)
    report['status'] = 'COMPLETE'
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
