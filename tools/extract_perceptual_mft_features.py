#!/usr/bin/env python3
"""Read existing proposal caches and extract causal RGB/new-DiffV2IR evidence.

One allowed physical GPU per worker, one complete sequence per invocation.
No GT is read. Three crop views share the preserved RGB identity encoder;
the additional views are input controls, not new pretrained sensors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

from yolox.tracker.perceptual_features import (
    clipped_crop, color_histogram, foreground, hog, letterbox_tensor_array,
    local_transport,
)
from tools.run_pseudo_ir_comparison import dump, sha256


class AppearanceEncoder(nn.Module):
    """Exact preserved train-only RGB fish ReID architecture; a control."""
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        self.projection = nn.Linear(2048, 512, bias=False)
        self.neck = nn.BatchNorm1d(512)
        self.classifier = nn.Linear(512, 98, bias=False)

    def forward(self, x):
        return F.normalize(self.projection(self.backbone(x)).float(), dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default='configs/perceptual_mft_20261122.json')
    parser.add_argument('--split', choices=['dev', 'test'], required=True)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--checkpoint', default='/data3/liuyeqiang/IR-MFT_assets/cfp/phase1/reid_rgb_seed0/epoch12.pt')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in {'2', '3'}:
        raise RuntimeError('Only one explicitly mapped physical GPU 2 or 3 is permitted')
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.55, 0)
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text())
    protocol = json.loads((ROOT / plan['protocol']).read_text())
    split_key = 'historical_test' if args.split == 'test' else 'dev'
    if args.sequence not in protocol[split_key]:
        raise ValueError('Sequence not in frozen split')
    out = Path(plan['work_root']) / 'features' / args.split / args.sequence
    if args.max_frames:
        out = out / f'smoke_{args.max_frames}'
    out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out).free < 40 * 2**30:
        raise RuntimeError('Work filesystem reserve below 40 GiB')
    if (out / 'complete.json').exists():
        raise RuntimeError('Completed extraction already exists; no silent overwrite')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = AppearanceEncoder()
    model.load_state_dict(checkpoint['model'], strict=True)
    metadata = {k: v for k, v in checkpoint.items() if k != 'model'}
    model = model.cuda().half().eval()
    del checkpoint
    sources, arrays = {}, {}
    expected = None
    for arm, key in [('new', 'new_diffv2ir'), ('rgb', 'rgb')]:
        source = Path(plan['frozen_detection_sources'][f'{key}_{args.split}']) / 'detections' / f'{args.sequence}.npz'
        with np.load(source) as z:
            cache = {k: z[k].copy() for k in z.files}
        frames = cache['frame_ids']
        if not np.array_equal(frames, np.arange(1, len(frames) + 1)):
            raise ValueError('Frame order/coverage violation')
        if expected is not None and not np.array_equal(frames, expected):
            raise ValueError('Different frame coverage in input controls')
        expected = frames
        count = len(cache['boxes'])
        cache.update({k: np.zeros((count, dim), np.float16) for k, dim in
                      [('embedding', 512), ('ir_mask_embedding', 512), ('rgb_mask_embedding', 512),
                       ('ir_hog', 144), ('rgb_hog', 144), ('color', 128),
                       ('flow_rgb', 4), ('flow_ir', 4), ('quality', 2)]})
        arrays[arm] = cache
        sources[arm] = dict(path=str(source), sha256=sha256(source))
    start = time.perf_counter()
    last_rgb = last_ir = None
    previous = {}
    limit = min(len(expected), args.max_frames or len(expected))
    for frame_index in range(limit):
        frame = int(expected[frame_index])
        rgb_path = Path(plan['data_root']) / ('test' if args.split == 'test' else 'train') / args.sequence / 'visible' / f'{frame:06d}.jpg'
        ir_path = Path(plan['generated_ir_root']) / args.split / args.sequence / 'infrared_diffv2ir' / f'{frame:06d}.png'
        rgb, ir = cv2.imread(str(rgb_path)), cv2.imread(str(ir_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or ir is None or ir.ndim != 2 or ir.dtype != np.uint8 or rgb.shape[:2] != ir.shape:
            raise ValueError(f'Invalid current RGB/new-IR pair at frame {frame}')
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        small_rgb = cv2.resize(gray, (960, 540), interpolation=cv2.INTER_AREA)
        small_ir = cv2.resize(ir, (960, 540), interpolation=cv2.INTER_AREA)
        views, placements = [], []
        for arm, data in arrays.items():
            lo, hi = map(int, data['offsets'][frame_index:frame_index + 2])
            boxes = data['boxes'][lo:hi]
            if last_rgb is not None:
                old_lo, old_hi = previous[arm]
                data['flow_rgb'][old_lo:old_hi] = local_transport(last_rgb, small_rgb, data['boxes'][old_lo:old_hi], rgb.shape[:2])
                data['flow_ir'][old_lo:old_hi] = local_transport(last_ir, small_ir, data['boxes'][old_lo:old_hi], rgb.shape[:2])
            for index, box in enumerate(boxes, lo):
                ir_crop = clipped_crop(ir, box, .05)
                rgb_crop = clipped_crop(rgb, box, .05)
                gray_crop = cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2GRAY)
                mask_ir, q_ir = foreground(ir_crop)
                mask_rgb, q_rgb = foreground(gray_crop)
                data['ir_hog'][index] = hog(ir_crop)
                data['rgb_hog'][index] = hog(gray_crop)
                data['color'][index] = color_histogram(rgb_crop)
                data['quality'][index] = [q_rgb, q_ir]
                for field, mask in [('embedding', None), ('ir_mask_embedding', mask_ir), ('rgb_mask_embedding', mask_rgb)]:
                    views.append(letterbox_tensor_array(rgb, box, mask))
                    placements.append((arm, field, index))
            previous[arm] = (lo, hi)
        with torch.inference_mode():
            for lo in range(0, len(views), args.batch_size):
                batch = torch.from_numpy(np.stack(views[lo:lo + args.batch_size])).cuda().half()
                output = model(batch).cpu().numpy()
                for value, (arm, field, index) in zip(output, placements[lo:lo + args.batch_size]):
                    arrays[arm][field][index] = value
        last_rgb, last_ir = small_rgb, small_ir
        if frame % 100 == 0 or frame_index + 1 == limit:
            progress = dict(frame=frame, frames=limit, seconds=time.perf_counter() - start,
                            physical_gpu=physical, peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30)
            dump(out / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    outputs = {}
    for arm, data in arrays.items():
        if args.max_frames:
            count = int(data['offsets'][limit])
            data = {k: (v[:limit] if k == 'frame_ids' else v[:limit + 1] if k == 'offsets' else v[:count]) for k, v in data.items()}
        path = out / f'{arm}.npz'
        temporary = out / f'{arm}.tmp.npz'
        np.savez_compressed(temporary, **data)
        os.replace(temporary, path)
        outputs[arm] = dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size,
                            frames=len(data['frame_ids']), detections=len(data['boxes']))
    manifest = dict(status='SMOKE_COMPLETE' if args.max_frames else 'COMPLETE',
                    split=args.split, sequence=args.sequence, frames=limit,
                    physical_gpu=physical, seconds=time.perf_counter() - start,
                    plan_sha256=sha256(plan_path), sources=sources, outputs=outputs,
                    appearance_checkpoint=args.checkpoint, appearance_sha256=sha256(args.checkpoint),
                    appearance_metadata=metadata, source_sha256=sha256(__file__),
                    feature_code_sha256=sha256(ROOT / 'yolox/tracker/perceptual_features.py'),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    GT_read=False, inputs='Current RGB and November DiffV2IR; LK uses only adjacent t-1,t',
                    flow_semantics='flow_* at row for source observation t-1 may only be used while processing current frame t; never apply it to source frame output')
    dump(out / 'complete.json', manifest)
    print('EXTRACTION_COMPLETE', json.dumps({k: v for k, v in manifest.items() if k != 'appearance_metadata'}), flush=True)


if __name__ == '__main__':
    main()
