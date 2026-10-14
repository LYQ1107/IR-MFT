#!/usr/bin/env python3
"""Produce current-only native features from one completed visible-token fit.

The proposal/frame protocol is copied from the frozen DINO producer, but the
encoder coordinate is kept explicit so the result cannot be mistaken for a
frozen or supervised-DINO bank.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from cfp.provenance import sha256, snapshot_sources, write_json
from cfp.visible_token_native_encoder import CurrentVisibleTokenIdentityEncoder


def idle_gpu(physical):
    if physical not in ('0', '1', '2', '3'):
        raise ValueError('Exactly one approved idle GPU0-3 required')
    memory, utilization = [int(v.strip()) for v in subprocess.check_output(
        ['nvidia-smi', '-i', physical, '--query-gpu=memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True).split(',')]
    if memory > 64 or utilization:
        raise ValueError('Selected GPU is not idle; preserve other jobs')


def main():
    ap = argparse.ArgumentParser()
    for name in ('config', 'data-root', 'frame-index', 'detections', 'reference',
                 'weights', 'source-proof', 'training', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--inner', required=True)
    ap.add_argument('--mode', choices=('masked_visible_tokens', 'visible_target'), required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--fit', nargs='+', required=True)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (config['state'] != 'FROZEN' or args.video not in config['videos']
            or args.output.exists() or args.output.resolve().is_relative_to(Path.cwd().resolve())):
        raise ValueError('Declared full-native video and new outsideGit output required')
    if args.video == args.inner or args.video in args.fit or args.inner in args.fit:
        raise ValueError('Outer/inner/fitting videos must be disjoint')
    idle_gpu(physical)
    frame_manifest = json.loads((args.frame_index / 'manifest.json').read_text())
    detection_manifest = json.loads((args.detections / 'manifest.json').read_text())
    frame_path = args.frame_index / (args.video + '.frames.json')
    detection_path = args.detections / (args.video + '.npz')
    if (frame_manifest['status'] != 'COMPLETE_INDEX_ONLY'
            or frame_manifest['detection_manifest_sha256'] != sha256(args.detections / 'manifest.json')
            or sha256(frame_path) != frame_manifest['videos'][args.video]['frame_paths_sha256']
            or sha256(detection_path) != detection_manifest['caches'][args.video]['sha256']):
        raise ValueError('Exact framepath-only index and causal proposal cache required')
    paths = json.loads(frame_path.read_text())
    with np.load(detection_path, allow_pickle=False) as arrays:
        frames, offsets, boxes, scores = [arrays[k].copy() for k in ('frame_ids', 'offsets', 'boxes', 'scores')]
    if (not np.array_equal(frames, np.arange(1, len(frames) + 1))
            or [row['frame_id'] for row in paths] != frames.tolist()
            or len(offsets) != len(frames) + 1 or offsets[0] != 0 or offsets[-1] != len(boxes)
            or np.any(np.diff(offsets) < 0) or len(scores) != len(boxes)
            or not np.isfinite(boxes).all() or not np.isfinite(scores).all()):
        raise ValueError('Full frame/proposal order and empty frames must be preserved')
    if Path(frame_manifest['data_root']).resolve() != args.data_root.resolve():
        raise ValueError('Declared current image source differs')
    planned = len(boxes) * (2 * 384 * 4 + 4 * 384 * 2 + 4 * 4 + 1) + len(offsets) * 8 + len(frames) * 8
    if planned > config['max_feature_bytes_per_video'] or shutil.disk_usage(args.output.parent).free - planned < 40 * 2**30:
        raise OSError('40GiB reserve or feature budget unavailable')
    for row in paths:
        relative = Path(row['rgb_file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Current RGB must stay inside declared data root')
    torch.set_num_threads(2)
    idle_gpu(physical)
    torch.cuda.set_per_process_memory_fraction(.4)
    encoder = CurrentVisibleTokenIdentityEncoder(
        args.reference, args.weights, args.source_proof, args.training, args.video,
        args.mode, args.seed, args.fit, args.inner, device='cuda', batch_size=config['encoder_batch_padding'])
    args.output.mkdir(parents=True)
    report = dict(
        status='RUNNING', config=config, config_sha256=sha256(args.config), video=args.video,
        mode=args.mode, seed=args.seed, inner=args.inner, fit=args.fit, physical_gpu=physical,
        encoder=encoder.metadata, frame_paths_sha256=sha256(frame_path), detection_sha256=sha256(detection_path),
        detection_manifest_sha256=sha256(args.detections / 'manifest.json'),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        sources=snapshot_sources(args.output / 'sources', [__file__, 'cfp/visible_token_native_encoder.py',
            'cfp/visible_token_fish_mae.py', 'cfp/dinov2_native_encoder.py', 'cfp/dinov2_identity.py',
            'cfp/identity_data.py', 'cfp/local_identity_encoder.py', 'cfp/provenance.py', str(args.config)]),
        completed_frames=0, frames=len(frames), proposals=len(boxes), scope=config['scope'],
        guard='No annotations/GT/physicalIDs/teacher/future frames loaded; only current RGB and frozen current proposal rows.',
        online_GT_teacher_future_used=False)
    np.save(args.output / 'frame_ids.npy', frames, allow_pickle=False)
    np.save(args.output / 'offsets.npy', offsets, allow_pickle=False)
    shapes = {'embedding': (384,), 'raw_cls': (384,), 'local4': (4, 384), 'visibility4': (4,), 'valid': ()}
    mapped = {name: np.lib.format.open_memmap(
        args.output / (name + '.npy'), mode='w+',
        dtype=np.bool_ if name == 'valid' else np.float16 if name == 'local4' else np.float32,
        shape=(len(boxes), *shape)) for name, shape in shapes.items()}
    write_json(args.output / 'manifest.json', report)
    totals, frame_seconds, input_digest = {}, [], hashlib.sha256()
    started = time.monotonic()
    try:
        for index, frame in enumerate(frames):
            frame_start = time.perf_counter()
            rgb = cv2.imread(str(args.data_root / 'train' / paths[index]['rgb_file']), cv2.IMREAD_COLOR)
            if rgb is None:
                raise ValueError('Current RGB decode failed: ' + str(paths[index]['rgb_file']))
            lo, hi = offsets[index:index + 2]
            values = encoder.encode(rgb, boxes[lo:hi])
            for name in mapped:
                mapped[name][lo:hi] = values[name]
            input_digest.update(int(frame).to_bytes(8, 'little'))
            input_digest.update(hashlib.sha256(rgb.tobytes()).digest())
            input_digest.update(boxes[lo:hi].tobytes())
            for key, value in values['stage_seconds'].items():
                totals[key] = totals.get(key, 0.) + value
            frame_seconds.append(time.perf_counter() - frame_start)
            report['completed_frames'] = index + 1
            if (index + 1) % 128 == 0 or index + 1 == len(frames):
                for value in mapped.values():
                    value.flush()
                report.update(stage_seconds=totals, elapsed_seconds=time.monotonic() - started)
                write_json(args.output / 'manifest.json', report)
                print(json.dumps(dict(video=args.video, frames=index + 1, total=len(frames),
                                      elapsed=report['elapsed_seconds'])), flush=True)
                if shutil.disk_usage(args.output).free < 40 * 2**30:
                    raise OSError('40GiB reserve')
        for value in mapped.values():
            value.flush()
        report.update(
            status='COMPLETE_NATIVE_CURRENT_VISIBLE_TOKEN_FEATURES', current_input_digest=input_digest.hexdigest(),
            stage_seconds=totals, elapsed_seconds=time.monotonic() - started,
            current_frame_latency_p50_p95_ms=(1000 * np.percentile(frame_seconds, [50, 95])).tolist(),
            empty_proposal_frames=int((np.diff(offsets) == 0).sum()), valid_crop_proposals=int(mapped['valid'].sum()),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            artifacts={name: dict(path=str(args.output / (name + '.npy')), sha256=sha256(args.output / (name + '.npy')),
                              shape=list(value.shape), dtype=str(value.dtype)) for name, value in mapped.items()},
            frame_ids_sha256=sha256(args.output / 'frame_ids.npy'), offsets_sha256=sha256(args.output / 'offsets.npy'))
        write_json(args.output / 'manifest.json', report)
    except Exception as error:
        report.update(status='FAILED_ACTUAL_EXECUTION', error=str(error))
        write_json(args.output / 'manifest.json', report)
        raise


if __name__ == '__main__':
    main()
