#!/usr/bin/env python3
"""Create declared RGB evidence controls from verified causal prefix states."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from cfp.provenance import config_hash, sha256, write_json


def enhance(channel, percentiles):
    values = channel.astype(np.float32)
    nonzero = values[values > 0]
    if len(nonzero) < 20:
        return channel.copy()
    low, high = np.percentile(nonzero, percentiles)
    if high <= low:
        return channel.copy()
    return np.clip((values - low) / (high - low) * 255, 0, 255).astype(np.uint8)


def transform(frame, mode, parameters, background=None, clahe=None):
    if mode == 'rgb_contrast':
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if background is None:
        raise ValueError('Past-only RGB background required')
    difference = cv2.absdiff(frame, np.clip(background, 0, 255).astype(np.uint8))
    channels = [enhance(channel, parameters['percentiles']) for channel in cv2.split(difference)]
    response = cv2.merge(channels).astype(np.float32) / 255
    return np.uint8(np.clip(np.power(response, parameters['gamma']) * 255, 0, 255))


def build_chunk(task):
    cv2.setNumThreads(1)
    root, output, split, sequence, images, prefix, mode, parameters = task
    root, output = Path(root), Path(output)
    background = None
    if mode == 'rgb_background_difference' and prefix:
        if sha256(prefix['path']) != prefix['sha256']:
            raise ValueError('Canonical prefix state changed')
        background = np.load(prefix['path'], allow_pickle=False)
    clahe = cv2.createCLAHE(parameters['clip_limit'], tuple(parameters['tile_grid_size'])) if mode == 'rgb_contrast' else None
    rows, durations = [], []
    for image in images:
        source = root / split / image['vi_file_name']
        frame = cv2.imread(str(source))
        if frame is None:
            raise ValueError(f'Cannot read {source}')
        if mode == 'rgb_background_difference' and background is None:
            background = frame.astype(np.float32)
        started = time.perf_counter()
        response = transform(frame, mode, parameters, background, clahe)
        if mode == 'rgb_background_difference':
            cv2.accumulateWeighted(frame, background, parameters['alpha'])
        durations.append(time.perf_counter() - started)
        target = output / split / image['ir_file_name']
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        if not cv2.imwrite(str(target), response, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(target)
        rows.append(dict(frame_id=image['frame_id'], source=image['vi_file_name'], derived=image['ir_file_name'],
                         source_sha256=sha256(source), derived_sha256=sha256(target)))
    report = dict(sequence=sequence, first_frame=images[0]['frame_id'], last_frame=images[-1]['frame_id'],
                  frames=len(images), generation_seconds=sum(durations),
                  latency_p50_p95_ms=(np.quantile(durations, [.5, .95]) * 1000).tolist(), prefix=prefix)
    write_json(output / 'provenance' / f'{split}_{sequence}_{images[0]["frame_id"]:06d}.json', dict(report, images=rows))
    print(mode, sequence, images[0]['frame_id'], images[-1]['frame_id'], flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--causal-root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--mode', choices=['rgb_contrast', 'rgb_background_difference'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Control datasets are immutable')
    config = json.loads(args.config.read_text())
    canonical = json.loads((args.causal_root / 'manifest.json').read_text())
    if config['state'] != 'FROZEN' or canonical['status'] != 'COMPLETE' or canonical.get('probe_frames', 0):
        raise ValueError('Frozen controls require complete canonical causal inputs')
    parameters = config[args.mode]
    if args.mode == 'rgb_background_difference' and parameters['alpha'] != canonical['config']['alpha']:
        raise ValueError('Background control must match available history exactly')
    annotation = json.loads((args.causal_root / 'annotations/train.json').read_text())
    names = {video['id']: video['file_name'] for video in annotation['videos']}
    images = {}
    for image in annotation['images']:
        images[(names[image['video_id']], int(image['frame_id']))] = image
    args.output.mkdir(parents=True)
    (args.output / 'annotations').symlink_to(args.causal_root / 'annotations', target_is_directory=True)
    tasks = []
    for sequence in sorted({chunk['sequence'] for chunk in canonical['chunks']}):
        target = args.output / 'train' / sequence
        target.mkdir(parents=True)
        for name in ['visible', 'gt', 'seqinfo.ini']:
            source = args.causal_root / 'train' / sequence / name
            (target / name).symlink_to(source, target_is_directory=source.is_dir())
    for chunk in canonical['chunks']:
        selected = [images[(chunk['sequence'], frame)] for frame in range(chunk['first_frame'], chunk['last_frame'] + 1)]
        tasks.append((str(args.causal_root), str(args.output), 'train', chunk['sequence'], selected,
                      chunk['prefix_checkpoint'], args.mode, parameters))
    manifest = dict(status='RUNNING', config=dict(mode=args.mode, parameters=parameters),
                    config_sha256=config_hash(config), canonical_generation_manifest_sha256=sha256(args.causal_root / 'manifest.json'),
                    probe_frames=0, source_sha256=sha256(__file__),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    scope='RGB auxiliary-view control; stored in annotation IR slot for schema compatibility; no thermal sensor/temperature')
    write_json(args.output / 'manifest.json', manifest)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        reports = list(pool.map(build_chunk, tasks))
    manifest.update(status='COMPLETE', total_seconds=time.perf_counter() - started, total_frames=sum(report['frames'] for report in reports), chunks=reports)
    write_json(args.output / 'manifest.json', manifest)
    print(args.mode, 'complete', manifest['total_frames'], 'frames', manifest['total_seconds'], 'seconds', flush=True)


if __name__ == '__main__':
    main()
