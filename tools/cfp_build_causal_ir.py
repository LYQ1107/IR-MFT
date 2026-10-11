#!/usr/bin/env python3
"""Build separate causal inputs, with sequential state and immutable manifests."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from cfp.provenance import config_hash, sha256, write_json
from cfp.pseudo_ir import CausalPseudoIR, SOURCES


def prepare_video(task, chunk_size):
    """Save only prefix-derived states; future frames never enter these states."""
    cv2.setNumThreads(1)
    root, output, split, sequence, images, scripts, config = task
    stream = CausalPseudoIR(scripts, config['mode'], config['alpha'])
    chunks = []
    started = time.perf_counter()
    state_dir = Path(output) / 'state_checkpoints' / split / sequence
    state_dir.mkdir(parents=True)
    last_chunk_start = ((len(images) - 1) // chunk_size) * chunk_size
    for position, image in enumerate(images):
        if position % chunk_size == 0:
            checkpoint = None
            if position and stream.mode == 'temporal_ema':
                path = state_dir / f'before_{image["frame_id"]:06d}.npy'
                np.save(path, stream.background, allow_pickle=False)
                checkpoint = dict(path=str(path), sha256=sha256(path), prior_frames=position)
            chunks.append((root, output, split, sequence, images[position:position + chunk_size],
                           scripts, config, position, checkpoint))
        if position >= last_chunk_start:
            break
        if stream.mode == 'temporal_ema':
            path = Path(root) / split / image['vi_file_name']
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f'Cannot decode prefix {path}')
            stream.advance_background(frame)
    return dict(sequence=sequence, chunks=chunks, preparation_seconds=time.perf_counter() - started)


def prepare_task(task):
    return prepare_video(*task)


def build_chunk(task):
    cv2.setNumThreads(1)
    root, output, split, sequence, images, scripts, config, position, checkpoint = task
    root, output = Path(root), Path(output)
    generator = CausalPseudoIR(scripts, config['mode'], config['alpha'])
    if checkpoint:
        if sha256(checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('Prefix checkpoint changed')
        generator.background = np.load(checkpoint['path'], allow_pickle=False)
        generator.frames_seen = checkpoint['prior_frames']
    rows, durations = [], []
    start_all = time.perf_counter()
    for count, image in enumerate(images):
        source = root / split / image['vi_file_name']
        destination = output / split / image['ir_file_name']
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        frame = cv2.imread(str(source))
        if frame is None:
            raise ValueError(f'Cannot decode {source}')
        started = time.perf_counter()
        pseudo = generator(frame)
        durations.append(time.perf_counter() - started)
        if not cv2.imwrite(str(destination), pseudo, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f'Cannot save {destination}')
        rows.append(dict(frame_id=image['frame_id'], rgb=image['vi_file_name'], ir=image['ir_file_name'],
                         rgb_sha256=sha256(source), ir_sha256=sha256(destination)))
        if count % 1000 == 0:
            print(f'{sequence} chunk@{position}: {count + 1}/{len(images)}', flush=True)
    result = dict(sequence=sequence, split=split, status='COMPLETE', frames=len(rows),
                  config_sha256=config_hash(config), source_file=SOURCES[config['mode']][0],
                  source_sha256=SOURCES[config['mode']][1], generator_seconds=sum(durations),
                  generator_latency_p50_p95_ms=(np.quantile(durations, [.5, .95]) * 1000).tolist(),
                  total_seconds=time.perf_counter() - start_all, frames_manifest=rows)
    result.update(first_frame=images[0]['frame_id'], last_frame=images[-1]['frame_id'], prefix_checkpoint=checkpoint)
    write_json(output / 'provenance' / f'{split}_{sequence}_{position:06d}.json', result)
    return {key: value for key, value in result.items() if key != 'frames_manifest'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--scripts-root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'test'], default='train')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--probe-frames', type=int, default=0)
    p.add_argument('--chunk-size', type=int, default=1000, help='Offline generation scheduling only; exact prefix states preserved')
    args = p.parse_args()
    if args.chunk_size <= 0:
        raise ValueError('Chunk size must be positive')
    if args.output.exists():
        raise FileExistsError('Causal datasets are immutable; use a new output path')
    if args.data_root.resolve() == args.output.resolve():
        raise ValueError('Never overwrite the original dataset')
    config = json.loads(args.config.read_text())
    if config['state'] != 'FROZEN':
        raise ValueError('Freeze the causal preprocessing before experiments')
    # Verify scripts before creating any derived output.
    CausalPseudoIR(args.scripts_root, config['mode'], config['alpha'])
    protocol = json.loads(args.protocol.read_text())
    annotation_path = args.data_root / 'annotations' / (args.split + '.json')
    annotation = json.loads(annotation_path.read_text())
    names = {v['id']: v['file_name'] for v in annotation['videos']}
    sequences = protocol['train'] + protocol['dev'] if args.split == 'train' else protocol['historical_test']
    tasks = []
    args.output.mkdir(parents=True)
    (args.output / 'annotations').symlink_to(args.data_root / 'annotations', target_is_directory=True)
    for sequence in sequences:
        images = sorted([x for x in annotation['images'] if names[x['video_id']] == sequence], key=lambda x: x['frame_id'])
        if args.probe_frames:
            images = images[:args.probe_frames]
        if [x['frame_id'] for x in images] != list(range(1, len(images) + 1)):
            raise ValueError(f'Non-contiguous causal stream {sequence}')
        target = args.output / args.split / sequence
        target.mkdir(parents=True)
        # Visible images, GT and sequence metadata are linked, never duplicated.
        for name in ('visible', 'gt', 'seqinfo.ini'):
            source = args.data_root / args.split / sequence / name
            if source.exists():
                (target / name).symlink_to(source, target_is_directory=source.is_dir())
        tasks.append((str(args.data_root), str(args.output), args.split, sequence, images,
                      str(args.scripts_root), config))
    manifest = dict(status='RUNNING', config=config, config_sha256=config_hash(config),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    protocol_sha256=sha256(args.protocol), annotation_sha256=sha256(annotation_path),
                    wrapper_sha256=sha256(__file__), adapter_sha256=sha256('cfp/pseudo_ir.py'),
                    cv2=cv2.__version__, probe_frames=args.probe_frames,
                    chunk_size=args.chunk_size, scheduling='prefix-state checkpoints; no future data in state',
                    warning='Probe inputs are incomplete and cannot be evaluated as a whole-video dataset' if args.probe_frames else None)
    write_json(args.output / 'manifest.json', manifest)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        preparations = list(pool.map(prepare_task, [(task, args.chunk_size) for task in tasks]))
        chunk_tasks = [chunk for item in preparations for chunk in item['chunks']]
        manifest.update(status='PREFIX_STATES_READY', prefix_preparations=[{k: v for k, v in item.items() if k != 'chunks'} for item in preparations])
        write_json(args.output / 'manifest.json', manifest)
        reports = list(pool.map(build_chunk, chunk_tasks))
    manifest.update(status='COMPLETE', total_seconds=time.perf_counter() - started, chunks=reports,
                    total_frames=sum(report['frames'] for report in reports))
    write_json(args.output / 'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'config'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
