#!/usr/bin/env python3
"""Cache aligned normalized single-modality GT crops outside the source tree."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from cfp.identity_data import build_identity_index, SingleIdentityDataset
from cfp.provenance import sha256, write_json


def worker_init(_):
    cv2.setNumThreads(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--modality', choices=['rgb', 'ir'], required=True)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Crop caches are immutable')
    if args.modality == 'ir':
        generation = json.loads((args.data_root / 'manifest.json').read_text())
        if generation['status'] != 'COMPLETE' or generation.get('probe_frames', 0):
            raise ValueError('IR training requires complete causal inputs')
    view_mode = generation['config'].get('mode') if args.modality == 'ir' else None
    if view_mode not in (None, 'rgb_contrast', 'rgb_background_difference'):
        raise ValueError('Unknown auxiliary control view')
    input_regime = ('RGB current-frame only; no IR read' if args.modality == 'rgb' else
                    f'complete causal RGB auxiliary control: {view_mode}' if view_mode else
                    'complete causal constructed IR')
    args.output.mkdir(parents=True)
    index = build_identity_index(args.data_root, args.protocol)
    write_json(args.output / 'index.json', index)
    dataset = SingleIdentityDataset(args.data_root, index['records'], args.modality)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=args.workers,
                                        worker_init_fn=worker_init, persistent_workers=args.workers > 0)
    temporary = args.output / 'crops.partial.npy'
    shape = (len(dataset), 3, 128, 256)
    mapped = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16, shape=shape)
    manifest = dict(status='RUNNING', modality=args.modality, shape=list(shape), dtype='float16',
                    preprocessing='Exact crop_to_tensor ImageNet normalization, then float16 storage; no augmentation',
                    protocol_sha256=sha256(args.protocol), index_sha256=sha256(args.output / 'index.json'),
                    source_sha256=sha256(__file__), identity_source_sha256=sha256('cfp/identity_data.py'),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    input_regime=input_regime, auxiliary_view=view_mode,
                    generation_manifest_sha256=sha256(args.data_root / 'manifest.json') if args.modality == 'ir' else None)
    write_json(args.output / 'manifest.json', manifest)
    started = time.perf_counter()
    next_position = 0
    for step, (crops, positions) in enumerate(loader):
        expected = list(range(next_position, next_position + len(positions)))
        if positions.tolist() != expected or not torch.isfinite(crops).all():
            raise ValueError('Invalid crop values or index order')
        mapped[positions.numpy()] = crops.numpy().astype(np.float16)
        next_position += len(positions)
        if step % 100 == 0:
            print(f'{args.modality} crops: {next_position}/{len(dataset)}', flush=True)
    if next_position != len(dataset):
        raise ValueError('Incomplete crop cache')
    mapped.flush()
    del mapped
    final = args.output / 'crops.npy'
    temporary.rename(final)
    manifest.update(status='COMPLETE', total_seconds=time.perf_counter() - started, crops_sha256=sha256(final), bytes=final.stat().st_size)
    write_json(args.output / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
