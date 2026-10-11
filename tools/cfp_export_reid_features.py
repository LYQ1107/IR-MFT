#!/usr/bin/env python3
"""Export a declared frozen representation without training or dev selection."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from cfp.provenance import sha256, write_json
from cfp.reid import FishReID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--embedding-stage', choices=['pre_neck', 'post_neck'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Representation exports are immutable')
    crop = json.loads((args.crop_root / 'manifest.json').read_text())
    if crop['status'] != 'COMPLETE' or sha256(args.crop_root / 'crops.npy') != crop['crops_sha256']:
        raise ValueError('Crop inputs changed')
    if sha256(args.crop_root / 'index.json') != crop['index_sha256']:
        raise ValueError('Crop indexing changed')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if checkpoint['metadata']['crop_manifest_sha256'] != sha256(args.crop_root / 'manifest.json'):
        raise ValueError('Encoder/crop provenance mismatch')
    if checkpoint['epoch'] != checkpoint['config']['epochs']:
        raise ValueError('Declared representation baseline uses fixed last epoch')
    state = checkpoint['model']
    model = FishReID(state['classifier.weight'].shape[0], state['projection.weight'].shape[0])
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).cuda()
    args.output.mkdir(parents=True)
    shutil.copyfile(args.crop_root / 'index.json', args.output / 'index.json')
    manifest = dict(checkpoint['metadata'], status='RUNNING', checkpoint_sha256=sha256(args.checkpoint),
                    embedding_stage=args.embedding_stage, representation_only_export=True,
                    index_sha256=crop['index_sha256'], export_source_sha256=sha256(__file__),
                    export_model_source_sha256=sha256('cfp/reid.py'),
                    export_git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    scope='Frozen checkpoint representation export; no train update, retrieval selection or new weights')
    write_json(args.output / 'manifest.json', manifest)
    mapped = np.load(args.crop_root / 'crops.npy', mmap_mode='r')
    features, latencies = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(mapped), args.batch_size):
            images = torch.from_numpy(np.array(mapped[start:start + args.batch_size])).cuda().float()
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.cuda.amp.autocast():
                embedding, _ = model(images, embedding_stage=args.embedding_stage)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t)
            if not torch.isfinite(embedding).all() or not torch.allclose(embedding.norm(dim=-1), torch.ones(len(embedding), device='cuda'), atol=1e-4):
                raise ValueError('Invalid frozen representation')
            features.append(embedding.cpu().numpy())
    values = np.concatenate(features)
    np.savez_compressed(args.output / 'features.npz', embedding=values)
    manifest.update(status='COMPLETE', features_sha256=sha256(args.output / 'features.npz'),
                    encoding_seconds=time.perf_counter() - started,
                    embedding_variance_sum=float(values.var(axis=0).sum()),
                    encoder_batch_latency_p50_p95_ms=(np.quantile(latencies, [.5, .95]) * 1000).tolist(),
                    latency_scope='Cache-backed crop encoding; excludes raw decoding, generation, detector and association')
    write_json(args.output / 'manifest.json', manifest)
    print(args.output, 'complete', manifest['embedding_variance_sum'], flush=True)


if __name__ == '__main__':
    main()
