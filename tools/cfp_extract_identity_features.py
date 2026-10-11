#!/usr/bin/env python3
"""Frozen paired ResNet50 feature diagnosis; does not train on development IDs."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet50

from cfp.identity_data import build_identity_index, PairedIdentityDataset
from cfp.provenance import config_hash, sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--pretrained', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples-per-identity', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--input-regime', choices=['legacy_exploratory', 'causal'], required=True)
    args = p.parse_args()
    input_manifest = args.data_root / 'manifest.json'
    if args.input_regime == 'causal':
        if not input_manifest.exists():
            raise ValueError('Causal input requires a verified generation manifest')
        generation = json.loads(input_manifest.read_text())
        if generation['status'] != 'COMPLETE' or generation.get('probe_frames', 0):
            raise ValueError('Causal generation incomplete or diagnostic-only')
    args.output.mkdir(parents=True, exist_ok=True)
    features_path = args.output / 'features.npz'
    if features_path.exists():
        raise FileExistsError(features_path)
    index = build_identity_index(args.data_root, args.protocol, args.samples_per_identity)
    write_json(args.output / 'index.json', index)
    dataset = PairedIdentityDataset(args.data_root, index['records'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.workers, pin_memory=args.device != 'cpu')
    model = resnet50(weights=None)
    model.load_state_dict(torch.load(args.pretrained, map_location='cpu'), strict=True)
    model.fc = torch.nn.Identity()
    model.eval().requires_grad_(False).to(args.device)
    config = dict(backbone='torchvision ResNet50', input_hw=[128, 256], margin=.05, normalization='ImageNet for both branches',
                  encoder_shared=True, encoder_frozen=True, fp16=False, seed=0,
                  samples_per_identity=args.samples_per_identity, batch_size=args.batch_size,
                  workers=args.workers, protocol_sha256=sha256(args.protocol), pretrained_sha256=sha256(args.pretrained))
    config.update(input_regime=args.input_regime, ir_manifest_sha256=sha256(input_manifest) if input_manifest.exists() else None)
    manifest = dict(config=config, config_sha256=config_hash(config), status='RUNNING',
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    source_sha256=sha256(__file__), index_sha256=sha256(args.output / 'index.json'),
                    torch=torch.__version__, device=args.device,
                    weight_domain='ImageNet RGB classification; constructed IR domain shift is a diagnosed condition',
                    scientific_scope='Exploratory legacy-input diagnosis; not strict-online evidence' if args.input_regime == 'legacy_exploratory' else 'Causal constructed input',
                    parameters=sum(p.numel() for p in model.parameters()))
    write_json(args.output / 'manifest.json', manifest)
    rgb_features, ir_features, indices, timings = [], [], [], []
    started_all = time.perf_counter()
    with torch.inference_mode():
        for step, (rgb, ir, positions) in enumerate(loader):
            tensor = torch.cat([rgb, ir]).to(args.device, non_blocking=True)
            if args.device != 'cpu':
                torch.cuda.synchronize()
            started = time.perf_counter()
            features = model(tensor)
            if args.device != 'cpu':
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
            if not torch.isfinite(features).all() or torch.any(features.norm(dim=-1) <= 1e-8):
                raise ValueError('Invalid/collapsed frozen features')
            features = F.normalize(features, dim=-1).cpu().numpy()
            a, b = np.split(features, 2)
            rgb_features.append(a)
            ir_features.append(b)
            indices.extend(positions.tolist())
            if step % 50 == 0:
                print(f'batch {step}/{len(loader)} crops {len(indices)}', flush=True)
    if indices != list(range(len(dataset))):
        raise ValueError('Feature-to-crop alignment broken')
    np.savez_compressed(features_path, rgb=np.concatenate(rgb_features), ir=np.concatenate(ir_features))
    manifest.update(status='COMPLETE', samples=len(indices), feature_dimension=2048,
                    features_sha256=sha256(features_path), total_seconds=time.perf_counter() - started_all,
                    batch_encoder_latency_p50_p95_ms=(np.quantile(timings, [.5, .95]) * 1000).tolist())
    write_json(args.output / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
