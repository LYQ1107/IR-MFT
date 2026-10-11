#!/usr/bin/env python3
"""Verify real CUDA/ResNet training resume on a tiny synthetic-only fixture."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np
import torch

from cfp.provenance import config_hash, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    protocol_path = args.output / 'protocol.json'
    write_json(protocol_path, dict(state='FROZEN', train=['synthetic_fixture'], dev=[],
               scope='Synthetic infrastructure test only; no fish experiment or accuracy conclusion'))
    config = json.loads(Path('configs/cfp/reid_baseline_v1.json').read_text())
    config.update(dimension=32, scope='Synthetic CUDA ResNet resume verification only; no research metrics',
                  architecture='ImageNet ResNet50, 32D test head, eight synthetic classes')
    config_path = args.output / 'config.json'
    write_json(config_path, config)
    crops = args.output / 'crops'
    crops.mkdir()
    values = np.random.default_rng(0).normal(size=(16, 3, 64, 64)).astype(np.float16)
    np.save(crops / 'crops.npy', values)
    records = [dict(partition='train', train_label=identity, sequence='synthetic_fixture', track_id=identity + 1,
                    frame_id=frame) for identity in range(8) for frame in [1, 30]]
    write_json(crops / 'index.json', dict(records=records, train_identity_count=8))
    write_json(crops / 'manifest.json', dict(status='COMPLETE', modality='rgb',
               input_regime='RGB current-frame only; no IR read', protocol_sha256=sha256(protocol_path),
               index_sha256=sha256(crops / 'index.json'), crops_sha256=sha256(crops / 'crops.npy'),
               scope='Synthetic Gaussian arrays, not acquired fish images or GT crops'))
    base = [sys.executable, '-u', 'tools/cfp_train_reid.py', '--crop-root', str(crops),
            '--protocol', str(protocol_path), '--pretrained', str(args.pretrained),
            '--config', str(config_path), '--seed', '0']
    uninterrupted, resumed = args.output / 'uninterrupted', args.output / 'resumed'
    with (args.output / 'uninterrupted.log').open('x') as stream:
        subprocess.run(base + ['--output', str(uninterrupted)], check=True, stdout=stream, stderr=subprocess.STDOUT)
    # Interrupt only this dedicated fixture child after its first checkpoint.
    # Any in-flight work in epoch two is rolled back to the saved epoch boundary.
    process = subprocess.Popen(base + ['--output', str(resumed)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    interrupted = False
    with (args.output / 'interrupted.log').open('x') as stream:
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get('epoch') == 1 and 'steps' in row:
                process.send_signal(signal.SIGINT)
                interrupted = True
                break
        for line in process.stdout:
            stream.write(line)
    process.wait(timeout=60)
    if not interrupted or json.loads((resumed / 'manifest.json').read_text())['status'] == 'COMPLETE':
        raise ValueError('Fixture was not interrupted before completion')
    saved_epoch = torch.load(resumed / 'last.pt', map_location='cpu')['epoch']
    with (args.output / 'resume.log').open('x') as stream:
        subprocess.run(base + ['--output', str(resumed), '--resume'], check=True, stdout=stream, stderr=subprocess.STDOUT)
    first, second = [torch.load(root / 'last.pt', map_location='cpu') for root in (uninterrupted, resumed)]
    equal = {key: torch.equal(value, second['model'][key]) for key, value in first['model'].items()}
    optimizer_equal = {}
    for key, values in first['optimizer']['state'].items():
        for field, value in values.items():
            other = second['optimizer']['state'][key][field]
            optimizer_equal[f'{key}:{field}'] = torch.equal(value, other) if torch.is_tensor(value) else value == other
    rng_equal = dict(sampler=first['rng_state']['sampler'] == second['rng_state']['sampler'],
                     augmentation=first['rng_state']['augmentation'] == second['rng_state']['augmentation'],
                     torch_cpu=torch.equal(first['rng_state']['torch_cpu'], second['rng_state']['torch_cpu']),
                     torch_cuda=all(torch.equal(a, b) for a, b in zip(first['rng_state']['torch_cuda'], second['rng_state']['torch_cuda'])))
    passed = all(equal.values()) and all(optimizer_equal.values()) and all(rng_equal.values())
    write_json(args.output / 'verification.json', dict(status='PASS' if passed else 'FAIL',
               interrupted_after_completed_epoch=saved_epoch, model_tensors_equal=equal,
               optimizer_tensors_equal=optimizer_equal, rng_states_equal=rng_equal,
               config_sha256=config_hash(config), source_sha256=sha256(__file__),
               trainer_sha256=sha256('tools/cfp_train_reid.py'), model_sha256=sha256('cfp/reid.py'),
               scope='Real CUDA ResNet/AdamW/AMP training, synthetic 16-array fixture, 12 one-step epochs. Infrastructure proof only; no fish-method result.'))
    if not passed:
        raise ValueError('Exact resume verification failed; see preserved comparison')
    print('Exact CUDA model/optimizer/RNG resume parity verified', flush=True)


if __name__ == '__main__':
    main()
