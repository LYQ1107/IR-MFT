#!/usr/bin/env python3
"""Interrupt our own predictor child and compare resumed SGD with a fixed run."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

from cfp.provenance import sha256, snapshot_sources, write_json


def tensors_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(tensors_equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(tensors_equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--queries', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--query-config', type=Path, required=True)
    p.add_argument('--training-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Resume proof is immutable')
    reference = json.loads((args.reference / 'manifest.json').read_text())
    if reference['status'] != 'COMPLETE' or reference['device'] != 'cuda':
        raise ValueError('Use a completed CUDA run with identical inputs')
    expected = {'config_sha256': args.config, 'query_config_sha256': args.query_config,
                'training_config_sha256': args.training_config, 'query_manifest_sha256': args.queries / 'manifest.json'}
    if any(sha256(path) != reference[k] for k, path in expected.items()):
        raise ValueError('Resume proof differs from fixed reference data/config')
    if reference['source_snapshots']['tools/cfp_train_predictor.py'] != sha256('tools/cfp_train_predictor.py'):
        raise ValueError('Trainer bytes differ from uninterrupted reference')
    args.output.mkdir(parents=True)
    run = args.output / 'interrupted'
    command = [sys.executable, '-u', 'tools/cfp_train_predictor.py', '--queries', str(args.queries),
               '--config', str(args.config), '--query-config', str(args.query_config), '--training-config', str(args.training_config),
               '--architecture', reference['architecture'], '--objective', reference['objective'], '--history', reference['history'],
               '--seed', str(reference['seed']), '--device', 'cuda', '--gpu-memory-fraction', str(reference['gpu_memory_fraction']), '--output', str(run)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='2', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONPATH='.')
    report = dict(status='RUNNING', reference_manifest_sha256=sha256(args.reference / 'manifest.json'),
                  commands=[command, command + ['--resume']],
                  source_snapshots=snapshot_sources(args.output, [__file__, 'tools/cfp_train_predictor.py', 'cfp/predictive.py']),
                  scope='SIGTERM only the explicitly spawned child PID of this verification; never touch another task. Epoch-boundary replay proof for this fixed predictor/runtime, not arbitrary mid-step continuation.')
    write_json(args.output / 'manifest.json', report)
    with (args.output / 'interrupted.log').open('x') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        while child.poll() is None:
            manifest_path = run / 'manifest.json'
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if len(manifest.get('epochs', [])) >= 3:
                    child.send_signal(signal.SIGTERM)
                    child.wait()
                    break
            time.sleep(.05)
        if child.returncode != -signal.SIGTERM:
            raise RuntimeError('The controlled child was not interrupted as declared')
    interrupted = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
    epoch = interrupted['epoch']
    if not 3 <= epoch < 8:
        raise ValueError('Need a real incomplete epoch-boundary checkpoint')
    report['interrupted_checkpoint_epoch'] = epoch
    report['interrupted_checkpoint_sha256'] = sha256(run / 'last.pt')
    write_json(args.output / 'manifest.json', report)
    with (args.output / 'resumed.log').open('x') as log:
        subprocess.run(command + ['--resume'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    original = torch.load(args.reference / 'last.pt', map_location='cpu', weights_only=False)
    resumed = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
    original_losses = [v['loss'] for v in reference['epochs']]
    resumed_meta = json.loads((run / 'manifest.json').read_text())
    report.update(status='COMPLETE', model_tensors_equal=tensors_equal(original['model'], resumed['model']),
                  optimizer_state_equal=tensors_equal(original['optimizer'], resumed['optimizer']),
                  epoch_losses_equal=original_losses == [v['loss'] for v in resumed_meta['epochs']],
                  resumed_manifest_sha256=sha256(run / 'manifest.json'))
    if not all(report[k] for k in ('model_tensors_equal', 'optimizer_state_equal', 'epoch_losses_equal')):
        report['status'] = 'FAILED_PARITY'
        write_json(args.output / 'manifest.json', report)
        raise ValueError('Resumed predictor is not numerically exact')
    write_json(args.output / 'manifest.json', report)
    print({k: report[k] for k in ('status', 'interrupted_checkpoint_epoch', 'model_tensors_equal', 'optimizer_state_equal', 'epoch_losses_equal')})


if __name__ == '__main__':
    main()
