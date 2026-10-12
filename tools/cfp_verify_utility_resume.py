#!/usr/bin/env python3
"""Verify interrupted CPU utility training against a completed fixed reference."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--queries', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Resume evidence is immutable')
    reference = json.loads((args.reference / 'manifest.json').read_text())
    if reference['status'] != 'COMPLETE' or reference['training_device'] != 'CPU' or reference['config_sha256'] != sha256(args.config) or reference['query_manifest_sha256'] != sha256(args.queries / 'manifest.json') or reference['source_sha256'] != sha256('tools/cfp_train_utility.py'):
        raise ValueError('Only matching completed CPU utility references may be checked')
    spec = importlib.util.spec_from_file_location('cfp_resume_tensor_comparison', Path(__file__).with_name('cfp_verify_predictor_resume.py'))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    args.output.mkdir(parents=True)
    run = args.output / 'interrupted'
    command = [sys.executable, '-u', 'tools/cfp_train_utility.py', '--queries', str(args.queries),
               '--config', str(args.config), '--kind', reference['kind'], '--seed', str(reference['seed']), '--output', str(run)]
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONPATH='.')
    report = dict(status='RUNNING', commands=[command, command + ['--resume']],
                  reference_manifest_sha256=sha256(args.reference / 'manifest.json'),
                  source_snapshots=snapshot_sources(args.output, [__file__, 'tools/cfp_verify_predictor_resume.py', 'tools/cfp_train_utility.py', 'cfp/utility.py']),
                  scope='Only the explicitly spawned child is interrupted. Fixed data/config/seed; epoch-boundary recovery, not arbitrary partial-checkpoint safety.')
    write_json(args.output / 'manifest.json', report)
    with (args.output / 'interrupted.log').open('x') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        while child.poll() is None:
            path = run / 'manifest.json'
            if path.exists() and len(json.loads(path.read_text()).get('epochs', [])) >= 3:
                child.send_signal(signal.SIGTERM)
                child.wait()
                break
            time.sleep(.02)
        if child.returncode != -signal.SIGTERM:
            raise RuntimeError('Utility child did not undergo the declared interruption')
    checkpoint = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
    if not 3 <= checkpoint['epoch'] < 8:
        raise ValueError('Need an incomplete completed-epoch utility checkpoint')
    report['interrupted_checkpoint_epoch'] = checkpoint['epoch']
    report['interrupted_checkpoint_sha256'] = sha256(run / 'last.pt')
    with (args.output / 'resumed.log').open('x') as log:
        subprocess.run(command + ['--resume'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    original = torch.load(args.reference / 'epoch8.pt', map_location='cpu', weights_only=False)
    resumed = torch.load(run / 'epoch8.pt', map_location='cpu', weights_only=False)
    resumed_meta = json.loads((run / 'manifest.json').read_text())
    report.update(status='COMPLETE', model_tensors_equal=helper.tensors_equal(original['model'], resumed['model']),
                  optimizer_state_equal=helper.tensors_equal(original['optimizer'], resumed['optimizer']),
                  epoch_losses_equal=[e['loss'] for e in reference['epochs']] == [e['loss'] for e in resumed_meta['epochs']],
                  calibration_equal=original['calibration'] == resumed['calibration'], resumed_manifest_sha256=sha256(run / 'manifest.json'))
    if not all(report[k] for k in ('model_tensors_equal', 'optimizer_state_equal', 'epoch_losses_equal', 'calibration_equal')):
        report['status'] = 'FAILED_PARITY'
        write_json(args.output / 'manifest.json', report)
        raise ValueError('Utility resume differs from uninterrupted fitting')
    write_json(args.output / 'manifest.json', report)
    print({k: report[k] for k in ('status', 'interrupted_checkpoint_epoch', 'model_tensors_equal', 'optimizer_state_equal', 'epoch_losses_equal', 'calibration_equal')})


if __name__ == '__main__':
    main()
