#!/usr/bin/env python3
"""Prove exact epoch-boundary continuation of the registered residual trainer."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from cfp.provenance import sha256, write_json


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--after-training', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    proof = args.output.with_suffix('.proof.json')
    if args.output.exists() or proof.exists():
        raise FileExistsError('Resume proof must preserve existing outputs')
    report = dict(status='WAITING_FOR_OWN_GPU0_TRAINING', reference=str(args.reference),
                  config_sha256=sha256(args.config), checks={})
    write_json(proof, report)
    try:
        while True:
            campaign = json.loads(args.after_training.read_text())
            if campaign['status'] == 'COMPLETE':
                break
            if campaign['status'] != 'RUNNING':
                raise ValueError('Own preceding training did not complete')
            time.sleep(10)
        original = json.loads((args.reference / 'manifest.json').read_text())
        if original['status'] != 'COMPLETE' or original['guard']['config_sha256'] != sha256(args.config):
            raise ValueError('Completed same-config reference required')
        for source, digest in original['guard']['sources'].items():
            if sha256(source) != digest:
                raise ValueError('Trainer source drift: ' + source)
        command = [sys.executable, 'tools/cfp_train_residual_relation.py', '--config', str(args.config), '--output', str(args.output)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONPATH='.',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        report['status'] = 'RUNNING'
        report['commands'] = [command + ['--stop-after-epoch', '3'], command + ['--resume']]
        write_json(proof, report)
        with args.output.with_suffix('.stop.log').open('x') as log:
            subprocess.run(report['commands'][0], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        interrupted = json.loads((args.output / 'manifest.json').read_text())
        if interrupted['status'] != 'INTERRUPTED_AT_EPOCH_BOUNDARY' or len(interrupted['epochs']) != 3:
            raise ValueError('Actual epoch3 incomplete state required')
        report['interrupted_checkpoint_sha256'] = sha256(args.output / 'last.pt')
        with args.output.with_suffix('.resume.log').open('x') as log:
            subprocess.run(report['commands'][1], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        a = torch.load(args.reference / 'last.pt', map_location='cpu', weights_only=False)
        b = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
        report['checks'] = {k: equal(a[k], b[k]) for k in a if k != 'report'}
        first, second = copy.deepcopy(a['report']), copy.deepcopy(b['report'])
        report['git_provenance'] = [first.pop('git_commit'), second.pop('git_commit')]
        report['checks']['report_except_git_commit'] = equal(first, second)
        report['status'] = 'EXACT_STATE_EQUAL' if all(report['checks'].values()) else 'FAILED_PARITY'
        report['scope'] = 'Real epoch3 stop/resume on physical GPU0 to fixed epoch8; model, EMA teacher, optimizer, all RNG/sample orders/losses and exact guard. Only report Git provenance excluded. No arbitrary minibatch-crash claim.'
        report['reference_manifest_sha256'] = sha256(args.reference / 'manifest.json')
        report['resumed_manifest_sha256'] = sha256(args.output / 'manifest.json')
        write_json(proof, report)
        if report['status'] != 'EXACT_STATE_EQUAL':
            raise ValueError('Residual continuation is not exact')
    except Exception as error:
        report.update(status='FAILED', error=repr(error))
        write_json(proof, report)
        raise


if __name__ == '__main__':
    main()
