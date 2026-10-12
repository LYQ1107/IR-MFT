#!/usr/bin/env python3
"""One frozen timestamp repair: unchanged queries/models/epochs, three seeds."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan['state'] != 'FROZEN' or plan['time_encoding'] not in ['log1p_15000','linear_clipped']:
        raise ValueError('Frozen bounded timestamp repair required')
    for path, expected in plan['input_sha256'].items():
        if sha256(path) != expected:
            raise ValueError('Timestamp repair input changed: ' + path)
    if args.output.exists():
        raise FileExistsError('Time-repair queue is immutable')
    summary = dict(status='RUNNING', plan_sha256=sha256(args.plan),
                   source_snapshots=snapshot_sources(args.output.with_suffix(''), [__file__, args.plan]), runs={})
    write_json(args.output, summary)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='2', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONPATH='.')
    for job in plan['jobs']:
        if shutil.disk_usage(args.output.parent).free < plan['minimum_free_gib'] * 2**30:
            raise OSError('Timestamp repair storage reserve reached')
        out = Path(job['output'])
        command = [sys.executable, '-u', 'tools/cfp_train_predictor.py', *plan['common_args'],
                   '--queries', job['queries'], '--architecture', job['architecture'],
                   '--objective', job['objective'], '--seed', str(job['seed']), '--output', str(out)]
        if job.get('training_config'):command+=['--training-config',job['training_config']]
        if job.get('identity_representation'):command+=['--identity-representation',job['identity_representation']]
        if job.get('second_rgb_features_root'):command+=['--second-rgb-features-root',job['second_rgb_features_root']]
        if not out.exists():
            print('Starting', out.name, flush=True)
            with out.with_name(out.name + '.log').open('x') as stream:
                subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        manifest = json.loads((out / 'manifest.json').read_text())
        if manifest['status'] != 'COMPLETE' or manifest['time_encoding'] != plan['time_encoding'] or manifest['architecture'] != job['architecture'] or manifest['objective'] != job['objective'] or manifest['seed'] != job['seed'] or sha256(out / 'epoch8.pt') != manifest['checkpoint_sha256']:
            raise ValueError('Completed timestamp-repair model differs')
        if manifest.get('identity_representation','rgb')!=job.get('identity_representation','rgb'):raise ValueError('Frozen representation changed')
        summary['runs'][out.name] = dict(command=command, status='COMPLETE', manifest_sha256=sha256(out / 'manifest.json'))
        write_json(args.output, summary)
        print('Completed', out.name, flush=True)
    summary['status'] = 'COMPLETE'
    write_json(args.output, summary)


if __name__ == '__main__':
    main()
