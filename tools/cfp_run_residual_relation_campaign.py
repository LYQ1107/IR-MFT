#!/usr/bin/env python3
"""Sequential frozen student controls on explicitly reserved physical GPU0."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan['state'] != 'FROZEN' or plan['physical_gpu'] != 0 or args.output.exists():
        raise ValueError('Frozen single reserved GPU0/new summary required')
    report = dict(status='RUNNING', plan_sha256=sha256(args.plan), runs={},
                  sources=snapshot_sources(args.output.parent / (args.output.stem + '_sources'), [__file__, args.plan]))
    write_json(args.output, report)
    try:
        for spec in plan['jobs']:
            config, output = Path(spec['config']), Path(spec['output'])
            command = [sys.executable, 'tools/cfp_train_residual_relation.py', '--config', str(config), '--output', str(output)]
            if not output.exists():
                with output.with_suffix('.log').open('w') as stream:
                    subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                        env=dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES='0',
                                 CUBLAS_WORKSPACE_CONFIG=':4096:8', OPENBLAS_NUM_THREADS='1'))
            result = json.loads((output / 'manifest.json').read_text())
            if result['status'] != 'COMPLETE' or result['guard']['config_sha256'] != sha256(config):
                raise ValueError('Preserved student incomplete/config drift')
            if sha256(output / 'inference.pt') != result['inference_sha256']:
                raise ValueError('Export drift')
            for source, digest in result['guard']['sources'].items():
                if sha256(source) != digest:
                    raise ValueError('Source drift: ' + source)
            report['runs'][output.name] = dict(manifest_sha256=sha256(output / 'manifest.json'), command=command, **result)
            write_json(args.output, report)
            print(output.name, result['validation'], flush=True)
        report['status'] = 'COMPLETE'; write_json(args.output, report)
    except Exception as error:
        report.update(status='FAILED', error=repr(error)); write_json(args.output, report); raise


if __name__ == '__main__':
    main()
