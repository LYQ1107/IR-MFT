#!/usr/bin/env python3
"""Bounded outer-video memory replays; no partial-video aggregate metric."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan['state'] != 'FROZEN' or not 1 <= plan['workers'] <= 2 or args.output.exists():
        raise ValueError('Frozen bounded plan and immutable summary required')
    report = dict(status='RUNNING', plan_sha256=sha256(args.plan), runs={},
                  sources=snapshot_sources(args.output.parent / (args.output.stem + '_sources'), [__file__, args.plan]))
    write_json(args.output, report)

    def job(spec):
        config, output = Path(spec['config']), Path(spec['output'])
        cfg = json.loads(config.read_text())
        command = [sys.executable, 'tools/cfp_replay_oof_gru.py', '--config', str(config), '--output', str(output)]
        if not output.exists():
            with output.with_suffix('.log').open('w') as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                               env=dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES='',
                                        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2'))
        result = json.loads((output / 'manifest.json').read_text())
        if result['status'] != 'COMPLETE' or result['config_sha256'] != sha256(config):
            raise ValueError('Preserved replay incomplete or changed')
        for sequence, entry in result['videos'].items():
            if sha256(output / (sequence + '.txt')) != entry['prediction_sha256']:
                raise ValueError('Prediction drift')
        return output.name, dict(family=spec['family'], sequence=spec['sequence'], command=command,
                                 manifest_sha256=sha256(output / 'manifest.json'), replay=result)

    try:
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            futures = [pool.submit(job, spec) for spec in plan['jobs']]
            for future in as_completed(futures):
                name, result = future.result()
                report['runs'][name] = result
                write_json(args.output, report)
                print(name, 'COMPLETE', flush=True)
        report['status'] = 'COMPLETE'
        write_json(args.output, report)
    except Exception as error:
        report.update(status='FAILED', error=repr(error))
        write_json(args.output, report)
        raise


if __name__ == '__main__':
    main()
