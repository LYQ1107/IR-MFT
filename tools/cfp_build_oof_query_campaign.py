#!/usr/bin/env python3
"""Bounded immutable candidate indexing for completed video-excluded causal states."""
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
        raise ValueError('Frozen bounded plan and immutable new summary required')
    report = dict(status='RUNNING', plan_sha256=sha256(args.plan), runs={},
                  sources=snapshot_sources(args.output.parent / (args.output.stem + '_sources'), [__file__, args.plan]))
    write_json(args.output, report)

    def job(spec):
        output, source = Path(spec['output']), Path(spec['run'])
        producer = json.loads((source / 'manifest.json').read_text())
        if producer['status'] != 'COMPLETE':
            raise ValueError('Incomplete state producer')
        command = [sys.executable, 'tools/cfp_build_oof_' + ('early_' if spec['view'] == 'early3' else '') +
                   'candidate_queries.py', '--run', str(source), '--output', str(output)]
        if not output.exists():
            with output.with_suffix('.log').open('w') as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                               env=dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES='',
                                        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2'))
        result = json.loads((output / 'manifest.json').read_text())
        if result['status'] != 'COMPLETE' or result['historical_trace_manifest_sha256'] != sha256(source / 'manifest.json'):
            raise ValueError('Preserved query output incomplete or changed source')
        return output.name, dict(path=str(output), manifest_sha256=sha256(output / 'manifest.json'),
                                 producer_excluded_videos=result['producer_excluded_videos'],
                                 videos=result['videos'], command=command)

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
