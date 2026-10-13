#!/usr/bin/env python3
"""Immutable bounded actual-state retrieval/oracle diagnostics, never online."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    args = p.parse_args(); plan = json.loads(args.plan.read_text())
    if plan['state'] != 'FROZEN' or not 1 <= plan['workers'] <= 2 or args.output.exists():
        raise ValueError('Frozen bounded plan/new summary required')
    report = dict(status='RUNNING', plan_sha256=sha256(args.plan), runs={},
                  sources=snapshot_sources(args.output.parent / (args.output.stem + '_sources'), [__file__, args.plan]))
    write_json(args.output, report)

    def job(spec):
        output = Path(spec['output']); command = [sys.executable, *spec['arguments']]
        if not output.exists():
            with output.with_suffix('.log').open('w') as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES='',
                             OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2'))
        result = json.loads(output.read_text())
        if result['status'] != 'COMPLETE' or list(result['videos']) != [spec['sequence']]:
            raise ValueError('Exactly one full outer-video query bank required')
        return output.stem, dict(sequence=spec['sequence'], view=spec['view'], mode=spec['mode'],
                                 output_sha256=sha256(output), command=command, result=result)

    try:
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            for future in as_completed([pool.submit(job, s) for s in plan['jobs']]):
                name, result = future.result(); report['runs'][name] = result
                write_json(args.output, report); print(name, 'COMPLETE', flush=True)
        report['status'] = 'COMPLETE'; write_json(args.output, report)
    except Exception as error:
        report.update(status='FAILED', error=repr(error)); write_json(args.output, report); raise


if __name__ == '__main__':
    main()
