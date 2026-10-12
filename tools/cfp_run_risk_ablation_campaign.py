#!/usr/bin/env python3
"""Bounded CPU campaign with immutable results and guarded completed-run reuse."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

from cfp.provenance import sha256, write_json, snapshot_sources


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan['state'] != 'FROZEN' or not 1 <= plan['max_workers'] <= 2 or args.output.exists():
        raise ValueError('New result path and bounded frozen plan required')
    record = dict(status='RUNNING', plan_sha256=sha256(args.plan), runs={},
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_snapshots=snapshot_sources(args.output.parent / (args.output.stem + '_sources'), [__file__, args.plan]),
        scope=plan['scope'])
    write_json(args.output, record)

    def job(spec):
        output, config = Path(spec['output']), Path(spec['config'])
        command = [sys.executable, 'tools/cfp_train_risk_ablation.py', '--config', str(config), '--output', str(output)]
        if not output.exists():
            with output.with_suffix('.log').open('w') as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1'))
        report = json.loads((output / 'manifest.json').read_text())
        if report['status'] != 'COMPLETE' or report['guard']['config_sha256'] != sha256(config):
            raise ValueError('Preserved incomplete or mismatched run: ' + str(output))
        if sha256(output / 'last.pt') != report['checkpoint_sha256'] or sha256(output / 'inference.pt') != report['inference_sha256']:
            raise ValueError('Completed checkpoint hash drift')
        for key, digest in report['guard']['sources'].items():
            if sha256(key) != digest:
                raise ValueError('Completed source changed: ' + key)
        return output.name, dict(command=command, manifest_sha256=sha256(output / 'manifest.json'), **report)

    try:
        with ThreadPoolExecutor(max_workers=plan['max_workers']) as pool:
            futures = [pool.submit(job, spec) for spec in plan['jobs']]
            for future in as_completed(futures):
                name, result = future.result()
                record['runs'][name] = result
                write_json(args.output, record)
                print(len(record['runs']), '/', len(plan['jobs']), name,
                    'accepted-AUC', result['validation_calibrated']['accepted_edges']['auc'], flush=True)
        record['status'] = 'COMPLETE'
        write_json(args.output, record)
    except Exception as error:
        record.update(status='FAILED', error=repr(error))
        write_json(args.output, record)
        raise


if __name__ == '__main__':
    main()
