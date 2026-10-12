#!/usr/bin/env python3
"""Execute fixed train-complete recovery/control replay jobs with official metrics."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError('Bounded CPU campaign')
    plan = json.loads(args.plan.read_text())
    if plan['state'] != 'FROZEN':
        raise ValueError('Frozen plan required')
    root = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    record = dict(status='RUNNING', plan_sha256=sha256(args.plan), runs={},
                  source_snapshots=snapshot_sources(args.output.parent / (args.output.stem + '_sources'), [__file__, args.plan]))
    if args.output.exists() and json.loads(args.output.read_text())['plan_sha256'] != record['plan_sha256']:
        raise ValueError('Resume plan drift')
    write_json(args.output, record)

    def execute(command, log):
        if shutil.disk_usage(root).free < 40 * 2**30:
            raise OSError('40GiB reserve')
        with log.open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPATH='.', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='0'))
        if result.returncode:
            raise RuntimeError('Preserved failed job log: ' + str(log))

    def job(spec):
        output = root / 'phase5' / spec['name']
        command = [sys.executable, 'tools/cfp_replay_recovery.py', '--kind', spec['kind'], '--seed', str(spec['seed']), '--weight', str(spec['weight']), '--output', str(output)]
        if spec.get('checkpoint'):
            command += ['--checkpoint', spec['checkpoint']]
        if spec.get('record_decisions', False):
            command += ['--record-decisions']
        if spec.get('read_view'):
            command += ['--read-view', spec['read_view']]
        if not output.exists():
            execute(command, output.with_name(output.name + '.log'))
        run = json.loads((output / 'manifest.json').read_text())
        if run['status'] != 'COMPLETE' or run['kind'] != spec['kind'] or run['seed'] != spec['seed'] or run['appearance_weight'] != spec['weight']:
            raise ValueError('Incomplete or incompatible replay; preserve/repair before retry: ' + spec['name'])
        if spec.get('checkpoint') and run['checkpoint_sha256'] != sha256(spec['checkpoint']):
            raise ValueError('Model changed')
        evaluation = output.with_name(output.name + '_evaluation')
        eval_command = [sys.executable, 'tools/cfp_evaluate.py', '--data-root', '/data3/liuyeqiang/datasets/mft26-ir-causal-v1', '--derived-gt-root', str(root / 'protocol_v2/gt'),
                        '--protocol', 'configs/cfp/protocol_v2.json', '--partition', 'dev', '--predictions', str(output), '--output', str(evaluation), '--trackeval-root', '/data3/liuyeqiang/CFP-FishMOT_references/TrackEval']
        if not (evaluation / 'metrics.json').exists():
            execute(eval_command, evaluation.with_name(evaluation.name + '.log'))
        metrics = json.loads((evaluation / 'metrics.json').read_text())
        for seq in ('BT-005', 'SN-013'):
            if sha256(output / (seq + '.txt')) != run['videos'][seq]['prediction_sha256'] or sha256(output / (seq + '.txt')) != metrics['artifacts'][seq]['prediction_sha256']:
                raise ValueError('Replay/evaluation bytes changed')
        return spec['name'], dict(spec=spec, replay_command=command, evaluate_command=eval_command,
                                run_manifest_sha256=sha256(output / 'manifest.json'), evaluation_sha256=sha256(evaluation / 'metrics.json'), metrics=metrics['metrics'])
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(job, spec) for spec in plan['jobs']]
            for future in as_completed(futures):
                name, entry = future.result()
                record['runs'][name] = entry
                write_json(args.output, record)
                print('Complete', len(record['runs']), '/', len(plan['jobs']), name, 'HOTA', entry['metrics']['COMBINED_SEQ']['HOTA'], flush=True)
        record['status'] = 'COMPLETE'
        write_json(args.output, record)
    except Exception as error:
        record.update(status='FAILED', error=str(error))
        write_json(args.output, record)
        raise


if __name__ == '__main__':
    main()
