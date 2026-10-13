#!/usr/bin/env python3
"""Complete-video official evaluation for immutable outer-fold family plans."""
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
    p.add_argument('--prefix', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); plan = json.loads(args.plan.read_text())
    if plan['state'] != 'FROZEN' or args.output.exists() or '/' in args.prefix:
        raise ValueError('Frozen complete plan/new summary/simple aggregate prefix required')
    families = sorted({j['family'] for j in plan['jobs']})
    report = dict(status='RUNNING', families={}, plan_sha256=sha256(args.plan),
                  sources=snapshot_sources(args.output.parent / (args.output.stem + '_sources'), [__file__, args.plan]))
    write_json(args.output, report)

    def execute(command, log):
        with log.open('w') as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                           env=dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1'))

    def job(family):
        aggregate = args.output.parent / (args.prefix + '_' + family + '_outer_s0_v1')
        command = [sys.executable, 'tools/cfp_aggregate_oof_predictions.py', '--plan', str(args.plan),
                   '--family', family, '--output', str(aggregate)]
        if not aggregate.exists():
            execute(command, aggregate.with_suffix('.log'))
        run = json.loads((aggregate / 'manifest.json').read_text())
        if run['status'] != 'COMPLETE' or run['plan_sha256'] != sha256(args.plan):
            raise ValueError('Incomplete or changed aggregate')
        evaluation = aggregate.with_name(aggregate.name + '_evaluation')
        evaluate = [sys.executable, 'tools/cfp_evaluate.py', '--data-root', '/data3/liuyeqiang/datasets/mft26-ir-causal-v1',
                    '--derived-gt-root', '/data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt',
                    '--protocol', 'configs/cfp/protocol_v2.json', '--partition', 'train',
                    '--predictions', str(aggregate), '--output', str(evaluation),
                    '--trackeval-root', '/data3/liuyeqiang/CFP-FishMOT_references/TrackEval']
        if not (evaluation / 'metrics.json').exists():
            execute(evaluate, evaluation.with_suffix('.log'))
        result = json.loads((evaluation / 'metrics.json').read_text())
        if set(result['sequences']) != set(run['videos']):
            raise ValueError('Complete matching protocol coverage required')
        for sequence, entry in run['videos'].items():
            if sha256(aggregate / (sequence + '.txt')) != entry['prediction_sha256'] or result['artifacts'][sequence]['prediction_sha256'] != entry['prediction_sha256']:
                raise ValueError('Prediction/evaluation drift')
        return family, dict(metrics=result['metrics'], aggregate_manifest_sha256=sha256(aggregate / 'manifest.json'),
                            evaluation_sha256=sha256(evaluation / 'metrics.json'), metric_sha256=result['metric_sha256'],
                            aggregate_command=command, evaluation_command=evaluate)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(job, f) for f in families]):
                family, result = future.result(); report['families'][family] = result
                write_json(args.output, report); print(family, result['metrics']['COMBINED_SEQ']['HOTA'], flush=True)
        report['status'] = 'COMPLETE'; write_json(args.output, report)
    except Exception as error:
        report.update(status='FAILED', error=repr(error)); write_json(args.output, report); raise


if __name__ == '__main__':
    main()
