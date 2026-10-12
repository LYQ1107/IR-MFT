#!/usr/bin/env python3
"""Bounded complete-video replay/official evaluation with immutable reuse."""
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
    p.add_argument('--replay-driver', default='tools/cfp_replay_candidate.py', choices=[
        'tools/cfp_replay_candidate.py', 'tools/cfp_replay_risk_memory.py',
        'tools/cfp_replay_risk_memory_ablation.py', 'tools/cfp_replay_hypothesis_stable.py'])
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan['state'] != 'FROZEN' or args.output.exists() or not 1 <= plan['workers'] <= 2:
        raise ValueError('Bounded frozen plan/new summary required')
    report = dict(status='RUNNING', plan_sha256=sha256(args.plan), runs={},
        source_snapshots=snapshot_sources(args.output.parent / (args.output.stem + '_sources'), [__file__, args.plan]))
    write_json(args.output, report)
    def execute(command, log):
        with log.open('w') as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1'))
    def job(spec):
        config, output = Path(spec['config']), Path(spec['output'])
        cfg = json.loads(config.read_text())
        command = [sys.executable, args.replay_driver, '--config', str(config), '--output', str(output)]
        if not output.exists():
            execute(command, output.with_suffix('.log'))
        run = json.loads((output / 'manifest.json').read_text())
        if run['status'] != 'COMPLETE' or run['config_sha256'] != sha256(config):
            raise ValueError('Preserved run incomplete or config changed: ' + str(output))
        for seq, entry in run['videos'].items():
            if sha256(output / (seq + '.txt')) != entry['prediction_sha256']:
                raise ValueError('Prediction drift')
        evaluation = output.with_name(output.name + '_evaluation')
        evaluate = [sys.executable, 'tools/cfp_evaluate.py', '--data-root', '/data3/liuyeqiang/datasets/mft26-ir-causal-v1',
            '--derived-gt-root', '/data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt', '--protocol',
            'configs/cfp/protocol_v2.json', '--partition', cfg['partition'], '--predictions', str(output),
            '--output', str(evaluation), '--trackeval-root', '/data3/liuyeqiang/CFP-FishMOT_references/TrackEval']
        if not (evaluation / 'metrics.json').exists():
            execute(evaluate, evaluation.with_suffix('.log'))
        metrics = json.loads((evaluation / 'metrics.json').read_text())
        if set(run['videos']) != set(metrics['sequences']):
            raise ValueError('Complete protocol video coverage required')
        for seq, entry in run['videos'].items():
            if metrics['artifacts'][seq]['prediction_sha256'] != entry['prediction_sha256']:
                raise ValueError('Evaluation drift')
        return output.name, dict(replay=run, metrics=metrics['metrics'], replay_command=command,
            evaluation_command=evaluate, replay_manifest_sha256=sha256(output / 'manifest.json'),
            official_metric_sha256=metrics['metric_sha256'], evaluation_sha256=sha256(evaluation / 'metrics.json'))
    try:
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            futures = [pool.submit(job, spec) for spec in plan['jobs']]
            for future in as_completed(futures):
                name, entry = future.result()
                report['runs'][name] = entry
                write_json(args.output, report)
                print(name, 'official HOTA', entry['metrics']['COMBINED_SEQ']['HOTA'], flush=True)
        report['status'] = 'COMPLETE'
        write_json(args.output, report)
    except Exception as error:
        report.update(status='FAILED', error=repr(error))
        write_json(args.output, report)
        raise


if __name__ == '__main__':
    main()
