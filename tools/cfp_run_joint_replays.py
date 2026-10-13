#!/usr/bin/env python3
"""Verify disabled parity, then execute matched complete-video joint controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

from cfp.provenance import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan['state'] != 'FROZEN' or args.output.exists() or not 1 <= plan['workers'] <= 2:
        raise ValueError('New frozen bounded full-video campaign required')
    report = dict(status='RUNNING', plan_sha256=sha256(args.plan), runs={})
    write_json(args.output, report)

    def execute(command, log):
        with log.open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True,
                env=dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1'))

    def job(spec):
        config, output = Path(spec['config']), Path(spec['output'])
        command = [sys.executable, 'tools/cfp_replay_joint_identity.py', '--config', str(config), '--output', str(output)]
        if not output.exists():
            execute(command, output.with_suffix('.log'))
        replay = json.loads((output / 'manifest.json').read_text())
        if replay['status'] != 'COMPLETE' or replay['config_sha256'] != sha256(config):
            raise ValueError('Preserved replay incomplete or config differs')
        for video, entry in replay['videos'].items():
            if sha256(output / (video + '.txt')) != entry['prediction_sha256']:
                raise ValueError('Prediction drift')
        evaluation = output.with_name(output.name + '_evaluation')
        evaluate = [sys.executable, 'tools/cfp_evaluate.py', '--data-root', '/data3/liuyeqiang/datasets/mft26-ir-causal-v1',
            '--derived-gt-root', '/data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt', '--protocol', plan['evaluation_protocol'],
            '--partition', 'train', '--predictions', str(output), '--output', str(evaluation),
            '--trackeval-root', '/data3/liuyeqiang/CFP-FishMOT_references/TrackEval']
        if not (evaluation / 'metrics.json').exists():
            execute(evaluate, evaluation.with_suffix('.log'))
        metrics = json.loads((evaluation / 'metrics.json').read_text())
        if set(metrics['sequences']) != set(replay['videos']):
            raise ValueError('Evaluation/replay video coverage differs')
        for video, entry in replay['videos'].items():
            if metrics['artifacts'][video]['prediction_sha256'] != entry['prediction_sha256']:
                raise ValueError('Evaluation predictions differ')
        return output.name, dict(replay_manifest_sha256=sha256(output / 'manifest.json'),
            evaluation_sha256=sha256(evaluation / 'metrics.json'), replay_command=command, evaluation_command=evaluate,
            videos=replay['videos'], metrics=metrics['metrics'], scope=replay['scope'])

    try:
        name, result = job(plan['baseline'])
        if not all(v['baseline_byte_equal'] for v in result['videos'].values()):
            raise ValueError('Complete disabled byte parity required before learned runs')
        report['runs'][name] = result
        write_json(args.output, report)
        print(name, 'baseline all-frame byte parity verified', flush=True)
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            for future in as_completed([pool.submit(job, spec) for spec in plan['jobs']]):
                name, result = future.result()
                report['runs'][name] = result
                write_json(args.output, report)
                print(name, {v: {k: values[k] for k in ('HOTA', 'IDF1', 'IDSW')} for v, values in result['metrics'].items()}, flush=True)
        report['status'] = 'COMPLETE'
        write_json(args.output, report)
    except Exception as error:
        report.update(status='FAILED', error=repr(error))
        write_json(args.output, report)
        raise


if __name__ == '__main__':
    main()
