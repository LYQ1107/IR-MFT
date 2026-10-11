#!/usr/bin/env python3
"""Execute a frozen ordinary-association campaign with immutable per-run outputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from cfp.provenance import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--assets-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--derived-gt-root', type=Path, required=True)
    parser.add_argument('--trackeval-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan['state'] != 'FROZEN' or plan['protocol_sha256'] != sha256(args.protocol):
        raise ValueError('Campaign must be frozen against the selected protocol')
    if not 1 <= args.workers <= 2:
        raise ValueError('CPU replay campaign permits one or two workers')
    names = [job['run_id'] for job in plan['jobs']]
    if len(set(names)) != len(names) or any(Path(name).name != name for name in names):
        raise ValueError('Run names must be unique single path components')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def run(job):
        out = args.assets_root / job['run_id']
        evaluation = args.assets_root / (job['run_id'] + '_evaluation')
        features = args.assets_root / job['features']
        replay = [sys.executable, 'tools/cfp_replay_association.py', '--detections', str(args.assets_root.parent / 'phase0/causal_v3_dev_detections'),
                  '--features-root', str(features), '--protocol', str(args.protocol), '--partition', 'dev',
                  '--config', 'configs/cfp/association_baseline_v1.json', '--mode', job['mode'],
                  '--appearance-weight', str(job['appearance_weight']), '--output', str(out)]
        evaluate = [sys.executable, 'tools/cfp_evaluate.py', '--data-root', str(args.data_root), '--protocol', str(args.protocol),
                    '--partition', 'dev', '--derived-gt-root', str(args.derived_gt_root), '--trackeval-root', str(args.trackeval_root),
                    '--predictions', str(out), '--output', str(evaluation)]
        if not out.exists():
            with out.with_name(out.name + '.log').open('x') as log:
                subprocess.run(replay, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest = json.loads((out / 'manifest.json').read_text())
        if manifest['status'] != 'COMPLETE' or manifest['protocol_sha256'] != sha256(args.protocol):
            raise ValueError('Existing incomplete/incompatible replay must be diagnosed separately')
        if manifest['mode'] != job['mode'] or manifest['config']['appearance_weight'] != job['appearance_weight']:
            raise ValueError('Existing replay differs from campaign')
        for sequence, video in manifest['videos'].items():
            if video['feature_manifest']['features_sha256'] != sha256(features / (sequence + '.npz')):
                raise ValueError('Existing replay used different features')
            if video['prediction_sha256'] != sha256(out / (sequence + '.txt')):
                raise ValueError('Existing replay predictions changed')
        if not evaluation.exists():
            with evaluation.with_name(evaluation.name + '.log').open('x') as log:
                subprocess.run(evaluate, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((evaluation / 'metrics.json').read_text())
        for sequence, artifact in report['artifacts'].items():
            if artifact['prediction_sha256'] != manifest['videos'][sequence]['prediction_sha256']:
                raise ValueError('Evaluation does not match replay')
        print(job['run_id'], report['metrics']['COMBINED_SEQ']['HOTA'], flush=True)
        return job['run_id'], dict(job=job, metrics=report['metrics'], replay_manifest_sha256=sha256(out / 'manifest.json'),
                                  evaluation_sha256=sha256(evaluation / 'metrics.json'), commands=[replay, evaluate])
    summary = dict(status='RUNNING', plan_sha256=sha256(args.plan), source_sha256=sha256(__file__), runs={})
    write_json(args.output, summary)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for name, result in executor.map(run, plan['jobs']):
            summary['runs'][name] = result
            write_json(args.output, summary)
    summary['status'] = 'COMPLETE'
    write_json(args.output, summary)


if __name__ == '__main__':
    main()
