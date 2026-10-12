#!/usr/bin/env python3
"""Run a frozen, bounded joint/leave-component-out diagnosis on canonical dev."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys

from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--assets-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan['state'] != 'FROZEN' or len({j['run_id'] for j in plan['jobs']}) != len(plan['jobs']):
        raise ValueError('Frozen unique joint campaign required')
    for path, expected in plan['input_sha256'].items():
        if sha256(path) != expected:
            raise ValueError('Frozen joint input changed: ' + path)
    if args.output.exists():
        raise FileExistsError('Joint campaign summary is immutable; resume individual outputs with a separate summary')
    summary = dict(status='RUNNING', plan_sha256=sha256(args.plan),
                   source_snapshots=snapshot_sources(args.output.with_suffix(''), [__file__, args.plan]), runs={})
    write_json(args.output, summary)

    def run(job):
        if shutil.disk_usage(args.assets_root).free < plan['minimum_free_gib'] * 2**30:
            raise OSError('Joint campaign storage reserve reached')
        name = job['run_id']
        if Path(name).name != name:
            raise ValueError('Unsafe joint run id')
        out = args.assets_root / name
        evaluation = out.with_name(name + '_evaluation')
        replay = [sys.executable, '-u', 'tools/cfp_replay_predictor.py', *plan['common_replay_args'], *job['args'], '--output', str(out)]
        evaluate = [sys.executable, '-u', 'tools/cfp_evaluate.py', *plan['common_evaluate_args'], '--predictions', str(out), '--output', str(evaluation)]
        if not out.exists():
            with out.with_name(name + '.log').open('x') as stream:
                subprocess.run(replay, stdout=stream, stderr=subprocess.STDOUT, check=True)
        manifest = json.loads((out / 'manifest.json').read_text())
        if manifest['status'] != 'COMPLETE':
            raise ValueError('Joint replay is incomplete')
        if not evaluation.exists():
            with evaluation.with_name(evaluation.name + '.log').open('x') as stream:
                subprocess.run(evaluate, stdout=stream, stderr=subprocess.STDOUT, check=True)
        metrics = json.loads((evaluation / 'metrics.json').read_text())
        for sequence, info in manifest['videos'].items():
            if sha256(out / (sequence + '.txt')) != info['prediction_sha256'] or metrics['artifacts'][sequence]['prediction_sha256'] != info['prediction_sha256']:
                raise ValueError('Joint output/evaluation differ')
        if job.get('baseline_reference'):
            for sequence in manifest['videos']:
                if (out / (sequence + '.txt')).read_bytes() != (Path(job['baseline_reference']) / (sequence + '.txt')).read_bytes():
                    raise ValueError('Disabled joint toggle changed the baseline')
        entry = dict(commands=[replay, evaluate], hypothesis=job['hypothesis'], metrics=metrics['metrics'],
                     manifest_sha256=sha256(out / 'manifest.json'), evaluation_sha256=sha256(evaluation / 'metrics.json'),
                     video_decisions={s: {k: v[k] for k in ('joint_counts', 'utility_counts', 'write_counts') if k in v} for s, v in manifest['videos'].items()})
        print(name, {k: entry['metrics']['COMBINED_SEQ'][k] for k in ('HOTA', 'AssA', 'IDF1', 'IDSW')}, flush=True)
        return name, entry

    with ThreadPoolExecutor(2) as pool:
        for name, result in pool.map(run, plan['jobs']):
            summary['runs'][name] = result
            write_json(args.output, summary)
    summary['status'] = 'COMPLETE'
    write_json(args.output, summary)


if __name__ == '__main__':
    main()
