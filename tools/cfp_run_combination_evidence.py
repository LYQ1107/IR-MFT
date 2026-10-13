#!/usr/bin/env python3
"""Complete retrospective branch evidence and exact official event comparisons."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from cfp.explanation_diagnosis import explanations
from cfp.paired_explanation_diagnosis import paired_cases
from cfp.provenance import sha256, snapshot_sources, write_json


def live(pid, identifier):
    command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    if str(identifier) not in command:
        raise ValueError('Specific producer is not live')


def execute(command, output):
    with output.with_suffix('.log').open('x') as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                       env=dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1'))


def pairs(baseline, variant, sequence, labels, frames):
    results = dict(N00=0, N01=0, N10=0, N11=0, unknown=0, baseline_only=0, variant_only=0,
                   N01_episodes=0, N10_episodes=0, examples=[])
    previous = {}
    seen = 0
    with (baseline / (sequence + '.hypotheses.jsonl')).open() as left, (variant / (sequence + '.hypotheses.jsonl')).open() as right:
        for a, b in zip(left, right):
            old, new = json.loads(a), json.loads(b)
            seen += 1
            if old['frame'] != seen or new['frame'] != seen:
                raise ValueError('Full current-frame pairing required')
            original = explanations(old, labels, frames)[old['winner']]
            changed = explanations(new, labels, frames)[new['winner']]
            cases, absent, born = paired_cases(original, changed)
            results['baseline_only'] += absent; results['variant_only'] += born
            for birth, category in cases.items():
                results[category] += 1
                if category in ('N01', 'N10'):
                    results[category + '_episodes'] += int(previous.get(birth) != (seen - 1, category))
                    if len(results['examples']) < 12:
                        results['examples'].append(dict(frame=seen, birth=list(birth), category=category))
                previous[birth] = (seen, category)
        if left.readline() or right.readline():
            raise ValueError('Mismatched video lengths')
    results.update(frames=seen, net_correction=results['N01'] - results['N10'])
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--matrix-pid', type=int, required=True)
    p.add_argument('--official-workflow', type=Path, required=True)
    p.add_argument('--official-workflow-pid', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Immutable new evidence campaign required')
    args.output.mkdir()
    plan = json.loads(args.plan.read_text())
    report = dict(status='RUNNING_BRANCH_DIAGNOSTICS', plan_sha256=sha256(args.plan), diagnostics={}, paired={}, official_events={},
        sources=snapshot_sources(args.output, [__file__, args.plan, 'cfp/explanation_diagnosis.py', 'cfp/paired_explanation_diagnosis.py', 'tools/cfp_diagnose_hypothesis_explanations.py', 'tools/cfp_trace_clear_events.py']),
        scope='Birth-reference frame/track comparisons on common support, unknown excluded, not IDSW. '
              'Missing/newborn explanations are separate. Exact upstream CLEAR events independently reproduce official switch totals. '
              'All GT evidence is retrospective and never enters causal replay.')
    manifest_path = args.output / 'manifest.json'
    write_json(manifest_path, report)
    # Original A/B controls have no hypothesis trace; all six new causal families do.
    jobs = [j for j in plan['jobs'] if j['family'] not in ('A', 'B')]
    try:
        while True:
            matrix = json.loads(args.matrix.read_text())
            for spec in jobs:
                key = spec['family'] + '_' + spec['sequence']
                run = Path(spec['output'])
                path = run / 'manifest.json'
                if key in report['diagnostics'] or not path.exists():
                    continue
                replay = json.loads(path.read_text())
                if replay['status'] != 'COMPLETE':
                    continue
                output = args.output / (key + '_birth.json')
                command = [sys.executable, 'tools/cfp_diagnose_hypothesis_explanations.py', '--replay', str(run), '--output', str(output)]
                execute(command, output)
                result = json.loads(output.read_text())
                report['diagnostics'][key] = dict(command=command, sha256=sha256(output), result=result)
                write_json(manifest_path, report)
                print(key, 'birth complete', flush=True)
            if matrix['status'] == 'COMPLETE':
                if len(report['diagnostics']) != len(jobs):
                    raise ValueError('Missing complete causal explanation videos')
                break
            if matrix['status'] != 'RUNNING':
                raise ValueError('Replay matrix failed')
            live(args.matrix_pid, args.matrix)
            time.sleep(10)
        physical_root = Path('/data3/liuyeqiang/IR-MFT_assets/cfp/phase5/recovery_train_queries_v1')
        physical = json.loads((physical_root / 'manifest.json').read_text())
        for sequence in sorted({j['sequence'] for j in jobs}):
            pp = physical_root / (sequence + '.npz')
            if sha256(pp) != physical['videos'][sequence]['sha256']:
                raise ValueError('Retrospective physical-bank drift')
            with np.load(pp, allow_pickle=False) as bank:
                labels, frames = bank['physical_labels'], bank['frame_of_proposal']
            baseline = next(Path(j['output']) for j in jobs if j['family'] == 'B0' and j['sequence'] == sequence)
            for spec in [j for j in jobs if j['sequence'] == sequence and j['family'] != 'B0']:
                result = pairs(baseline, Path(spec['output']), sequence, labels, frames)
                expected = report['diagnostics']['B0_' + sequence]['result']['videos'][sequence]['counts']['frames']
                if result['frames'] != expected:
                    raise ValueError('Partial paired explanation evidence')
                report['paired'][spec['family'] + '_' + sequence] = result
        report['status'] = 'WAITING_FOR_OFFICIAL_EVALUATIONS'; write_json(manifest_path, report)
        while True:
            workflow = json.loads(args.official_workflow.read_text())
            if workflow['status'] == 'COMPLETE':
                break
            if workflow['status'] == 'FAILED':
                raise ValueError('Official evaluation workflow failed')
            live(args.official_workflow_pid, args.official_workflow)
            time.sleep(10)
        official_path = Path(workflow['results'])
        if sha256(official_path) != workflow['results_sha256']:
            raise ValueError('Official summary drift')
        official = json.loads(official_path.read_text())
        report['official_summary_sha256'] = sha256(official_path)
        report['status'] = 'OFFICIAL_CLEAR_TRACES'; write_json(manifest_path, report)
        for family in ('B0', 'C', 'BC', 'ABC'):
            entry = official['families'][family]
            command = entry['evaluation_command']
            evaluation = Path(command[command.index('--output') + 1])
            if sha256(evaluation / 'metrics.json') != entry['evaluation_sha256']:
                raise ValueError('Official metric drift')
            output = args.output / (family + '_official_clear.json')
            command = [sys.executable, 'tools/cfp_trace_clear_events.py', '--evaluation-root', str(evaluation),
                       '--derived-gt-root', '/data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt',
                       '--trackeval-root', '/data3/liuyeqiang/CFP-FishMOT_references/TrackEval', '--output', str(output)]
            execute(command, output)
            result = json.loads(output.read_text())
            report['official_events'][family] = dict(command=command, sha256=sha256(output), result=result)
            write_json(manifest_path, report)
            print(family, 'official CLEAR complete', flush=True)
        report['status'] = 'COMPLETE'; write_json(manifest_path, report)
    except Exception as error:
        report.update(status='FAILED', error=repr(error)); write_json(manifest_path, report); raise


if __name__ == '__main__':
    main()
