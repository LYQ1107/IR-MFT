#!/usr/bin/env python3
"""Finish an already registered training campaign with immutable replay/evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from cfp.provenance import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-summary', type=Path, required=True)
    parser.add_argument('--replay-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    if args.output.exists() or '/' in args.prefix:
        raise ValueError('Immutable new workflow output and simple prefix required')
    plan_hash = sha256(args.replay_plan)
    report = dict(status='WAITING_FOR_TRAINING', training_summary=str(args.training_summary),
                  replay_plan=str(args.replay_plan), replay_plan_sha256=plan_hash, stages=[])
    write_json(args.output, report)
    try:
        while True:
            training = json.loads(args.training_summary.read_text())
            if training['status'] == 'COMPLETE':
                break
            if training['status'] != 'RUNNING':
                raise ValueError('Training did not complete: ' + training['status'])
            time.sleep(10)
        if sha256(args.replay_plan) != plan_hash:
            raise ValueError('Registered replay plan drift')
        report['training_summary_sha256'] = sha256(args.training_summary)
        jobs = json.loads(args.replay_plan.read_text())['jobs']
        training_outputs = set(training['runs'])
        for job in jobs:
            config = json.loads(Path(job['config']).read_text())
            for model in config['student_models'].values():
                if Path(model).name not in training_outputs:
                    raise ValueError('Replay student outside completed training campaign')
        for stage, tool, output in [
            ('REPLAY', 'tools/cfp_run_oof_candidate_relation_campaign.py',
             args.output.with_name(args.prefix + '_replays_v1.json')),
            ('OFFICIAL_EVALUATION', 'tools/cfp_evaluate_oof_families.py',
             args.output.with_name(args.prefix + '_official_v1.json')),
        ]:
            command = [sys.executable, tool, '--plan', str(args.replay_plan), '--output', str(output)]
            if stage == 'OFFICIAL_EVALUATION':
                command += ['--prefix', args.prefix]
            report['status'] = stage
            write_json(args.output, report)
            with output.with_suffix('.log').open('x') as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                               env=dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES='',
                                        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2'))
            result = json.loads(output.read_text())
            if result['status'] != 'COMPLETE' or result['plan_sha256'] != plan_hash:
                raise ValueError('Incomplete or drifted stage: ' + stage)
            report['stages'].append(dict(stage=stage, command=command, output=str(output), sha256=sha256(output)))
        report['status'] = 'COMPLETE'
        write_json(args.output, report)
    except Exception as error:
        report.update(status='FAILED', error=repr(error))
        write_json(args.output, report)
        raise


if __name__ == '__main__':
    main()
