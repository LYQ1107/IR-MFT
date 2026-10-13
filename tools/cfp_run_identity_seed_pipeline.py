#!/usr/bin/env python3
"""Registered identity-coordinate/producer/module seed replication pipeline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bootstrap-summary', type=Path, required=True)
    p.add_argument('--bootstrap-pid', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Immutable new identity pipeline summary required')
    root = args.output.parent
    stages = [
        ('ACTUAL_STATES', 'tools/cfp_run_identity_state_campaign.py', 'configs/cfp/phase_vi_identity_states_v1.json', 'identity_states_v1.json'),
        ('EARLY_QUERIES', 'tools/cfp_build_oof_query_campaign.py', 'configs/cfp/phase_vi_identity_early_queries_v1.json', 'identity_early_queries_v1.json'),
        ('MATCHED_PREDICTION_FITS', 'tools/cfp_run_residual_relation_campaign.py', 'configs/cfp/phase_vi_identity_residual_training_v1.json', 'identity_residual_training_v1.json'),
        ('WHOLE_VIDEO_REPLAYS', 'tools/cfp_run_oof_combination_extension.py', 'configs/cfp/phase_vi_identity_replays_v1.json', 'identity_replays_v1.json'),
        ('OFFICIAL_EVALUATIONS', 'tools/cfp_evaluate_oof_families.py', 'configs/cfp/phase_vi_identity_replays_v1.json', 'identity_official_v1.json'),
    ]
    sources = [__file__, *[x[1] for x in stages], *[x[2] for x in stages]]
    report = dict(status='WAITING_FOR_OWN_BOOTSTRAP', bootstrap_summary=str(args.bootstrap_summary),
                  bootstrap_pid=args.bootstrap_pid, stages=[], source_hashes=snapshot_sources(root / (args.output.stem + '_sources'), sources),
                  scope='Frozen detector shared; independently frozen RGB1/2 coordinates plus excluded-video ordinary/model seeds1/2. '
                        'Actual accepted histories generated causally after clean GT-supervised bootstrap; no GT repair. '
                        'Corresponding within-seed controls, no scene-independence claim.')
    write_json(args.output, report)
    try:
        while True:
            bootstrap = json.loads(args.bootstrap_summary.read_text())
            if bootstrap['status'] == 'COMPLETE':
                break
            if bootstrap['status'] != 'RUNNING':
                raise ValueError('Bootstrap failed or stopped')
            command = Path(f'/proc/{args.bootstrap_pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
            if str(args.bootstrap_summary) not in command:
                raise ValueError('Specific bootstrap producer is not live')
            time.sleep(10)
        report['bootstrap_summary_sha256'] = sha256(args.bootstrap_summary)
        for stage, tool, plan, output in stages:
            for source, digest in report['source_hashes'].items():
                if sha256(source) != digest:
                    raise ValueError('Registered pipeline source/plan drift: ' + source)
            destination = root / output
            command = [sys.executable, tool, '--plan', plan, '--output', str(destination)]
            if stage == 'OFFICIAL_EVALUATIONS':
                command += ['--prefix', 'identity_seed']
            report['status'] = stage; write_json(args.output, report)
            with destination.with_suffix('.log').open('x') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, PYTHONPATH='.', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2'))
            result = json.loads(destination.read_text())
            if result['status'] != 'COMPLETE' or result['plan_sha256'] != sha256(plan):
                raise ValueError('Incomplete registered pipeline stage')
            report['stages'].append(dict(stage=stage, command=command, output=str(destination), sha256=sha256(destination)))
            write_json(args.output, report)
        report['status'] = 'COMPLETE'; write_json(args.output, report)
    except Exception as error:
        report.update(status='FAILED', error=repr(error)); write_json(args.output, report); raise


if __name__ == '__main__':
    main()
