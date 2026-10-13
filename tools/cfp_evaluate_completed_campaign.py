#!/usr/bin/env python3
"""Evaluate a specific live replay campaign once all complete videos are verified."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from cfp.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign-summary', type=Path, required=True)
    p.add_argument('--campaign-pid', type=int, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--prefix', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or '/' in args.prefix:
        raise ValueError('Immutable new evaluation workflow required')
    digest = sha256(args.plan)
    report = dict(status='WAITING_FOR_LIVE_REPLAYS', campaign=str(args.campaign_summary),
                  campaign_pid=args.campaign_pid, plan_sha256=digest)
    write_json(args.output, report)
    try:
        while True:
            campaign = json.loads(args.campaign_summary.read_text())
            if campaign['status'] == 'COMPLETE':
                break
            if campaign['status'] != 'RUNNING':
                raise ValueError('Replay campaign failed or stopped')
            command = Path(f'/proc/{args.campaign_pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
            if str(args.campaign_summary) not in command:
                raise ValueError('The specific replay producer is no longer live')
            time.sleep(10)
        if campaign['plan_sha256'] != digest or sha256(args.plan) != digest:
            raise ValueError('Registered campaign/plan drift')
        result_path = args.output.with_name(args.prefix + '_official_v1.json')
        command = [sys.executable, 'tools/cfp_evaluate_oof_families.py', '--plan', str(args.plan),
                   '--prefix', args.prefix, '--output', str(result_path)]
        report.update(status='OFFICIAL_EVALUATION', command=command, campaign_sha256=sha256(args.campaign_summary))
        write_json(args.output, report)
        with result_path.with_suffix('.log').open('x') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, PYTHONPATH='.', CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1'))
        result = json.loads(result_path.read_text())
        if result['status'] != 'COMPLETE' or result['plan_sha256'] != digest:
            raise ValueError('Incomplete official evaluation')
        report.update(status='COMPLETE', results=str(result_path), results_sha256=sha256(result_path))
        write_json(args.output, report)
    except Exception as error:
        report.update(status='FAILED', error=repr(error))
        write_json(args.output, report)
        raise


if __name__ == '__main__':
    main()
