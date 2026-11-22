#!/usr/bin/env python3
"""Register the final matched Dev stage after the common-budget decision."""
import json
import os
from pathlib import Path
import subprocess
import sys

from tools.run_pseudo_ir_comparison import dump, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    work = Path(json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())['work_root'])
    decision_path = work/'native_query_common_e30_decision.json'
    decision = json.loads(decision_path.read_text())
    extension = json.loads((work/'native_query_common_e30_extension.json').read_text())
    if decision['status'] != 'EXTEND_ALL_MATCHED_HEADS' or extension['status'] != 'EXECUTED_COMMON_EXTENSION':
        raise ValueError('The final common extension must already be justified and scheduled')
    registration = work/'native_query_final_stage_registration.json'
    if registration.exists():
        raise ValueError('Final Dev stage already registered')
    jobs = []
    for family, fit in [('robust', 'native_robust_query'), ('consensus', 'native_robust_query'),
                        ('ordered', 'native_query'), ('consensus_clean', 'native_query')]:
        source = ROOT/f'configs/perceptual_{family}30_dev_20261123'
        destination = ROOT/f'configs/perceptual_{family}36_dev_20261123'
        destination.mkdir(parents=True, exist_ok=False)
        configs = []
        for path in sorted(source.glob('*.json')):
            config = json.loads(path.read_text())
            config['id'] = config['id'].replace('_e30', '_e36')
            config['native_query_checkpoint'] = config['native_query_checkpoint'].replace('/epoch30.pt', '/epoch36.pt')
            config['registration'] = ('Final common budget36, forced by the registered complete Dev lookback rule. '
                                      'All clean and mismatch-robust RGB/IR fits receive the same budget. '
                                      'Preserve every prior checkpoint; no Test selection or extension beyond36.')
            output = destination/(config['id']+'.json')
            dump(output, config)
            configs.append(dict(path=str(output), sha256=sha256(output)))
        if len(configs) != 12:
            raise ValueError('All matched frontend/cue/seed combinations are required')
        label = family+'36_dev'
        command = [sys.executable, '-u', 'tools/launch_native_query_dev.py', '--label', label,
                   '--config-dir', str(destination), '--fit-state', fit+'_fit_e36_state.json']
        jobs.append(dict(label=label, command=command, configs=configs))
    state = dict(status='REGISTERED_BEFORE_FINAL_DEV_ACCURACY', decision_sha256=sha256(decision_path),
                 maximum_epochs=36, Test_selection=False, families=jobs)
    dump(registration, state)
    for job in jobs:
        log = work/(job['label']+'.controller.log')
        with log.open('ab', buffering=0) as handle:
            process = subprocess.Popen(job['command'], cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        job.update(pid=process.pid, log=str(log))
    state['status'] = 'SCHEDULED'
    dump(registration, state)
    print(json.dumps(dict(status=state['status'], families=4, Dev_runs=48)))


if __name__ == '__main__':
    main()
