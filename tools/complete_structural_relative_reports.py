#!/usr/bin/env python3
"""Export only whole finished stages while the fixed-seed research runs."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def wait_complete(path, stage=None):
    while True:
        state = json.loads(path.read_text())
        if any(job['status'] == 'FAILED' for job in state['jobs'].values()):
            raise RuntimeError(f'Research failure retained at {path}; no partial report')
        status = state['status'] if stage is None else state['stages'][stage]['status']
        if status == 'COMPLETE':
            return
        if 'FAILED' in state['status']:
            raise RuntimeError(f'Research dependency failed at {path}; no report')
        time.sleep(5)


def run(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    wait_complete(work/'structural_relative_fit_state.json')
    run('tools/export_structural_relative_training_ledger.py')
    state = work/'structural_relative_dev_state.json'
    stages = [
        ('full_controls', 'STRUCTURAL_RELATIVE_MAIN_DEV', 'All25 matched structural-relative controls',
         'All3cues x4snapshots xraw/priority plus original RGB. Fixedseed0,145600params, '
         '24passes over1152Trainframes/6912updates. Bothfullsource8/32/full59proofs '
         'passed beforeaccuracy. Sharedproducer timing is not per-method end-to-end FPS.', []),
        ('module_removals', 'STRUCTURAL_RELATIVE_REMOVALS_DEV', 'All24 structural-relative module removals',
         'no_structure/no_cycle/no_centering each4snapshots xraw/priority. Same seed0, '
         'capacity and24E/6912budget. Cyclecheck is nested in structuralvalidation, '
         'not a third independent innovation. Allcurves and paired epoch/policy comparison retained.',
         ['--include-removals']),
        ('IR_interventions', 'STRUCTURAL_RELATIVE_INTERVENTIONS_DEV', 'All10 actual-IR intervention controls',
         'Quarter-image IRroll BEFORE encoding/descriptor computation at all4IRsnapshots '
         'x2policies, plus unavailableIR at epoch24. The internal quarter-FPNnull '
         'reference is a separate operation. No newseed/Test or noveltyclaim.',
         ['--include-removals', '--include-interventions'])]
    for stage, tag, title, note, flags in stages:
        wait_complete(state, stage)
        run('tools/export_perceptual_dev_family.py', '--stage', 'structural_relative_dev',
            '--config-dir', 'configs/perceptual_structural_relative_dev_20261123',
            '--completed-stage', stage, '--tag', tag, '--title', title, '--note', note)
        run('tools/summarize_structural_relative.py', *flags)


if __name__ == '__main__':
    main()
