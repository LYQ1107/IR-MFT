#!/usr/bin/env python3
"""Read the complete32 new component cases together, retaining all references."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    path=Path(plan['work_root'])/'structural_factorial_dev_state.json'
    while True:
        state=json.loads(path.read_text())
        if 'FAIL'in state['status']:raise RuntimeError('Retain failure; no partial component-family accuracy')
        if state['status']=='COMPLETE':break
        time.sleep(5)
    subprocess.run([sys.executable,'tools/export_perceptual_dev_family.py',
        '--stage','structural_factorial_dev','--config-dir','configs/perceptual_structural_factorial_dev_20261123',
        '--tag','STRUCTURAL_FACTORIAL_DEV','--title','Complete32 missing S/C/R single/static component cases',
        '--note','Four distinct seed0 fits,145600parameters/24passes/6912updates each; source17/prefix37/full37 '
                 'checks precede all accuracy. Reuse49completed reference rows; this is not81newruns. '
                 'Raw6/component effects and all equally selected curves retained; no Testselection.'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'tools/summarize_structural_factorial.py'],cwd=ROOT,check=True)


if __name__=='__main__':main()
