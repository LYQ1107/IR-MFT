#!/usr/bin/env python3
"""Read the new complete family together; retain all existing reference curves."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    path = Path(plan['work_root'])/'structural_nocycle_dev_state.json'
    while True:
        state = json.loads(path.read_text())
        if 'FAIL' in state['status']:
            raise RuntimeError('Preserve failure; no partial new-family accuracy report')
        if state['status'] == 'COMPLETE':
            break
        time.sleep(5)
    command = [sys.executable, 'tools/export_perceptual_dev_family.py',
        '--stage', 'structural_nocycle_dev', '--config-dir', 'configs/perceptual_structural_nocycle_dev_20261123',
        '--tag', 'STRUCTURAL_NOCYCLE_DEV', '--title', 'All36 new no-cycle matched controls and removals',
        '--note', 'Four distinct seed0 fits,24passes/6912updates/1152Traincacheframes and145600parameters each. '
                  'Bothsource8/prefix32/full39proofs precede accuracy. Existing IR no_cycle/no_structure and '
                  'canonical RGB17results are referenced in the analysis, not rerun. No newTest/noveltyclaim.']
    subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run([sys.executable, 'tools/summarize_structural_nocycle.py'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
