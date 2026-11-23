#!/usr/bin/env python3
"""Audit all four completed fits and their identical seed0 sample schedules."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pseudo_ir_comparison import dump, sha256


def main():
    phase_path = ROOT/'configs/perceptual_structural_nocycle_20261123.json'
    phase = json.loads(phase_path.read_text())
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    state = json.loads((work/'structural_nocycle_fit_state.json').read_text())
    assert state['status'] == 'COMPLETE' and len(state['jobs']) == 4
    old_path = work/'structural_relative_fit/ir_no_cycle_s0/complete.json'
    original = json.loads(old_path.read_text())
    schedule = [r['sample_digest'] for r in original['history']]
    assert len(schedule) == 24
    rows = []
    for fit in phase['fits']:
        key = f"{fit['cue']}_{fit['variant']}_s0"
        path = work/'structural_nocycle_fit'/key/'complete.json'
        meta = json.loads(path.read_text())
        assert meta['status'] == 'COMPLETE_TRAIN_STRUCTURAL_RELATIVE'
        assert meta['seed'] == 0 and meta['refinement_plan_sha256'] == sha256(phase_path)
        assert meta['epochs'] == 24 and meta['successful_steps_this_invocation'] == 6912
        assert meta['trainable_parameters'] == 145600 and meta['Train_frames'] == 1152
        assert meta['initial_raw_RGB_head_bit_exact'] and meta['Dev_fit_rows'] == 0 and not meta['Test_read']
        assert [r['sample_digest'] for r in meta['history']] == schedule
        checkpoints = []
        for epoch in phase['snapshots']:
            cp = path.parent/f'epoch{epoch:02d}.pt'
            checkpoints.append(dict(epoch=epoch, path=str(cp), sha256=sha256(cp), bytes=cp.stat().st_size))
        rows.append(dict(id=key, metadata=meta, complete_sha256=sha256(path), checkpoints=checkpoints))
    report = dict(status='COMPLETE_FOUR_DISTINCT_SEED0_NO_CYCLE_FITS', fits=rows,
        fixed_seed=0, additional_seed_work=False, refinement_plan_sha256=sha256(phase_path),
        all24_sample_digests_exact_to_existing_primary_IR=True,
        primary_IR_retrained=False, primary_complete_sha256=sha256(old_path),
        scope='Four architecture/control fits, not four random seeds. Each24passes over1152 cached Trainframes '
              '(192 per6videos),6912updates at480x864; not24passes over the full27251frameTrain. '
              'No Dev/Test rows used for fitting. Existing FPN/descriptors reused, no full image/feature cache duplication.',
        goal_met=False, supported_innovations=0)
    stem = ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_STRUCTURAL_NOCYCLE_TRAINING_LEDGER'
    dump(stem.with_suffix('.json'), report)
    lines = ['# Four fixed-seed no-cycle control fits', '', report['scope'], '',
             '| Fit | Seed | Parameters | Cache passes | Updates | Checkpoints |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['id']} | 0 | 145600 | 24 | 6912 | 4 |")
    lines += ['', 'All24 sampling digests match the completed original IR no-cycle fit. '
              'The primary IR model was reused, not retrained. All16 checkpoints and original runtime sources '
              'are hashed in the adjacent JSON; large checkpoint files remain outside Git.']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(fits=4, snapshots=16, fixed_seed=0, all24_sample_digests_exact=True)))


if __name__ == '__main__':
    main()
