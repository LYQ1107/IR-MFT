#!/usr/bin/env python3
"""Publish small provenance for the six completed fixed-seed structural fits."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config_path = ROOT/'configs/perceptual_structural_relative_20261123.json'
    config = json.loads(config_path.read_text())
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    state = json.loads((work/'structural_relative_fit_state.json').read_text())
    assert state['status'] == 'COMPLETE'
    reference = json.loads((work/'correlation_fit/ir_full_s0/complete.json').read_text())
    draws = [r['sample_digest'] for r in reference['history']]
    records = []
    for row in config['initial_fits']:
        key = f"{row['cue']}_{row['variant']}_s{row['seed']}"
        assert row['seed'] == 0 and state['jobs'][key]['status'] == 'COMPLETE'
        folder = work/'structural_relative_fit'/key
        source = folder/'complete.json'; fit = json.loads(source.read_text())
        assert fit['status'] == 'COMPLETE_TRAIN_STRUCTURAL_RELATIVE'
        assert (fit['cue'], fit['variant'], fit['seed']) == (row['cue'], row['variant'], 0)
        assert fit['epochs'] == 24 and fit['successful_steps_this_invocation'] == 6912
        assert fit['Train_frames'] == 1152 and fit['Dev_fit_rows'] == 0 and not fit['Test_read']
        assert fit['frozen_original_head'] and fit['frozen_original_encoder']
        assert fit['initial_raw_RGB_head_bit_exact'] and fit['trainable_parameters'] == 145600
        assert fit['physical_gpu'] in ['2', '3']
        assert fit['config_sha256'] == digest(config_path)
        assert [h['epoch'] for h in fit['history']] == list(range(1, 25))
        assert sum(h['steps'] for h in fit['history']) == 6912
        assert [h['sample_digest'] for h in fit['history']] == draws
        for name in ['FPN_sources', 'structure_sources']:
            assert sum(r['frames'] for r in fit[name].values()) == 1152
            for r in fit[name].values():
                assert digest(Path(r.get('manifest', r.get('path')))) == r['sha256']
        checkpoints = []
        for epoch in [6, 12, 18, 24]:
            path = folder/f'epoch{epoch:02d}.pt'
            checkpoints.append(dict(epoch=epoch, bytes=path.stat().st_size,
                                    sha256=digest(path), path=str(path)))
        records.append(dict(id=key, manifest=str(source), manifest_sha256=digest(source),
                            metadata=fit, checkpoints=checkpoints))
    assert len(records) == 6
    for name in ['config_sha256', 'source_RGB_sha256', 'single_seed_user_policy_sha256']:
        assert len({r['metadata'][name] for r in records}) == 1
    report = dict(status='COMPLETE_SIX_FIXED_SEED_STRUCTURAL_FITS', rows=records,
        fixed_seed=0, additional_seed_work=False, all24_sample_digests_match_original_seed0=True,
        budget='Each fit:24 passes over1152 cached Train frames,6912 updates; '
               'not24 passes over all27251 Train frames. Train480x864, Dev800x1440.',
        compute='Matched145600 adapter parameters. RGB-repeat reuses one encoder; '
                'gray/actualIR use two shared-weight encodings. Shared producer timing '
                'is not independent per-method end-to-end FPS.',
        accuracy_read=False, Test_read=False, goal_met=False, supported_innovations=0)
    stem = ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_STRUCTURAL_RELATIVE_TRAINING_LEDGER'
    stem.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Fixed-seed structural-relative training ledger', '', report['budget'], '', report['compute'], '',
        '| Fit | Epochs | Updates | Parameters | Physical GPU | Seconds |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in records:
        fit = row['metadata']
        lines.append(f"| {row['id']} | 24 | 6912 | 145600 | {fit['physical_gpu']} | {fit['seconds']:.1f} |")
    lines += ['', 'All six fits use seed0 and exactly the same24 epoch sample digests '
              'as the original seed0 fit. Every checkpoint hash, training loss curve, '
              'cache manifest and source version is in the adjacent JSON. No accuracy claim.']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'], fits=6, accuracy_read=False)))


if __name__ == '__main__':
    main()
