#!/usr/bin/env python3
"""Audit matched adapter training without loading models or reading accuracy."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = json.loads((ROOT / 'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    records = []
    sample_draws = {}
    for seed in range(3):
        for cue in ['rgb', 'gray', 'ir']:
            folder = work / 'correlation_fit' / f'{cue}_full_s{seed}'
            source = folder / 'complete.json'
            fit = json.loads(source.read_text())
            assert fit['status'] == 'COMPLETE_TRAIN_ONLY_ADAPTER'
            assert (fit['cue'], fit['variant'], fit['seed']) == (cue, 'full', seed)
            assert fit['epochs'] == 24 and fit['successful_steps_this_invocation'] == 6912
            assert not fit['Test_read'] and fit['Dev_fit_rows'] == 0
            assert fit['frozen_original_head'] and fit['frozen_original_encoder']
            assert fit['physical_gpu'] in ['2', '3']
            history = fit['history']
            assert [row['epoch'] for row in history] == list(range(1, 25))
            assert sum(row['steps'] for row in history) == 6912
            draws = [row['sample_digest'] for row in history]
            if seed in sample_draws:
                assert draws == sample_draws[seed], f'Unmatched sample draws: {cue}, seed{seed}'
            sample_draws[seed] = draws
            assert sum(row['frames'] for row in fit['Train_sources'].values()) == 1152
            for row in fit['Train_sources'].values():
                assert digest(Path(row['manifest'])) == row['sha256']
            checkpoints = []
            for epoch in [6, 12, 18, 24]:
                path = folder / f'epoch{epoch:02d}.pt'
                checkpoints.append(dict(epoch=epoch, path=str(path), bytes=path.stat().st_size,
                                        sha256=digest(path)))
            records.append(dict(id=f'{cue}_full_s{seed}', fit_manifest=str(source),
                fit_manifest_sha256=digest(source), metadata={key: value for key, value in fit.items()
                if key not in ['history', 'Train_sources']}, Train_sources=fit['Train_sources'],
                epoch_history=history, checkpoints=checkpoints))
    for key in ['config_sha256', 'source_RGB_sha256', 'source_sha256',
                'adapter_source_sha256', 'loss_source_sha256', 'trainable_parameters']:
        assert len({row['metadata'][key] for row in records}) == 1, key
    assert len({json.dumps(row['Train_sources'], sort_keys=True) for row in records}) == 1
    report = dict(status='COMPLETE_MATCHED_9_FIT_TRAINING_AUDIT', rows=records,
        matching=dict(all_three_cues_per_seed_identical_24_sample_digests=True,
                      same_source_cache_head_loss_and_update_budget=True),
        budget='24 passes over a fixed1152-frame Train cache,6912 updates per fit; '
               'not24 passes over all27251 original Train frames. Train480x864; Dev800x1440.',
        compute='RGB-repeat reuses RGB encoding; gray and actualIR each require a second '
                'shared-weight encoder pass. Equal adapter capacity is not equal backbone compute.',
        protocol_sha256=digest(ROOT / plan['protocol']),
        main_plan_sha256=digest(ROOT / 'configs/perceptual_mft_20261122.json'),
        Test_read=False, accuracy_read=False, supported_multimodal_innovations=0,
        scope='Training provenance only. No accuracy, robustness or novelty claim. '
              'Checkpoints stay outside Git; hashes and small metadata are published.')
    stem = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261123_CORRELATION_TRAINING_LEDGER'
    stem.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Matched correlation adapter training ledger', '', report['budget'], '',
             report['compute'], '', '| Fit | Epochs | Updates | Parameters | GPU | Seconds |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in records:
        fit = row['metadata']
        lines.append(f"| {row['id']} | 24 | 6912 | {fit['trainable_parameters']} | "
                     f"{fit['physical_gpu']} | {fit['seconds']:.1f} |")
    lines += ['', 'All24 epoch sample digests match across the three cues for each seed. '
              'The adjacent JSON retains every epoch history, six cache-manifest hashes, '
              'source hashes and four checkpoint hashes per fit.', '', report['scope']]
    stem.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=report['status'], fits=len(records), accuracy_read=False)))


if __name__ == '__main__':
    main()
