#!/usr/bin/env python3
"""All-seed summary with dispersion, sequence disagreement and interventions."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    docs = Path(__file__).resolve().parents[1]/'docs/fusion_research'
    stem = docs/'PERCEPTUAL_MFT_20261123_CORRELATION_REPLICATION_ANALYSIS'
    report = json.loads(stem.with_suffix('.json').read_text())
    assert report['status'] == 'COMPLETE_3_SEED_DEV_REPLICATION_NOT_TEST'
    roles = ['full', 'rgb_boxes', 'rgb_scores']
    role_names = ['Full fused', 'RGB boxes,\nfused confidence', 'Fused boxes,\nRGB confidence']
    cues = [('rgb', 'RGB repeat', '#277da8'), ('gray', 'Grayscale RGB', '#b08400'),
            ('ir', 'Actual new pseudo-IR', '#aa4239')]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), layout='constrained')
    for ax, policy in [(axes[0, 0], 'raw'), (axes[0, 1], 'priority')]:
        for i, (cue, label, color) in enumerate(cues):
            rows = [report['fixed_epoch_head_roles'][role+'_'+policy]['groups'][cue] for role in roles]
            ax.errorbar(np.arange(3)+(i-1)*.13, [r['mean']['HOTA'] for r in rows],
                yerr=[r['sample_std']['HOTA'] for r in rows], fmt='o', capsize=4, color=color, label=label)
        ax.set_xticks(np.arange(3), role_names); ax.grid(alpha=.2)
        ax.axhline(47.579, color='.5', ls='--', lw=1)
        ax.set_ylabel('Official combined Dev HOTA')
        ax.set_title(policy+': mean ± sample SD over3 adapter seeds')
    ax = axes[1, 0]
    for i, (cue, label, color) in enumerate(cues):
        for j, role in enumerate(['full', 'rgb_boxes']):
            row = report['fixed_epoch_head_roles'][role+'_raw']['groups'][cue]
            ax.plot(np.arange(2)+(i-1)*.04,
                [row['per_sequence'][seq]['HOTA'] for seq in ['BT-005', 'SN-013']],
                'o-' if j == 0 else 's--', color=color,
                label=label+(' full' if j == 0 else ' RGB boxes'))
    ax.set_xticks([0, 1], ['BT-005', 'SN-013']); ax.set_ylabel('Per-sequence mean HOTA')
    ax.set_title('Raw outputs: inspect each video separately'); ax.grid(alpha=.2)
    ax.legend(fontsize=7, ncol=2)
    ax = axes[1, 1]
    old = json.loads((docs/'PERCEPTUAL_MFT_20261123_TEMPORAL_GEOMETRY_DEV.json').read_text())
    old = {row['id']: row for row in old['rows']}
    for role, label, color in [('full', 'IR full', '#935116'), ('rgb_boxes', 'IR RGB boxes', '#aa4239'),
                               ('rgb_scores', 'IR RGB confidence', '#287f75')]:
        key = 'TC_ir_'+('none' if role == 'full' else role)+'_raw'
        values = [old[key]['metrics']['HOTA']]
        values += [next(r['metrics']['HOTA'] for r in report['seed0_IR_feature_interventions']
                    if r['intervention'] == intervention and r['head_role'] == role and r['policy'] == 'raw')
                   for intervention in ['unavailable', 'spatial_roll']]
        ax.plot(np.arange(3), values, 'o-', color=color, label=label)
    ax.set_xticks(np.arange(3), ['Correct IR', 'Unavailable', 'Quarter-feature roll'])
    ax.set_title('IR feature interventions: fixed adapter seed0 only')
    ax.set_ylabel('Official combined Dev HOTA'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Complete84-case replication plus unchanged seed0 anchors;4834 Dev frames\n'
                 'Fixed RGB18 / gray18 / IR24 epochs; no Test or established contribution claim', fontsize=11)
    path = stem.with_suffix('.png'); fig.savefig(path, dpi=125); plt.close(fig)
    assert path.stat().st_size <= 512*1024
    print(json.dumps(dict(path=str(path), bytes=path.stat().st_size, Test_read=False)))


if __name__ == '__main__':
    main()
