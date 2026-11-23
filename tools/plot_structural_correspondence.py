#!/usr/bin/env python3
"""Standalone scientific plot of the complete24-frame Train diagnostic."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    docs = Path(__file__).resolve().parents[1]/'docs/fusion_research'
    stem = docs/'PERCEPTUAL_MFT_20261123_STRUCTURE_TRAIN_AUDIT'
    report = json.loads(stem.with_suffix('.json').read_text())
    assert report['status'] == 'COMPLETE_SMALL_TRAIN_STRUCTURE_DIAGNOSTIC'
    sequences = list(dict.fromkeys(row['sequence'] for row in report['rows']))
    assert len(sequences) == 6 and len(report['rows']) == 24
    rows = {(r['sequence'], r['view']): r for r in report['rows']}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    x = np.arange(6); width = .24
    views = [('IR_current', 'Current new IR', '#277da8'),
             ('IR_wrong_frame', 'Wrong-frame IR', '#a84c32'),
             ('IR_spatial_roll', 'Quarter-image rolled IR', '#9a7d0a')]
    for i, (view, label, color) in enumerate(views):
        axes[0].bar(x+(i-1)*width, [rows[s, view]['mean_best_cost'] for s in sequences], width,
                    label=label, color=color)
        axes[1].bar(x+(i-1)*width, [rows[s, view]['inverse_consistent_fraction'] for s in sequences], width,
                    label=label, color=color)
    for ax in axes:
        ax.set_xticks(x, sequences, rotation=25); ax.grid(axis='y', alpha=.2)
    axes[0].set_title('Local self-similarity match cost (lower is better)')
    axes[0].set_ylabel('Mean absolute descriptor distance')
    axes[1].set_title('Forward/backward agreement (higher is better)')
    axes[1].set_ylabel('Fraction within1 diagnostic pixel'); axes[1].set_ylim(0, .7)
    axes[0].legend(fontsize=8)
    fig.suptitle('Only24 Train frames; same textured bbox-grid pixels for all IR views\n'
                 'Known gray translation passes6/6; actual IR pixel geometry and HOTA remain unproved', fontsize=11)
    path = stem.with_suffix('.png'); fig.savefig(path, dpi=125); plt.close(fig)
    assert path.stat().st_size <= 512*1024
    print(json.dumps(dict(path=str(path), bytes=path.stat().st_size, Test_read=False)))


if __name__ == '__main__':
    main()
