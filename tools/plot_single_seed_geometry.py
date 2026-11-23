#!/usr/bin/env python3
"""Plot complete seed0 NMS/emission comparisons, without partial removals."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    docs = Path(__file__).resolve().parents[1]/'docs/fusion_research'
    stem = docs/'PERCEPTUAL_MFT_20261123_SINGLE_SEED_GEOMETRY_ANALYSIS'
    report = json.loads(stem.with_suffix('.json').read_text())
    assert report['status'] in ['COMPLETE_SINGLE_SEED_GEOMETRY_MAIN',
                                'COMPLETE_SINGLE_SEED_GEOMETRY_AND_REMOVALS']
    roles = ['full', 'rgb_boxes', 'fused_nms_rgb_boxes', 'rgb_nms_fused_boxes']
    labels = ['Fused / fused', 'RGB / RGB', 'Fused / RGB', 'RGB / fused']
    cues = [('rgb', 'RGB repeat', '#277da8'), ('gray', 'Grayscale RGB', '#a17e00'),
            ('ir', 'Actual new pseudo-IR', '#aa4239')]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout='constrained')
    for ax, sequence in zip(axes, [None, 'BT-005', 'SN-013']):
        for i, (cue, label, color) in enumerate(cues):
            factorial = report['raw_factorials'][cue]
            values = factorial['raw_HOTA'] if sequence is None else factorial['per_sequence'][sequence]
            ax.plot(np.arange(4), [values[role] for role in roles], 'o-', label=label, color=color)
        ax.set_xticks(np.arange(4), labels, rotation=23, ha='right')
        ax.set_xlabel('NMS geometry / emitted geometry')
        ax.set_title('Official combined Dev' if sequence is None else sequence)
        ax.set_ylabel('HOTA'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Complete4834-frame Dev, fixed seed0; fused confidence held fixed\n'
                 'RGB18 / gray18 / IR24 weights unchanged; no Test or established contribution claim', fontsize=11)
    path = stem.with_suffix('.png'); fig.savefig(path, dpi=125); plt.close(fig)
    assert path.stat().st_size <= 512*1024
    print(json.dumps(dict(path=str(path), bytes=path.stat().st_size, Test_read=False)))


if __name__ == '__main__':
    main()
