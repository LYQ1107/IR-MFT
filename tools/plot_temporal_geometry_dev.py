#!/usr/bin/env python3
"""All completed seed0 Dev controls; show aggregate and sequence disagreement."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    work = Path(json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())['work_root'])
    state = json.loads((work/'temporal_geometry_dev_state.json').read_text())
    assert state['status'] == 'COMPLETE' and len(state['jobs']) == 36
    docs = ROOT/'docs/fusion_research'
    report = json.loads((docs/'PERCEPTUAL_MFT_20261123_TEMPORAL_GEOMETRY_DEV.json').read_text())
    rows = {r['id']: r for r in report['rows']}; assert len(rows) == 36
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout='constrained')
    cues = ['rgb', 'gray', 'ir']; labels = ['RGB repeat', 'Grayscale RGB', 'Actual new pseudo-IR']
    colors = ['#2874a6', '#d68910', '#b03a2e']; positions = np.arange(3); width = .24
    for ax, suffix, title in [(axes[0, 0], '_raw', 'Original NMS'), (axes[0, 1], '', 'Priority integration')]:
        for i, (cue, label, color) in enumerate(zip(cues, labels, colors)):
            values = [rows[f'TC_{cue}_{role}{suffix}']['metrics']['HOTA'] for role in ['none', 'rgb_boxes', 'rgb_scores']]
            ax.bar(positions+(i-1)*width, values, width, color=color, label=label)
        ax.set_xticks(positions, ['Full fused', 'RGB boxes\nFused confidence', 'Fused boxes\nRGB confidence'])
        ax.set_title(title+': official combined Dev HOTA'); ax.set_ylim(43, 52)
        ax.axhline(47.579, color='0.4', ls='--', lw=1); ax.grid(axis='y', alpha=.2)
    ax = axes[1, 0]
    policies = ['none', 'identity', 'flow', 'without_current_evidence']
    for cue, label, color in zip(cues, labels, colors):
        ax.plot(range(4), [rows[f'TC_{cue}_{p}_raw']['metrics']['HOTA'] for p in policies], 'o-', color=color, label=label)
    ax.set_xticks(range(4), ['None', 'Identity', 'RGB flow', 'No current\nevidence guard'])
    ax.set_title('Temporal repair, raw output: no IR improvement'); ax.grid(alpha=.2)
    ax = axes[1, 1]; positions = np.arange(2); width = .19
    selected = [('TC_ir_none_raw', 'IR full fused', '#935116'),
                ('TC_ir_rgb_boxes_raw', 'IR RGB boxes', '#b03a2e'),
                ('TC_gray_rgb_scores_raw', 'Strongest gray', '#d68910'),
                ('TC_rgb_none_raw', 'Strongest RGB', '#2874a6')]
    for i, (key, label, color) in enumerate(selected):
        values = [rows[key]['per_sequence'][s]['HOTA'] for s in ['BT-005', 'SN-013']]
        ax.bar(positions+(i-1.5)*width, values, width, color=color, label=label)
    ax.set_xticks(positions, ['BT-005', 'SN-013']); ax.set_ylim(35, 65)
    ax.set_title('Per-sequence HOTA: IR geometry gain is uneven'); ax.grid(axis='y', alpha=.2)
    axes[0, 0].legend(fontsize=8); axes[1, 1].legend(fontsize=8, ncol=2)
    fig.suptitle('Complete36-case Dev diagnosis, fixed seed0 weights\nCandidate role separation; three-seed replication and IR interventions pending, no Test claim', fontsize=12)
    path = docs/'PERCEPTUAL_MFT_20261123_TEMPORAL_GEOMETRY_DEV.png'
    fig.savefig(path, dpi=125); plt.close(fig); assert path.stat().st_size <= 512*1024
    print(json.dumps(dict(path=str(path), bytes=path.stat().st_size, Test_read=False)))


if __name__ == '__main__':
    main()
