#!/usr/bin/env python3
"""Static scientific figure from verified complete metrics; no model selection."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = json.loads(Path('docs/research/PHASE_VI_SYNTHESIS_MEASURED.json').read_text())
    matrix = data['official']['combinations']['result']['families']
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    families = ['B0', 'A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC']
    ax = axes[0, 0]
    scores = [matrix[f]['metrics']['COMBINED_SEQ']['HOTA'] for f in families]
    ax.scatter(np.arange(8), scores, color='#2563eb', s=44)
    ax.axhline(scores[0], color='#64748b', linestyle='--', linewidth=1)
    for i, score in enumerate(scores):
        ax.annotate(f'{score:.3f}', (i, score), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9)
    ax.set_xticks(np.arange(8), families)
    ax.set_ylim(33.2, 34.7)
    ax.set_ylabel('HOTA (zoomed axis)')
    ax.set_title('A. Complete same-fold matrix, initialization 0')

    ax = axes[0, 1]
    kinds = ['ordinary', 'noema', 'ema']
    statistics = [data['model_seed_statistics'][k]['COMBINED_SEQ']['HOTA'] for k in kinds]
    ax.errorbar(np.arange(3), [v['mean'] for v in statistics], yerr=[v['std_population'] for v in statistics],
                fmt='o', capsize=5, color='#7c3aed', markersize=7)
    ax.axhline(scores[0], color='#64748b', linestyle='--', linewidth=1, label='Unchanged ordinary GRU')
    ax.set_xticks(np.arange(3), ['Ordinary residual', 'Predictive, no EMA', 'Predictive, EMA'])
    ax.set_ylim(32.8, 33.9)
    ax.set_ylabel('HOTA: mean ± population SD')
    ax.set_title('B. Three model initialization/query seeds')
    ax.text(.03, .97, 'Encoder/producer 0 fixed; not scene confidence intervals', transform=ax.transAxes,
            va='top', fontsize=8)
    ax.legend(loc='lower right', fontsize=8)

    ax = axes[1, 0]
    sequences = [s for s in matrix['B0']['metrics'] if s != 'COMBINED_SEQ']
    x = np.arange(len(sequences))
    for offset, family, color in [(-.25, 'A', '#f59e0b'), (0, 'B', '#7c3aed'), (.25, 'AB', '#2563eb')]:
        values = [matrix[family]['metrics'][s]['HOTA'] - matrix['B0']['metrics'][s]['HOTA'] for s in sequences]
        ax.bar(x+offset, values, width=.24, label=family, color=color)
    ax.axhline(0, color='#64748b', linewidth=1)
    ax.set_xticks(x, sequences, rotation=20)
    ax.set_ylabel('Per-video HOTA change vs corresponding B0')
    ax.set_title('C. Benefit varies across complete videos')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    variants = ['C', 'AB', 'AC', 'BC', 'ABC']
    values = []
    for family in variants:
        rows = [v for name, v in data['events']['paired'].items() if name.startswith(family+'_')]
        values.append([sum(v['N01'] for v in rows), sum(v['N10'] for v in rows)])
    values = np.asarray(values); x = np.arange(len(variants))
    ax.bar(x-.18, values[:, 0], width=.35, label='N01: wrong → correct', color='#16a34a')
    ax.bar(x+.18, values[:, 1], width=.35, label='N10: correct → wrong', color='#dc2626')
    ax.set_xticks(x, variants)
    ax.set_ylabel('Known frame/track decisions on common birth support')
    ax.set_title('D. Local corrections are not official IDSW')
    ax.legend(fontsize=8)
    ax.text(.03, .97, 'Unknown and changed births excluded', transform=ax.transAxes, va='top', fontsize=8)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Phase VI: completed matrix and initialization controls\nFrozen backbones exposed; identity-pipeline replication pending', fontsize=13)
    output = Path('docs/research/figures'); output.mkdir(exist_ok=True)
    fig.savefig(output / 'phase_vi_synthesis.png', dpi=160)
    svg = output / 'phase_vi_synthesis.svg'
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
