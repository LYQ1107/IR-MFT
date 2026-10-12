#!/usr/bin/env python3
"""Static scientific figure of complete three-seed temporal selections."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cfp.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--include-modality', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Scientific figures are immutable')
    paths = [Path('configs/cfp/baseline_selection_post_neck_v1.json'), Path('configs/cfp/plim_first_three_seed_selection_v1.json'), Path('configs/cfp/plim_anchor_three_seed_selection_v3.json')]
    external, first, anchor = [json.loads(path.read_text())['families'] for path in paths]
    labels, rows, colors = [], [], []
    entries = [('BoT-SORT RGB', external['bot_sort'], '#64748b'),
               ('First GRU predict', first['gru_predictive_clean'], '#2a9d8f'),
               ('First GRU ordinary', first['gru_association_clean'], '#e9a23b'),
               ('First Transformer predict', first['transformer_predictive_clean'], '#2a9d8f'),
               ('First Transformer ordinary', first['transformer_association_clean'], '#e9a23b'),
               ('Anchor GRU predict', anchor['plim_v3_dev_gru_predictive_anchor'], '#2a9d8f'),
               ('Anchor GRU ordinary', anchor['plim_v3_dev_gru_association_anchor'], '#e9a23b'),
               ('Anchor Transformer predict', anchor['plim_v3_dev_transformer_predictive_anchor'], '#2a9d8f'),
               ('Anchor Transformer ordinary', anchor['plim_v3_dev_transformer_association_anchor'], '#e9a23b')]
    if args.include_modality:
        for view, label in [('ir', 'IR'), ('rgb_ir', 'RGB+IR'), ('two_rgb', 'Two RGB')]:
            source = Path(f'configs/cfp/plim_modality_{view}_three_seed_selection_v1.json')
            paths.append(source)
            data = json.loads(source.read_text())['families']
            for objective, color in [('predictive', '#2a9d8f'), ('association', '#e9a23b')]:
                entries.append((label + ' GRU ' + ('predict' if objective == 'predictive' else 'ordinary'),
                                data[f'plim_mod_dev_{view}_gru_{objective}_anchor'], color))
    for label, entry, color in entries:
        labels.append(label)
        rows.append(entry['selected'])
        colors.append(color)
    figure, axes = plt.subplots(1, 3, figsize=(14, 9 if args.include_modality else 6), sharey=True, constrained_layout=True)
    y = np.arange(len(rows))
    for axis, metric in zip(axes, ['HOTA', 'IDF1', 'IDSW']):
        values = [row['mean'][metric] for row in rows]
        deviations = [row['sample_std'][metric] for row in rows]
        for index, (value, deviation, color) in enumerate(zip(values, deviations, colors)):
            axis.errorbar(value, y[index], xerr=deviation, fmt='o', color=color, capsize=3, markersize=6)
        axis.set_yticks(y, labels, fontsize=9)
        axis.set_xlabel(metric + (' (%)' if metric != 'IDSW' else ' (count; lower is better)'))
        axis.grid(axis='x', alpha=.2)
        axis.set_axisbelow(True)
    axes[0].invert_yaxis()
    axes[0].set_xlim(45, 55)
    axes[1].set_xlim(52, 66)
    axes[2].set_xlim(70, 125)
    figure.suptitle('Fixed-proposal association development: complete three-seed means ± sample SD\nEach family selects its highest mean HOTA from the same three settings', fontsize=11)
    args.output.mkdir(parents=True)
    image = args.output / 'temporal_three_seed_selection.png'
    figure.savefig(image, dpi=180)
    plt.close(figure)
    write_json(args.output / 'manifest.json', dict(status='COMPLETE', figure_sha256=sha256(image),
                input_sha256={str(path): sha256(path) for path in paths}, source_sha256=sha256(__file__),
                labels=labels, measured_rows=rows, scope='Canonical dev only; different native BoT lifecycle disclosed. Sample SD is seed variation, not independent-video confidence. No historical-test comparison.'))


if __name__ == '__main__':
    main()
