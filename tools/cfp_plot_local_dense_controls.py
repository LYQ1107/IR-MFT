#!/usr/bin/env python3
"""Static exportable scientific figure; seed/support scope stays visible."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--measured', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if any(args.output.with_suffix('.'+ext).exists() for ext in ('png', 'svg', 'pdf')):
        raise ValueError('New immutable figure stem required')
    d = json.loads(args.measured.read_text())
    if not d['exact_train_sample_digests_and_frozen_query_supports'] or len(d['runs'])!=9:
        raise ValueError('Actual matched firstseed controls required')
    modes = ['global', 'global_gru', 'local_attention', 'local_temporal', 'local_contrastive',
             'no_ema', 'no_prediction', 'no_id', 'full']
    labels = ['Frozen CLS FIFO', 'Global adapter', 'Global + GRU384', 'Local attention',
        'Local temporal Transformer', 'Local temporal contrastive', 'Dense no EMA',
        'No masked-prediction loss', 'No explicit ID loss', 'Full local Dense JEPA']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1.4, 1.4, 1]})
    for ax, video in zip(axes[:2], ('BT-001', 'SN-015')):
        reference = d['frozen']['models']['cls_fifo'][video]['protocol']['gt']['detector']['25']
        values = [reference['retrieval']['rank1_percent']]+[d['runs'][m]['evidence']['evaluations'][video]['protocol']['gt']['detector']['25']['retrieval']['rank1_percent'] for m in modes]
        ax.barh(labels, values, color=['#7a8793']+['#2e78a8']*8+['#cb6549'])
        ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel('Rank-1 (%)')
        ax.set_title(f'{video}: {reference["retrieval"]["supported_history_and_target"]} supported queries', fontsize=11)
        for i, value in enumerate(values):
            ax.text(value+1, i, f'{value:.2f}', va='center', fontsize=8)
        ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    stages = ('encoder_fifo', 'context_pool', 'prediction')
    values = [d['stages']['videos']['BT-001'][s]['protocol']['gt']['detector']['25']['retrieval']['rank1_percent'] for s in stages]
    ax = axes[2]
    ax.bar(['Encoder\nFIFO', 'Context\npool', 'Regional\nprediction'], values, color=['#2e78a8', '#2e78a8', '#cb6549'])
    ax.set_ylim(0, 100); ax.set_ylabel('Rank-1 (%)'); ax.set_title('BT-001 inner-only\nfixed-model stages', fontsize=11)
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    for i, value in enumerate(values):
        ax.text(i, value+1, f'{value:.2f}', ha='center', fontsize=9)
    fig.suptitle('Same-scene identity diagnostics: seed 0 only\nGT history / actual detector pool / minimum past gap 25', fontsize=12)
    fig.text(.5, .01, 'Clean physical offline grouping; unknowns retained. No native HOTA/AssA or raw online FPS claim.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .035, 1, .91))
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(args.output.with_suffix('.'+ext), dpi=180, bbox_inches='tight')


if __name__=='__main__':
    main()
