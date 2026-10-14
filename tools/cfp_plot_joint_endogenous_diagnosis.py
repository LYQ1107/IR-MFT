#!/usr/bin/env python3
"""Scientific plots from all nine exact inner endogenous trajectory diagnoses."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if a.output.with_suffix('.json').exists():
        raise ValueError('New immutable figure/provenance required')
    d = json.loads(a.manifest.read_text())
    if d['status'] != 'COMPLETE_ALL_NINE_INNER_ENDOGENOUS_IDENTITY_DIAGNOSES' or len(d['runs']) != 9:
        raise ValueError('All-nine complete actual inner diagnoses required')
    architectures = ['pair', 'cross_transformer', 'joint']
    rows = [[d['runs'][f'{arch}_s{s}']['offline_diagnosis'] for s in [0, 1, 2]] for arch in architectures]
    plot_values = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.arange(3)
    for offset, (key, label, color) in enumerate([('model_existing_logits', 'Model ID logits', '#285f99'), ('fifo_cost', 'Ordinary FIFO', '#bb8041'), ('geometry', 'Geometry only', '#6c887a')]):
        values = np.array([[r['local_same_set_existing_rank1'][key]['percentage'] for r in group] for group in rows])
        axes[0].bar(x+(offset-1)*.24, values.mean(1), .24, yerr=values.std(1, ddof=1), capsize=3, label=label, color=color)
        plot_values[key] = values.tolist()
    axes[0].set_ylim(0, 100); axes[0].set_ylabel('Known existing-ID local Rank1 (%)')
    axes[0].set_title('Same-set cue ranking within each native model')
    axes[0].legend(fontsize=8, frameon=True, facecolor='white', framealpha=.96, loc='lower left')
    for offset, (key, label, color) in enumerate([('random_codes', 'Four alternative code draws', '#285f99'), ('zero_dictionary', 'Zero dictionary intervention', '#bb8041')]):
        values = np.array([[100*(np.mean([r['offline_dictionary_sensitivity'][str(k)]['selected_assignment_changes'] for k in [1001, 1002, 1003, 1004]]) if key=='random_codes' else r['offline_dictionary_sensitivity'][key]['selected_assignment_changes'])/r['offline_dictionary_sensitivity']['original']['detections'] for r in group] for group in rows])
        axes[1].bar(x+(offset-.5)*.32, values.mean(1), .32, yerr=values.std(1, ddof=1), capsize=3, label=label, color=color)
        plot_values[key] = values.tolist()
    axes[1].set_ylim(0, 40); axes[1].set_ylabel('Assignments changed on fixed observed sets (%)')
    axes[1].set_title('Dictionary nuisance sensitivity, stride5 inner sets')
    axes[1].legend(fontsize=8, frameon=False, loc='upper left')
    for ax in axes:
        ax.set_xticks(x, ['Pair', 'Cross Transformer', 'Joint set'])
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.text(.5, .03, 'Three seeds; error bars are sample SD. Each architecture has its own endogenous sets.\nOffline interventions only: these bars are neither full-tracker gains nor causal attribution.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .1, 1, 1])
    artifacts = {}
    for suffix in ['png', 'svg', 'pdf']:
        path = a.output.with_suffix('.'+suffix); fig.savefig(path, dpi=180, bbox_inches='tight')
        artifacts[suffix] = dict(path=str(path), sha256=sha256(path))
    write_json(a.output.with_suffix('.json'), dict(status='GENERATED_FROM_ALL_NINE_MEASURED_INNER_DIAGNOSES', manifest_sha256=sha256(a.manifest), values=plot_values, artifacts=artifacts,
        sources=snapshot_sources(a.output.with_suffix('.sources'), [__file__]), scope='Exact same-set cue ranks and offline fixed-input dictionary interventions. No native HOTA/IDF1 improvement claim; different endogenous support across architectures.'))


if __name__ == '__main__':
    main()
