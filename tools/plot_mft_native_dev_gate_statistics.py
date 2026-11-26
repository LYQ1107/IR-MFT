#!/usr/bin/env python3
"""Plot all saved normal/roll IR gate pairs, without choosing checkpoints."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-directory', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only scientific figures')
    audit = args.audit_directory/'complete.json'
    source = args.audit_directory/'all_paired_normal_vs_roll.csv'
    before = {str(p): sha(p) for p in (audit, source, Path(__file__).resolve())}
    report = json.loads(audit.read_text())
    if report['split'] != 'dev' or report['normal_roll_level_pairs'] != report['stage']*12:
        raise ValueError('Complete two-mode, two-video, three-level Dev pairs required')
    with source.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    args.output_directory.mkdir(exist_ok=False)
    results = []
    colors = {'BT-005': '#0072B2', 'SN-013': '#D55E00'}
    with plt.rc_context({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'axes.titlesize': 10}):
        for field, title, label, stem in [
            ('gate', 'Mean quality gate: correct and spatially rolled IR', 'Mean image-level gate', 'all_Dev_IR_gate_response'),
            ('relative_RMS', 'Relative feature correction: correct and spatially rolled IR', 'Relative feature correction RMS', 'all_Dev_IR_correction_response')]:
            fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.1), sharex=True)
            for i, mode in enumerate(('quality', 'temporal')):
                for j, level in enumerate(('P3', 'P4', 'P5')):
                    ax = axes[i, j]
                    for sequence, color in colors.items():
                        selected = sorted([r for r in rows if r['mode'] == mode and r['level'] == level and
                                           r['sequence'] == sequence], key=lambda r: int(r['epoch']))
                        if [int(r['epoch']) for r in selected] != list(range(1, report['stage']+1)):
                            raise ValueError('Do not plot partial or cherry-picked epochs')
                        for kind, style in [('normal', '-'), ('rolled', '--')]:
                            ax.plot([int(r['epoch']) for r in selected], [float(r[kind+'_'+field]) for r in selected],
                                    style, marker='o', markersize=3, linewidth=1.3, color=color,
                                    label=sequence+(' correct' if kind == 'normal' else ' rolled'))
                    ax.set_title(('Quality' if mode == 'quality' else 'Causal temporal')+' / '+level)
                    ax.set_ylim(bottom=0, top=1 if field == 'gate' else None)
                    ax.grid(alpha=.18)
                    ax.set_xticks(list(range(1, report['stage']+1)))
                    if i == 1:
                        ax.set_xlabel('Complete Train epoch')
                    if j == 0:
                        ax.set_ylabel(label)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.suptitle(title, y=.995, fontsize=12)
            fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(.5, .963), frameon=False)
            fig.text(.5, .014, 'All original Dev videos and checkpoints; aggregate moments, not calibration or a gate-removal experiment.',
                     ha='center', fontsize=8)
            fig.tight_layout(rect=(0, .035, 1, .91))
            for suffix in ('.svg', '.png'):
                path = args.output_directory/(stem+suffix)
                fig.savefig(path, dpi=170, facecolor='white', metadata={'Creator': 'IR-MFT saved Dev gate diagnostic'})
                results.append(dict(file=str(path), sha256=sha(path), bytes=path.stat().st_size,
                                    scope='All normal/roll IR pairs, not RGB/gray/off curves; those remain in the complete audit CSV.'))
            plt.close(fig)
    if any(sha(p) != h for p, h in before.items()):
        raise ValueError('An audited plot source changed')
    value = dict(status='COMPLETE_ALL_DEV_IR_GATE_AND_CORRECTION_FIGURES',
                 all_paired_rows=len(rows), source_hashes=before, outputs=results,
                 new_GPU_work=False, new_accuracy=False, supported_innovations=0, goal_met=False)
    (args.output_directory/'complete.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
