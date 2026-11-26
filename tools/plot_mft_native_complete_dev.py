#!/usr/bin/env python3
"""Plot an actually completed native Dev stage from its original official CSVs."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_mft_native_complete_dev import (  # noqa: E402
    CANONICAL, CUES, FIELDS, FRESH, LENGTHS, MODES, completed_report, require, sha)


COLORS = {'rgb': '#0072B2', 'gray': '#777777', 'ir': '#D55E00'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev-report', type=Path, required=True)
    parser.add_argument('--expect-report-sha256', required=True)
    parser.add_argument('--stage', type=int, choices=(6, 9, 12), required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--preview-directory', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only figure export required')
    source = args.dev_report.resolve()
    require(source.is_relative_to(ROOT), 'Use the original repository Dev audit report')
    data, lookup, own, _, bindings = completed_report(source, args.expect_report_sha256, args.stage)
    helper = ROOT / 'tools/analyze_mft_native_complete_dev.py'
    inputs = {str(source): args.expect_report_sha256, str(helper): sha(helper),
              str(Path(__file__).resolve()): sha(__file__)}
    points = []
    for row in data['rows']:
        path = ROOT / row['official_CSV']['file']
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            raw, columns = list(reader), reader.fieldnames
        require(columns is not None and len(columns) == len(set(columns)) == 276 and
                [item['seq'] for item in raw] == list(LENGTHS), 'Complete original CSV required')
        stored = {item['seq']: item for item in row['per_sequence47']}
        for original in raw:
            sequence = original['seq']
            values = {}
            for field in data['official_fields']:
                value = float(original[field['source_column']]) * field['scale_from_raw_detailed']
                require(math.isfinite(value) and value == stored[sequence][field['metric']],
                        'A plotted input differs from its original official CSV')
                values[field['metric']] = value
                if sequence == 'COMBINED':
                    require(value == row['combined47'][field['metric']], 'Combined official value changed')
            config = row['config']
            points.append(dict(id=row['id'], mode=config.get('mode', ''), cue=config['cue'],
                epoch=config.get('epoch', ''), intervention=config.get('intervention', ''),
                baseline=config.get('baseline', ''), sequence=sequence, frames=LENGTHS[sequence],
                **{field: values[field] for field in FIELDS},
                config_sha256=row['config_sha256'], official_CSV_sha256=row['official_CSV']['sha256']))
    for mode in MODES:
        for epoch in range(1, args.stage + 1):
            key = f'NP_{mode}_ir_e{epoch:02d}'
            require(lookup[key + '_roll']['config']['intervention'] == 'IR_image_spatial_roll' and
                    lookup[key + '_off']['config']['intervention'] == 'unavailable',
                    'Actual image interventions required at every snapshot')
    output, preview = args.output_directory.resolve(), args.preview_directory.resolve()
    require(not output.exists() and not preview.exists() and
            not output.is_relative_to(preview) and not preview.is_relative_to(output),
            'Use fresh, separate vector and PNG preview directories')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
        'svg.hashsalt': f'native-complete-stage{args.stage}-Dev-20261126'})
    indexed = {(row['id'], row['sequence']): row for row in points}
    epochs = list(range(1, args.stage + 1))
    figures = []

    def values(mode, cue, suffix='', sequence='COMBINED'):
        return [indexed[f'NP_{mode}_{cue}_e{epoch:02d}' + suffix, sequence]['HOTA']
                for epoch in epochs]

    output.mkdir(parents=True, exist_ok=False)
    preview.mkdir(parents=True, exist_ok=False)

    def save(fig, name, caption):
        vector, png = output / (name + '.svg'), preview / (name + '.png')
        fig.savefig(vector, bbox_inches='tight', metadata={'Date': None, 'Creator': 'IR-MFT complete Dev evidence'})
        fig.savefig(png, dpi=140, bbox_inches='tight')
        plt.close(fig)
        require(vector.stat().st_size < 2 * 1024**2 and png.stat().st_size < 2 * 1024**2,
                'Oversized scientific figure')
        figures.append(dict(name=name, caption=caption, SVG=str(vector), SVG_sha256=sha(vector),
            SVG_bytes=vector.stat().st_size, preview=str(png), preview_sha256=sha(png)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True, constrained_layout=True)
    for ax, mode in zip(axes, MODES):
        for cue in CUES:
            ax.plot(epochs, values(mode, cue), color=COLORS[cue], marker='o', label=cue.upper())
            key = own[mode + '_' + cue]
            ax.scatter([lookup[key]['config']['epoch']], [lookup[key]['combined47']['HOTA']],
                       color=COLORS[cue], marker='*', edgecolors='black', linewidths=.6, s=105, zorder=4)
        for key, label, style, color in [(FRESH, 'Fresh B1 RGB', ':', '#222222'),
                                        (CANONICAL, 'Canonical B4 RGB', '--', '#999999')]:
            ax.axhline(indexed[key, 'COMBINED']['HOTA'], linestyle=style, color=color,
                       linewidth=1, label=label)
        ax.set(title=mode.capitalize(), xlabel='Complete Train traversals', xticks=epochs)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Complete Dev HOTA (%)')
    handles, labels = axes[-1].get_legend_handles_labels()
    handles.append(Line2D([], [], marker='*', color='black', linestyle='None', markersize=9))
    labels.append('Original Dev choice')
    axes[-1].legend(handles, labels, fontsize=8, frameon=False,
                    loc='upper left', bbox_to_anchor=(1.02, 1))
    save(fig, 'all_six_normal_Dev_curves',
        f'All{6 * args.stage} normal complete-Dev checkpoints. Full-precision official combined values, '
        'not an average of video scores. Stars preserve the original rounded39 own-Dev choices; '
        'no budget or checkpoint selection is changed.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True, constrained_layout=True)
    for ax, mode in zip(axes, MODES):
        for suffix, label, color, marker in [('', 'Correct IR image', COLORS['ir'], 'o'),
                ('_roll', 'Spatially rolled IR image', '#CC79A7', 's'),
                ('_off', 'IR unavailable = fresh RGB', '#555555', '^')]:
            ax.plot(epochs, values(mode, 'ir', suffix), color=color, marker=marker, label=label)
        ax.set(title=mode.capitalize(), xlabel='Complete Train traversals', xticks=epochs)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Complete Dev HOTA (%)')
    axes[-1].legend(fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1))
    save(fig, 'all_IR_image_interventions',
        f'Both IR modes at all{args.stage} epochs, including all correct, wrong and missing-image controls. '
        'The source independent audit checks missing-modality CSVs and MOT against fresh RGB; '
        'this plotting run does not replay discarded detection packets or repeat MOT evaluation.')

    fig, axes = plt.subplots(2, 3, figsize=(12.8, 6.8), sharex=True, sharey=True, constrained_layout=True)
    for row_index, mode in enumerate(MODES):
        for ax, sequence in zip(axes[row_index], LENGTHS):
            ir = values(mode, 'ir', sequence=sequence)
            controls = [('rgb', '', 'IR minus RGB', COLORS['rgb'], 'o'),
                        ('gray', '', 'IR minus gray', COLORS['gray'], 's'),
                        ('ir', '_roll', 'IR minus rolled IR', '#CC79A7', '^')]
            for cue, suffix, label, color, marker in controls:
                control = values(mode, cue, suffix, sequence)
                ax.plot(epochs, [a - b for a, b in zip(ir, control)], color=color,
                        marker=marker, label=label)
            ax.axhline(0, color='#222222', linewidth=.8)
            ax.set(title=mode.capitalize() + ': ' + sequence, xticks=epochs)
            ax.grid(alpha=.2)
            if row_index == 1:
                ax.set_xlabel('Complete Train traversals')
        axes[row_index, 0].set_ylabel('Same-epoch HOTA difference (pp)')
    axes[0, -1].legend(fontsize=8, frameon=False)
    save(fig, 'matched_control_effects_by_Dev_video',
        'Same-mode, same-epoch IR-minus-RGB, IR-minus-gray and IR-minus-rolled-IR for both full Dev videos '
        'and the official combined result. All panels share the same vertical scale. '
        'Videos/checkpoints are not independent random repetitions. No confidence intervals, '
        'significance or isolated-module contribution is claimed.')

    table = output / 'plotted_complete_Dev_points.csv'
    with table.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(points)
    require(all(sha(path) == digest for path, digest in inputs.items()) and
            all(sha(ROOT / item['file']) == item['sha256'] for item in bindings),
            'Source evidence changed during plotting')
    metadata = dict(status='COMPLETE_NATIVE_FULL_DEV_STAGE_FIGURES_NOT_TEST_OR_INNOVATION_PROOF',
        stage=args.stage, source_hashes=inputs, source_CSVs=bindings, cases=len(lookup),
        per_sequence_points=len(points), official47_values_checked=47 * len(points),
        original_frozen_budget_decision=data['original_frozen_budget_decision'],
        complete_cases_including_negative_results=True, no_sequence_metric_averaging=True,
        original_checkpoint_or_budget_selection_changed=False, GPU_work=False, Test_results_read=False,
        new_inference_or_evaluation=False, new_accuracy_estimated=False,
        no_statistical_independence_or_significance_claim=True,
        plotted_table=dict(file=str(table), sha256=sha(table)), figures=figures,
        supported_innovations=0, goal_met=False)
    (output / 'figure_sources.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status=metadata['status'], actual_stage=args.stage,
        actual_cases=len(lookup), figures=len(figures), GPU_work=False, Test_results_read=False)), flush=True)


if __name__ == '__main__':
    main()
