#!/usr/bin/env python3
"""Render all minimum-six-epoch native Dev curves from completed raw evidence.

No inference, fitting, Test result reads, selection, sequence averaging, or
significance claims. SVGs are small publication artifacts; PNGs stay separate.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs/fusion_research'
REPORT = DOCS / 'PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV.json'
CONFIGS = DOCS / 'PERCEPTUAL_MFT_20261126_NATIVE_FULL_DEV_CONFIGS_REGISTERED.json'
SEQUENCES = {'BT-005': 1834, 'SN-013': 3000, 'COMBINED': 4834}
FRESH, CANONICAL = 'NP_source_fresh_RGB', 'NP_source_canonical_RGB'
COLORS = {'rgb': '#0072B2', 'gray': '#777777', 'ir': '#D55E00'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready():
    if not REPORT.exists():
        return dict(status='WAITING_FOR_COMPLETE_NATIVE_MINIMUM6_FULL_DEV_AUDIT',
                    complete_report_exists=False, figures_created=False,
                    GPU_work=False, Test_results_read=False, new_accuracy_claimed=False)
    report = read(REPORT)
    require(report['status'] == 'COMPLETE_INDEPENDENT_MINIMUM6_FULL_DEV_AUDIT_NOT_FINAL_STUDY' and
            report['stage'] == 6 and report['split'] == 'dev' and report['frames'] == 4834 and
            report['all62_cases_included'] is True and report['Test_results_read'] is False,
            'Require the original complete62-case minimum6 full Dev audit')
    return dict(status='READY_COMPLETE_NATIVE_DEV_REQUIRES_RAW_CSV_FIGURE_CHECKS',
                complete_report_exists=True, figures_created=False,
                GPU_work=False, Test_results_read=False, new_accuracy_claimed=False)


def source_data():
    require(ready()['complete_report_exists'], 'No figures before actual complete native Dev evidence')
    report, published = read(REPORT), read(CONFIGS)
    require(report['prepublished_config_manifest_sha256'] == sha(CONFIGS), 'Prepublished case bindings changed')
    schema = report['official_fields']
    require(len(schema) == len({field['metric'] for field in schema}) == 47,
            'Require all47 official fields')
    rows = report['rows']
    lookup = {row['id']: row for row in rows}
    require(len(rows) == len(lookup) == 62 and set(lookup) == {row['id'] for row in published['cases']},
            'Missing or extra full Dev case')
    points, references = [], []
    for reference in published['cases']:
        key = reference['id']
        row, case = lookup[key], lookup[key]['config']
        require(row['config_sha256'] == reference['sha256'] == sha(ROOT / reference['file']) and
                case == read(ROOT / reference['file']) and case['split'] == 'dev' and case['seed'] == 0,
                'Original case changed')
        path = ROOT / row['official_CSV']['file']
        require(sha(path) == row['official_CSV']['sha256'], 'Original published CSV changed')
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            raw, columns = list(reader), reader.fieldnames
        require(columns is not None and len(columns) == len(set(columns)) == 276 and
                [item['seq'] for item in raw] == list(SEQUENCES), 'Incomplete original CSV')
        stored = {item['seq']: item for item in row['per_sequence47']}
        require(len(row['per_sequence47']) == len(stored) == 3 and set(stored) == set(SEQUENCES),
                'Incomplete stored per-sequence47 values')
        for original in raw:
            sequence = original['seq']
            values = {}
            for field in schema:
                value = float(original[field['source_column']]) * field['scale_from_raw_detailed']
                require(math.isfinite(value) and value == stored[sequence][field['metric']],
                        'A plotted source differs from its original official CSV')
                values[field['metric']] = value
                if sequence == 'COMBINED':
                    require(value == row['combined47'][field['metric']], 'Combined official value changed')
            require(values['CLR_Frames'] == SEQUENCES[sequence], 'Partial Dev video is not plottable')
            points.append(dict(id=key, mode=case.get('mode', ''), cue=case['cue'],
                epoch=case.get('epoch', ''), intervention=case.get('intervention', ''),
                baseline=case.get('baseline', ''), sequence=sequence, frames=SEQUENCES[sequence],
                **{name: values[name] for name in ['HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW']},
                config_sha256=row['config_sha256'], official_CSV_sha256=sha(path)))
        references.append(dict(file=str(path.relative_to(ROOT)), sha256=sha(path)))
    expected = {(mode, cue, epoch) for mode in ['quality', 'temporal']
                for cue in ['rgb', 'gray', 'ir'] for epoch in range(1, 7)}
    normal = [row['config'] for row in rows if not row['config'].get('baseline') and
              row['config']['intervention'] == 'normal']
    require(len(normal) == 36 and {(case['mode'], case['cue'], case['epoch']) for case in normal} == expected,
            'Every normal mode/cue/epoch is required, including negative results')
    for mode in ['quality', 'temporal']:
        for epoch in range(1, 7):
            key = f'NP_{mode}_ir_e{epoch:02d}'
            require(lookup[key + '_roll']['config']['intervention'] == 'IR_image_spatial_roll' and
                    lookup[key + '_off']['config']['intervention'] == 'unavailable' and
                    lookup[key + '_off']['combined47'] == lookup[FRESH]['combined47'] and
                    lookup[key + '_off']['per_sequence47'] == lookup[FRESH]['per_sequence47'],
                    'Actual wrong/missing-image interventions are required at every snapshot')
    return report, points, references


def export(output, preview):
    report, points, references = source_data()
    source_hashes = {str(REPORT): sha(REPORT), str(CONFIGS): sha(CONFIGS), str(Path(__file__)): sha(__file__)}
    output, preview = output.resolve(), preview.resolve()
    require(output != preview and not output.exists() and not preview.exists(),
            'Preserve earlier figure directories and keep PNG previews separate')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
        'svg.hashsalt': 'native-minimum6-complete-Dev-20261126'})
    indexed = {(row['id'], row['sequence']): row for row in points}
    epochs, figures = list(range(1, 7)), []

    def values(mode, cue, suffix='', sequence='COMBINED'):
        return [indexed[f'NP_{mode}_{cue}_e{epoch:02d}' + suffix, sequence]['HOTA'] for epoch in epochs]

    output.mkdir(parents=True, exist_ok=False)
    preview.mkdir(parents=True, exist_ok=False)

    def save(fig, name, caption):
        vector, png = output / (name + '.svg'), preview / (name + '.png')
        fig.savefig(vector, bbox_inches='tight', metadata={'Date': None, 'Creator': 'IR-MFT complete Dev evidence'})
        fig.savefig(png, dpi=140, bbox_inches='tight')
        plt.close(fig)
        require(vector.stat().st_size < 2*1024**2 and png.stat().st_size < 2*1024**2,
                'Oversized scientific figure')
        figures.append(dict(name=name, caption=caption, SVG=str(vector), SVG_sha256=sha(vector),
            SVG_bytes=vector.stat().st_size, preview=str(png), preview_sha256=sha(png)))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4), sharey=True, constrained_layout=True)
    for ax, mode in zip(axes, ['quality', 'temporal']):
        for cue in ['rgb', 'gray', 'ir']:
            ax.plot(epochs, values(mode, cue), color=COLORS[cue], marker='o', label=cue.upper())
        for key, label, style, color in [(FRESH, 'Fresh B1 RGB', ':', '#222222'),
                                        (CANONICAL, 'Canonical B4 RGB', '--', '#999999')]:
            ax.axhline(indexed[key, 'COMBINED']['HOTA'], linestyle=style, color=color, linewidth=1, label=label)
        ax.set(title=mode.capitalize(), xlabel='Complete Train traversals', xticks=epochs)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Complete Dev HOTA (%)')
    axes[-1].legend(fontsize=8, frameon=False)
    save(fig, 'all_six_normal_Dev_curves',
        'All36 normal complete-Dev checkpoints. Full-precision combined CSV values, not an average of video scores. '
        'The original budget decision uses its registered rounded39-field summaries.')

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4), sharey=True, constrained_layout=True)
    for ax, mode in zip(axes, ['quality', 'temporal']):
        for suffix, label, color, marker in [('', 'Correct IR image', COLORS['ir'], 'o'),
                ('_roll', 'Spatially rolled IR image', '#CC79A7', 's'),
                ('_off', 'IR unavailable = fresh RGB', '#555555', '^')]:
            ax.plot(epochs, values(mode, 'ir', suffix), color=color, marker=marker, label=label)
        ax.set(title=mode.capitalize(), xlabel='Complete Train traversals', xticks=epochs)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Complete Dev HOTA (%)')
    axes[-1].legend(fontsize=8, frameon=False)
    save(fig, 'all_IR_image_interventions',
        'Both IR modes at all six epochs, including wrong-image and unavailable-image controls. '
        'Unavailable curves are byte-checked against fresh RGB in the source audit; no favorable snapshot is omitted.')

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.5), sharex=True, constrained_layout=True)
    for row_index, mode in enumerate(['quality', 'temporal']):
        for ax, sequence in zip(axes[row_index], SEQUENCES):
            ir = values(mode, 'ir', sequence=sequence)
            for cue, marker in [('rgb', 'o'), ('gray', 's')]:
                control = values(mode, cue, sequence=sequence)
                ax.plot(epochs, [x-y for x, y in zip(ir, control)], color=COLORS[cue],
                        marker=marker, label='IR minus ' + cue.upper())
            ax.axhline(0, color='#222222', linewidth=.8)
            ax.set(title=mode.capitalize() + ': ' + sequence, xticks=epochs)
            ax.grid(alpha=.2)
            if row_index == 1:
                ax.set_xlabel('Complete Train traversals')
        axes[row_index, 0].set_ylabel('Paired-snapshot HOTA difference (pp)')
    axes[0, -1].legend(fontsize=8, frameon=False)
    save(fig, 'matched_control_effects_by_Dev_video',
        'Same-mode, same-epoch IR-minus-RGB and IR-minus-gray differences for each full Dev video and the official '
        'combined result. Videos/checkpoints are not independent repetitions; no confidence interval or significance '
        'claim is made. These comparisons are not single-module ablations.')

    table = output / 'plotted_complete_Dev_points.csv'
    with table.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(points)
    for path, digest in source_hashes.items():
        require(sha(path) == digest, 'Input changed during figure export')
    for reference in references:
        require(sha(ROOT / reference['file']) == reference['sha256'], 'Raw CSV changed during figure export')
    metadata = dict(status='COMPLETE_MINIMUM6_FULL_DEV_FIGURES_NOT_TEST_COMPLETION',
        source_hashes=source_hashes, source_CSVs=references, cases=62, per_sequence_points=len(points),
        original_frozen_budget_decision=report['original_frozen_budget_decision'],
        complete_cases_including_negative_results=True, no_sequence_metric_averaging=True,
        original_checkpoint_or_budget_selection_changed=False, GPU_work=False, Test_results_read=False,
        new_inference_or_evaluation=False, new_accuracy_estimated=False,
        no_statistical_independence_or_significance_claim=True,
        plotted_table=dict(file=str(table), sha256=sha(table)), figures=figures,
        supported_innovations=0, goal_met=False)
    with (output / 'figure_sources.json').open('x') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    lines = ['# Native pyramid: minimum-six-epoch complete Dev', '',
        'All62 registered cases; both complete Dev videos,4834 frames. These views do not change the original '
        '6/9/12 budget decision or final selection and do not establish Test gains or independent innovations.', '']
    for figure in figures:
        lines += ['![' + figure['name'] + '](' + figure['name'] + '.svg)', '', figure['caption'], '']
    lines += ['Full precision plotted values are retained in `plotted_complete_Dev_points.csv`; all raw276-column '
        'official CSV bindings and input hashes are retained in `figure_sources.json`. '
        'Historical Test exposure, RGB pretraining seeing Dev and RGB-derived pseudoIR remain disclosed.', '']
    with (output / 'README.md').open('x') as handle:
        handle.write('\n'.join(lines))
    print(json.dumps(dict(status=metadata['status'], figures=len(figures), cases=62,
        per_sequence_points=len(points), output_directory=str(output), preview_directory=str(preview),
        supported_innovations=0, goal_met=False)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check-ready', action='store_true')
    action.add_argument('--export', action='store_true')
    parser.add_argument('--output-directory', type=Path)
    parser.add_argument('--preview-directory', type=Path)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit CPU-only environment required')
    if args.check_ready:
        print(json.dumps(ready()), flush=True)
    else:
        require(args.output_directory is not None and args.preview_directory is not None,
                'Separate SVG and preview directories are required')
        export(args.output_directory, args.preview_directory)


if __name__ == '__main__':
    main()
