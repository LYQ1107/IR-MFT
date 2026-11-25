#!/usr/bin/env python3
"""Plot completed six-arm evidence; preserve original Dev and Test selections."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs/fusion_research'
ARMS = ['rgb_111_s0', 'gray_111_s0', 'ir_111_s0', 'ir_011_s0', 'ir_101_s0', 'ir_110_s0']
LABELS = ['RGB (111)', 'Gray (111)', 'IR (111)', 'IR, S removed (011)',
          'IR, C removed (101)', 'IR, R removed (110)']
COLORS = ['#0072b2', '#777777', '#009e73', '#cc79a7', '#e69f00', '#d55e00']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--preview-directory', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Figure export is CPU only')
    started = time.monotonic()
    files = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        value = sha(path)
        if expected is not None and value != expected:
            raise ValueError('Original source changed: ' + str(path))
        if str(path) in files and files[str(path)] != value:
            raise ValueError('Source changed during export')
        files[str(path)] = value
        return value

    def load(name):
        path = DOCS / name
        bind(path)
        return read(path)

    ledger = load('PERCEPTUAL_MFT_20261123_FULL_TRAIN_TRAINING_LEDGER.json')
    choice = load('PERCEPTUAL_MFT_20261123_FULL_TRAIN_DEV_ANALYSIS.json')
    dev = load('PERCEPTUAL_MFT_20261125_FULL_TRAIN_DEV47_AUDIT.json')
    test = load('PERCEPTUAL_MFT_20261125_FULL_TRAIN_TEST47_AUDIT.json')
    bind(__file__)
    bind(ROOT / 'tools/train_structural_relative.py')
    if (dev['original_Dev_choice_sha256'] != sha(DOCS / 'PERCEPTUAL_MFT_20261123_FULL_TRAIN_DEV_ANALYSIS.json') or
            test['original_Dev_choice_sha256'] != dev['original_Dev_choice_sha256'] or
            test['Test_primary_reselection'] or choice['Test_accuracy_used_for_selection'] or
            dev['complete_cases'] != 50 or test['complete_cases'] != 13):
        raise ValueError('Original complete choices and audits must remain exact')
    hcolumn = next(s for s in dev['official_schema'] if s['metric'] == 'HOTA')
    # Public byte-exact CSV copies make reproduction independent of private work paths.
    for report, stage in [(dev, 'structural_full_train_dev'), (test, 'structural_full_train_test')]:
        for row in report['rows']:
            path = DOCS / 'perceptual_20261122_official_tables' / stage / (row['id']+'.csv')
            bind(path, row['official_CSV_sha256'])
            with path.open(newline='') as stream:
                raw = list(csv.DictReader(stream))
            observed = {r['seq']: float(r[hcolumn['source_column']]) * hcolumn['scale_from_raw_detailed']
                        for r in raw}
            for sequence in row['per_sequence47']:
                if observed[sequence['seq']] != sequence['HOTA']:
                    raise ValueError('A plotted sequence HOTA differs from the original CSV')
            if observed['COMBINED'] != row['combined47']['HOTA']:
                raise ValueError('A plotted official combined HOTA differs')
    histories = {r['id']: r for r in ledger['rows']}
    if set(histories) != set(ARMS):
        raise ValueError('All six original fits, including negative arms, are required')
    curves, epoch_loss = [], {}
    for key in ARMS:
        row = histories[key]
        if (row['fixed_seed'] != 0 or row['logical_Train_epochs'] != 3 or
                row['updates'] != 20736 or row['trainable_parameters'] != 145600 or
                len(row['history']) != 72):
            raise ValueError('Original common Train budget changed')
        for i, point in enumerate(row['history'], 1):
            if (point['epoch'] != i or point['steps'] != 288 or len(point['loss']) != 5 or
                    not all(math.isfinite(v) for v in point['loss'])):
                raise ValueError('The original five-field history is incomplete')
        curves.append(dict(fit_id=key, history=row['history']))
        epoch_loss[key] = [sum(p['loss'][0] for p in row['history'][i:i+24])/24 for i in [0, 24, 48]]
    devby = {r['id']: r for r in dev['rows']}
    testby = {r['id']: r for r in test['rows']}
    roles = test['roles']
    selected = {key: testby[roles[role]]['source_Dev_case'] for key, role in
                [('rgb_111_s0', 'matched_rgb'), ('gray_111_s0', 'matched_gray'), ('ir_111_s0', 'primary_IR')]}
    for row in dev['removals']:
        selected['ir_' + row['components'] + '_s0'] = row['Dev_best']
    normal = [r for r in dev['rows'] if not r['config'].get('baseline') and
              r['config'].get('intervention') is None]
    if len(normal) != 36:
        raise ValueError('All 6x3x2 normal Dev curves must be retained')
    devcurves = []
    for key in ARMS:
        cue, components, _ = key.split('_')
        points = [r for r in normal if r['config']['cue'] == cue and r['config']['components'] == components]
        if {(r['config']['epoch'], r['config']['policy']) for r in points} != {
                (e, p) for e in [24, 48, 72] for p in ['raw', 'priority']}:
            raise ValueError('A plotted Dev policy/snapshot is missing')
        devcurves.append(dict(fit_id=key, original_selected_id=selected[key],
            points=[dict(id=r['id'], logical_Train_epoch=r['config']['epoch']/24,
                         policy=r['config']['policy'], HOTA=r['combined47']['HOTA']) for r in points]))
    effects = []
    for row in test['removals']:
        original = next(r for r in dev['removals'] if r['mechanism'] == row['mechanism'])
        if row['original_Dev_effects'] != original:
            raise ValueError('Dev and Test removal identities disagree')
        effects.append(dict(mechanism=row['mechanism'],
            Dev_paired=original['primary_minus_paired_HOTA'],
            Test_paired=row['primary_minus_paired_Test_HOTA'],
            Dev_own_best=original['primary_minus_removal_own_best_HOTA'],
            Test_own_best=row['primary_minus_removal_own_Dev_best_Test_HOTA']))
    sequences = []
    for sequence in [*test['sequences'], 'COMBINED']:
        values = {role: next(p['HOTA'] for p in testby[roles[role]]['per_sequence47'] if p['seq'] == sequence)
                  for role in ['primary_IR', 'matched_rgb', 'matched_gray']}
        sequences.append(dict(sequence=sequence, HOTA=values,
            IR_minus_RGB=values['primary_IR']-values['matched_rgb'],
            IR_minus_gray=values['primary_IR']-values['matched_gray']))
    if (sequences[-1]['IR_minus_RGB'] != test['primary_minus_matched_RGB_Test_HOTA'] or
            sequences[-1]['IR_minus_gray'] != test['primary_minus_matched_gray_Test_HOTA']):
        raise ValueError('Official combined effects differ')
    output, preview = args.output_directory, args.preview_directory
    output.mkdir(exist_ok=False)
    preview.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'svg.fonttype': 'none', 'svg.hashsalt': 'ir-mft-full-train-20261125'})
    figures = []

    def save(fig, name):
        vector, png = output / (name+'.svg'), preview / (name+'.png')
        fig.savefig(vector, bbox_inches='tight', metadata={'Date': None, 'Creator': 'IR-MFT evidence exporter'})
        fig.savefig(png, dpi=140, bbox_inches='tight')
        plt.close(fig)
        figures.append(dict(name=name, vector=str(vector), vector_sha256=sha(vector),
                            vector_bytes=vector.stat().st_size, review_preview=str(png)))

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.0), sharex=True, layout='constrained')
    for key, label, color in zip(ARMS, LABELS, COLORS):
        points = histories[key]['history']
        for ax, index in zip(axes, [0, 4]):
            ax.plot([p['epoch']/24 for p in points], [p['loss'][index] for p in points],
                    label=label, color=color, linewidth=1.25)
    axes[0].set(title='Completed structural family: all 432 recorded training macros', ylabel='Mean total training loss')
    axes[1].set(ylabel='Mean pre-clipping gradient L2 norm', xlabel='Complete Train traversals (24 macros each)')
    for ax in axes:
        ax.grid(alpha=.2)
        for x in [1, 2]:
            ax.axvline(x, color='#444444', alpha=.25, linewidth=.8)
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].set_xticks([0, .5, 1, 1.5, 2, 2.5, 3])
    save(fig, 'training_loss_and_gradient')

    fig, axes = plt.subplots(2, 3, figsize=(11.7, 6.1), sharex=True, sharey=True, layout='constrained')
    baseline = devby[testby[roles['source_RGB_fresh']]['source_Dev_case']]['combined47']['HOTA']
    for ax, curve, label in zip(axes.flat, devcurves, LABELS):
        for policy, color, style in [('raw', '#0072b2', '-'), ('priority', '#e69f00', '--')]:
            points = sorted((p for p in curve['points'] if p['policy'] == policy), key=lambda p: p['logical_Train_epoch'])
            ax.plot([p['logical_Train_epoch'] for p in points], [p['HOTA'] for p in points],
                    color=color, linestyle=style, marker='o', markersize=4, label=policy)
        selectedpoint = next(p for p in curve['points'] if p['id'] == curve['original_selected_id'])
        ax.scatter(selectedpoint['logical_Train_epoch'], selectedpoint['HOTA'], marker='*', s=150,
                   color='#222222', zorder=5, label='original Dev choice')
        ax.axhline(baseline, color='#777777', linestyle=':', linewidth=1, label='fresh source RGB')
        ax.set(title=label, xticks=[1, 2, 3])
        ax.grid(alpha=.2)
    for ax in axes[:, 0]:
        ax.set_ylabel('Complete Dev HOTA')
    for ax in axes[-1]:
        ax.set_xlabel('Complete Train traversals')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=4, loc='outside lower center', frameon=False)
    fig.suptitle('All 36 normal Dev points; stars preserve the original choices', fontsize=12)
    save(fig, 'complete_dev_curves')

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True, layout='constrained')
    for ax, suffix, title in zip(axes, ['paired', 'own_best'],
                                ['Same macro and policy', 'Each removal\'s own Dev choice']):
        for shift, split, color in [(-.17, 'Dev', '#0072b2'), (.17, 'Test', '#e69f00')]:
            bars = ax.bar([i+shift for i in range(3)], [e[split+'_'+suffix] for e in effects],
                          width=.32, color=color, label=split)
            ax.bar_label(bars, fmt='%+.2f', padding=3, fontsize=9)
        ax.axhline(0, color='#333333', linewidth=.8)
        ax.set(title=title, xticks=range(3), xticklabels=['S', 'C', 'R'])
        ax.grid(axis='y', alpha=.2)
        ax.legend()
    axes[0].set_ylabel('HOTA points: full IR minus removal')
    axes[0].set_ylim(-2.3, 2.3)
    fig.suptitle('Conditional removal effects; effects are not additive', fontsize=12)
    save(fig, 'conditional_component_effects')

    fig, ax = plt.subplots(figsize=(10.5, 5.6), layout='constrained')
    for shift, key, color, label in [(-.17, 'IR_minus_RGB', '#0072b2', 'IR minus matched RGB'),
                                    (.17, 'IR_minus_gray', '#e69f00', 'IR minus matched gray')]:
        bars = ax.barh([i+shift for i in range(8)], [p[key] for p in sequences],
                       height=.32, color=color, label=label)
        ax.bar_label(bars, fmt='%+.2f', padding=4, fontsize=9)
    ax.axvline(0, color='#333333', linewidth=.8)
    ax.axhline(6.5, color='#555555', linestyle=':', linewidth=1)
    ax.set_yticks(range(8), [p['sequence'] if p['sequence'] != 'COMBINED' else 'COMBINED (official)' for p in sequences])
    ax.invert_yaxis()
    ax.set(xlabel='Complete Test HOTA difference (absolute points)', xlim=(-4.7, 6.3),
           title='Frozen full-Train primary: all 7 Test videos and official combination')
    ax.grid(axis='x', alpha=.2)
    ax.legend(loc='lower right')
    save(fig, 'per_sequence_test_effects')
    if any(sha(p) != h for p, h in files.items()):
        raise ValueError('A completed source changed while rendering')
    report = dict(status='COMPLETE_SOURCE_CHECKED_FOUR_FULL_TRAIN_EVIDENCE_FIGURES',
        source_hashes=files, matplotlib_version=matplotlib.__version__, figures=figures,
        raw_training_rows=432, normal_Dev_points=36, all_Dev_cases_bound=50, all_Test_cases_bound=13,
        actual_full_Train_epochs_each=3, optimizer_updates_each=20736, raw_loss_column_order=
            ['total_loss', 'supervised_semantic_loss', 'reliable_RGB_loss', 'wrong_auxiliary_loss', 'preclip_gradient_norm'],
        training_curves=curves, mean_total_loss_per_24_macro_traversal=epoch_loss,
        complete_Dev_curves=devcurves, conditional_effects=effects, per_sequence_Test_effects=sequences,
        original_Dev_choices=selected, no_new_selection=True, fixed_seed=0, additional_seed_work=False,
        new_fit=False, new_NN_or_GPU_work=False, new_Dev_or_Test_accuracy=False,
        original_primary_Test_HOTA=test['primary_Test_HOTA'], target_Test_HOTA=34.529,
        goal_met=False, supported_innovations=0, seconds=time.monotonic()-started,
        peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        scope='All original six fits/432macro records and normal36Dev points,original selections only. '
              'HOTA checked against all63unchanged detailed CSVs.All seven Test videos and official COMBINED retained. '
              'No smoothing,error bars,significance claims,Test model reselection or inference for the new native family. '
              'SVG vectors only for publication;PNG previews are local review artifacts,not dataset caches.')
    if report['peak_process_RSS_GiB'] > 8:
        raise RuntimeError('Figure process exceeded8GiB')
    (output / 'complete.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], figures=4, source_files=len(files),
                         seconds=report['seconds'], peak_process_RSS_GiB=report['peak_process_RSS_GiB'])))


if __name__ == '__main__':
    main()
