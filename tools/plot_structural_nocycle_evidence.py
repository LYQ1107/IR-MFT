#!/usr/bin/env python3
"""Plot complete frozen families; never select a model using Test accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT/'docs/fusion_research'
STEM = 'PERCEPTUAL_MFT_20261123_STRUCTURAL_NOCYCLE'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_bars(ax, first, second, labels, names):
    x = np.arange(len(labels))
    for offset, values, name, color in [(-.18, first, names[0], '#277da8'),
                                         (.18, second, names[1], '#a74440')]:
        bars = ax.bar(x+offset, values, .34, label=name, color=color)
        ax.bar_label(bars, labels=[f'{v:+.3f}' for v in values], padding=3, fontsize=8)
    low, high = min([0]+list(first)+list(second)), max([0]+list(first)+list(second))
    margin = max(1., (high-low)*.24)
    ax.set_ylim(low-margin, high+margin)
    ax.set_xticks(x, labels)
    ax.axhline(0, color='.4', lw=1)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)


def save(fig, suffix, paths):
    output = DOCS/(STEM+suffix+'.png')
    sources = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    fig.savefig(output, dpi=130, metadata={'Description': json.dumps(sources, sort_keys=True)})
    plt.close(fig)
    assert output.stat().st_size <= 512*1024
    record = dict(path=str(output), bytes=output.stat().st_size, source_sha256=sources,
                  plot_source_sha256=digest(Path(__file__)), accuracy_selection=False)
    output.with_suffix('.figure.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


def dev_plot(path):
    dev = json.loads(path.read_text())
    assert dev['status'] == 'COMPLETE_FIXED_SEED0_STRUCTURAL_NOCYCLE_DEV'
    assert len(dev['all_new_rows']) == 36 and len(dev['all_retained_reference_rows']) == 17
    primary, base = dev['primary'], dev['baseline']
    assert primary['id'] == 'SR_ir_no_cycle_s0_e06_raw'
    rows = [dev['best_matched_controls'][c] for c in ['rgb', 'gray']]+[primary]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), layout='constrained')
    delta = [r['metrics']['HOTA']-base['metrics']['HOTA'] for r in rows]
    bars = axes[0].bar(['RGB repeat', 'Gray', 'Actual new IR'], delta, color=['#999999', '#cccccc', '#277da8'])
    axes[0].bar_label(bars, labels=[f'{r["metrics"]["HOTA"]:.3f}\n({v:+.3f})' for r, v in zip(rows, delta)], padding=4)
    axes[0].axhline(0, color='.4', lw=1)
    axes[0].set_ylim(min(0, min(delta))-1, max(0, max(delta))+1.6)
    axes[0].set_ylabel('HOTA change from original RGB Dev47.579')
    axes[0].set_title('Complete matched controls; fixed primary IR6/raw')
    paired_bars(axes[1], [r['primary_minus_paired_HOTA'] for r in dev['removals']],
                [r['primary_minus_best_removal_HOTA'] for r in dev['removals']],
                ['Structure', 'Centering', 'Feature risk'], ['Same6/raw removal', 'Best removal curve'])
    axes[1].set_title('Primary IR minus removal: positive favors the module')
    axes[1].set_ylabel('HOTA difference')
    fig.suptitle('All4834Devframes:36new evaluations +17reused references; fixed seed0\n'
                 'Dev evidence alone does not establish Test gains or originality', fontsize=11)
    save(fig, '_DEV_EVIDENCE', [path])


def test_plot(dev_path, test_path):
    dev, test = [json.loads(p.read_text()) for p in [dev_path, test_path]]
    assert test['status'] == 'COMPLETE_FROZEN_SEED0_STRUCTURAL_NOCYCLE_ALL7_TEST'
    spec_path = ROOT/'configs/perceptual_structural_nocycle_test_20261123.json'
    spec = json.loads(spec_path.read_text())
    assert spec['complete_Dev_analysis_sha256'] == digest(dev_path)
    assert test['specification_sha256'] == digest(spec_path)
    rows = {r['id']: r for r in test['rows']}
    roles = test['roles']
    primary, baseline = rows[test['primary']], rows[roles['source_RGB_canonical']]
    assert len(primary['per_sequence']) == 7
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), layout='constrained')
    drows = [dev['best_matched_controls'][c] for c in ['rgb', 'gray']]+[dev['primary']]
    trows = [rows[roles['matched_'+c]] for c in ['rgb', 'gray']]+[primary]
    paired_bars(axes[0], [r['metrics']['HOTA']-dev['baseline']['metrics']['HOTA'] for r in drows],
                [r['metrics']['HOTA']-baseline['metrics']['HOTA'] for r in trows],
                ['RGB repeat', 'Gray', 'Actual IR'], ['Dev', 'Frozen Test'])
    axes[0].axhline(5, ls='--', color='.45', lw=1)
    low, high = axes[0].get_ylim()
    axes[0].set_ylim(low, max(5.8, high))
    axes[0].set_title('Same Dev-selected cases in both splits')
    axes[0].set_ylabel('HOTA change from original RGB in each split')
    seqs = list(primary['per_sequence'])
    changes = [primary['per_sequence'][s]['HOTA']-baseline['per_sequence'][s]['HOTA'] for s in seqs]
    bars = axes[1].barh(seqs, changes, color=['#277da8' if v > 0 else '#a74440' for v in changes])
    axes[1].bar_label(bars, labels=[f'{v:+.3f}' for v in changes], padding=3, fontsize=8)
    axes[1].axvline(0, color='.4', lw=1)
    axes[1].invert_yaxis()
    axes[1].set_xlim(min(0, min(changes))-1.5, max(0, max(changes))+1.5)
    axes[1].set_xlabel('Primary IR minus RGB HOTA')
    axes[1].set_title('Every Test sequence retained')
    paired_bars(axes[2], [r['primary_minus_paired_HOTA'] for r in test['removals']],
                [r['primary_minus_Dev_best4_HOTA'] for r in test['removals']],
                ['Structure', 'Centering', 'Feature risk'], ['Same6/raw removal', 'Dev-selected removal'])
    axes[2].set_title('Frozen Test module contribution')
    axes[2].set_ylabel('Primary IR minus removal HOTA')
    fig.suptitle(f"Full15985-frame Test: fixed IR{primary['metrics']['HOTA']:.3f}, "
                 f"RGB{baseline['metrics']['HOTA']:.3f}, change{test['primary_minus_canonical_RGB_HOTA']:+.3f}\n"
                 'Historical Test is exposed; no Test model reselection or additional seeds', fontsize=11)
    save(fig, '_DEV_TEST_EVIDENCE', [dev_path, test_path, spec_path])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    paths = [DOCS/(STEM+suffix+'.json') for suffix in ['_ANALYSIS', '_TEST']]
    work = Path(json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())['work_root'])
    source_hash = digest(Path(__file__))
    for path, state_name in zip(paths, ['structural_nocycle_dev_state.json', 'structural_nocycle_test_state.json']):
        while not path.exists():
            if not args.wait:
                raise FileNotFoundError(path)
            state_path = work/state_name
            if state_path.exists() and 'FAIL' in json.loads(state_path.read_text())['status']:
                raise RuntimeError('Retain source failure; no partial-family accuracy plotting')
            assert digest(Path(__file__)) == source_hash
            time.sleep(5)
        if path == paths[0]:
            dev_plot(path)
        else:
            test_plot(*paths)


if __name__ == '__main__':
    main()
