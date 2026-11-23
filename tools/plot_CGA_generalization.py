#!/usr/bin/env python3
"""Show complete frozen CGA Dev/Test evidence without selecting a Test winner."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT/'docs/fusion_research'


def main():
    dev_path = DOCS/'PERCEPTUAL_MFT_20261123_SINGLE_SEED_GEOMETRY_ANALYSIS.json'
    test_path = DOCS/'PERCEPTUAL_MFT_20261123_CGA_FORMAL_TEST.json'
    spec = json.loads((ROOT/'configs/perceptual_CGA_formal_test_20261123.json').read_text())
    assert hashlib.sha256(dev_path.read_bytes()).hexdigest() == spec['source_Dev_analysis_sha256']
    dev = json.loads(dev_path.read_text())
    test = json.loads(test_path.read_text())
    assert test['status'] == 'COMPLETE_FROZEN_SEED0_CGA_ALL7_TEST'
    rows = {r['id']: r for r in test['rows']}
    baseline_test = rows['CGA_source_RGB_canonical']
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    baseline_dev_HOTA = plan['reference']['rgb_dev_HOTA']
    cues = ['rgb', 'gray', 'ir']
    dev_delta, test_delta = [], []
    for cue in cues:
        t = rows[f'CGA_{cue}_Dev_selected']
        d = dev['best_by_cue'][cue]
        role = t['config']['head_role']
        assert t['config']['output_policy'] == 'raw'
        # The three original readouts reuse TC results; GFX adds the two
        # crossed geometries. Their full-stream parity was frozen before Test.
        expected = f"TC_{cue}_{'none' if role == 'full' else role}_raw"
        assert d['id'] == expected
        dev_delta.append(d['metrics']['HOTA']-baseline_dev_HOTA)
        test_delta.append(t['metrics']['HOTA']-baseline_test['metrics']['HOTA'])
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5), layout='constrained')
    x = np.arange(3)
    for offset, values, label, color in [(-.18, dev_delta, 'Dev', '#277da8'),
                                         (.18, test_delta, 'Frozen Test', '#a74440')]:
        bars = axes[0].bar(x+offset, values, .34, label=label, color=color)
        axes[0].bar_label(bars, labels=[f'{v:+.3f}' for v in values], padding=3, fontsize=9)
    axes[0].axhline(0, color='.35', lw=1)
    axes[0].set_xticks(x, ['RGB repeat', 'Gray', 'Actual new IR'])
    axes[0].set_ylabel('HOTA change from original RGB in each split')
    axes[0].set_title('Same Dev-selected cases; no Test reselection')
    axes[0].legend()
    axes[0].set_ylim(-1.1, 3.7)
    primary = rows[test['primary']]
    seqs = list(primary['per_sequence'])
    changes = [primary['per_sequence'][s]['HOTA']-baseline_test['per_sequence'][s]['HOTA'] for s in seqs]
    bars = axes[1].barh(seqs, changes, color=['#277da8' if v > 0 else '#a74440' for v in changes])
    axes[1].bar_label(bars, labels=[f'{v:+.3f}' for v in changes], padding=3, fontsize=9)
    axes[1].axvline(0, color='.35', lw=1)
    axes[1].invert_yaxis()
    axes[1].set_xlim(-3.8, 6.2)
    axes[1].set_xlabel('Primary IR minus original RGB HOTA')
    axes[1].set_title('All7 Test sequences; combined change -0.098')
    for ax in axes:
        ax.grid(axis='x' if ax == axes[1] else 'y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Frozen seed0 CGA: full15985-frame Test HOTA29.431 vs RGB29.529\n'
                 'Dev baseline47.579; Test baseline29.529; no new seed runs or supported innovation claim', fontsize=11)
    path = DOCS/'PERCEPTUAL_MFT_20261123_CGA_DEV_TEST_GENERALIZATION.png'
    fig.savefig(path, dpi=130, metadata=dict(Description='Complete Dev/Test reports: '+
        hashlib.sha256(dev_path.read_bytes()).hexdigest()+' / '+hashlib.sha256(test_path.read_bytes()).hexdigest()))
    plt.close(fig)
    assert path.stat().st_size <= 512*1024
    print(json.dumps(dict(path=str(path), bytes=path.stat().st_size)))


if __name__ == '__main__':
    main()
