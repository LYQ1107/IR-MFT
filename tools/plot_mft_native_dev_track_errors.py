#!/usr/bin/env python3
"""Plot every Dev GT identity from the completed, hash-bound CPU diagnosis."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-directory', required=True, type=Path)
    parser.add_argument('--output-directory', required=True, type=Path)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    audit_path = args.input_directory / 'complete.json'
    audit = json.loads(audit_path.read_text())
    assert audit['cases'] == 14 and audit['frames_per_case'] == 4834
    assert audit['official_19_alpha_fields_reproduced']
    data_path = args.input_directory / audit['identity_csv']['file']
    assert sha256(data_path) == audit['identity_csv']['sha256']
    with data_path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 294
    assert not args.output_directory.exists()
    args.output_directory.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = [('NP_quality_rgb_e02', 'Strongest RGB (quality, epoch 2)', '#5c6f91'),
               ('NP_temporal_ir_e06', 'Correct pseudoIR (temporal, epoch 6)', '#16857b'),
               ('NP_temporal_ir_e06_roll', 'Rolled pseudoIR (same epoch 6)', '#c77c45')]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.6), sharey=True)
    for ax, seq in zip(axes, audit['sequences']):
        ids = sorted({int(x['original_GT_ID']) for x in rows if x['sequence'] == seq})
        for offset, (case, label, color) in enumerate(methods):
            selected = {int(x['original_GT_ID']): x for x in rows
                        if x['case'] == case and x['sequence'] == seq}
            assert set(selected) == set(ids)
            ax.bar([i + (offset - 1) * .25 for i in range(len(ids))],
                   [100 * float(selected[g]['dominant_tracker_GT_coverage']) for g in ids],
                   width=.24, label=label, color=color)
        ax.set_title(f'{seq}: all {len(ids)} annotated identities', loc='left', fontsize=11)
        ax.set_xticks(range(len(ids)), [str(g) for g in ids])
        ax.set_xlabel('Original ground-truth identity (sequence local)')
        ax.set_ylabel('Dominant predicted-ID\ncoverage of GT detections (%)')
        ax.set_ylim(0, 105)
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.22)
        ax.spines[['right', 'top']].set_visible(False)
    fig.suptitle('Dev identity diagnosis at HOTA matching alpha = 0.5', fontsize=14, y=.985)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .945), ncol=3, fontsize=9)
    fig.text(.5, .015, 'All 21 GT identities shown. This coverage is diagnostic, not HOTA or IDF1; no Test data or model reselection.',
             ha='center', fontsize=9)
    fig.subplots_adjust(top=.86, bottom=.10, left=.09, right=.985, hspace=.48)
    svg = args.output_directory / 'all_Dev_identity_coverage.svg'
    png = svg.with_suffix('.png')
    fig.savefig(svg, metadata={'Date': None})
    fig.savefig(png, dpi=150)
    plt.close(fig)
    manifest = dict(status='COMPLETE_CPU_DEV_IDENTITY_FIGURE',
        audit_sha256=sha256(audit_path), identity_CSV_sha256=sha256(data_path),
        source_code_sha256=sha256(__file__),
        SVG=dict(file=svg.name, sha256=sha256(svg), bytes=svg.stat().st_size),
        private_visual_check_PNG=dict(file=png.name, sha256=sha256(png)),
        GT_IDs=21, Test_read=False, GPU_work=False, supported_innovations=0, goal_met=False)
    (args.output_directory / 'sources.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
