#!/usr/bin/env python3
"""Plot the completed frozen-primary Test report, without selecting any method."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.fusion_memory_evaluation import official_csv47
from tools.fusion_memory_observation import LENGTHS

ORDER = ['NP_source_canonical_RGB', 'NP_source_fresh_RGB', 'NP_temporal_ir_e06']
LABELS = ['Original RGB', 'Fresh B1 RGB', 'Frozen native IR']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--preview-directory', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output_directory.exists() and not args.preview_directory.exists()
    report = json.loads(args.report.read_text())
    assert report['status'] == 'COMPLETE_INDEPENDENT_NATIVE_PRIMARY_FULL7_TEST_DELIVERY_AUDIT'
    assert report['frames'] == 15985 and report['videos'] == 7 and report['cases'] == 3
    assert report['Test_used_for_selection'] is False and report['primary_IR'] == ORDER[-1]
    assert report['frozen_Test_written_before_original_production_controller_started'] is True
    schema = report['official_fields']
    rows = {row['id']: row for row in report['rows']}
    assert set(rows) == set(ORDER)
    sources, points = [], []
    for identifier in ORDER:
        row = rows[identifier]
        binding = row['official_CSV']
        path = ROOT / binding['file']
        assert sha(path) == binding['sha256']
        parsed = official_csv47(path, schema, 'test')
        assert parsed['combined47'] == row['combined47'] and parsed['per_sequence47'] == row['per_sequence47']
        sources.append(binding)
        points.extend(dict(id=identifier, seq=point['seq'], HOTA=point['HOTA']) for point in row['per_sequence47'])
    assert len(points) == 24
    baseline = rows[ORDER[0]]['combined47']['HOTA']
    assert baseline == report['original_RGB_baseline_HOTA'] and round(baseline, 3) == 29.529
    combined = [rows[key]['combined47']['HOTA'] for key in ORDER]
    names = list(LENGTHS['test'])
    original = {point['seq']:point['HOTA'] for point in rows[ORDER[0]]['per_sequence47']}
    native = {point['seq']:point['HOTA'] for point in rows[ORDER[-1]]['per_sequence47']}
    changes = [native[name] - original[name] for name in names]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'svg.fonttype':'none'})
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.4), gridspec_kw={'height_ratios':[1.2, 1]})
    fig.subplots_adjust(left=.085, right=.77, top=.87, bottom=.12, hspace=.62)
    fig.suptitle('Dev-frozen native IR system against the original RGB baseline', y=.965, fontsize=14)
    upper, lower = axes
    upper.bar(range(3), combined, color=['#737b83', '#8eabc5', '#258476'], width=.55, zorder=3)
    upper.set_xticks(range(3), LABELS)
    upper.set_ylabel('Official COMBINED HOTA')
    upper.set_ylim(0, max(baseline+13, max(combined)+4))
    upper.grid(axis='y', alpha=.2, zorder=0)
    upper.axhline(baseline+5, color='#be6233', linestyle='--', label=f'+5 target: {baseline+5:.3f}', zorder=2)
    upper.axhline(baseline+10, color='#5e65a6', linestyle=':', label=f'+10 target: {baseline+10:.3f}', zorder=2)
    for index, value in enumerate(combined):
        upper.text(index, value+.65, f'{value:.3f}', ha='center', fontsize=11)
    upper.legend(loc='upper left', bbox_to_anchor=(1.02, 1.02), frameon=False, fontsize=9)
    upper.set_title(f'Full-Test gain over original RGB: {combined[-1]-baseline:+.3f} HOTA points', loc='left', pad=14)
    lower.axhline(0, color='#555b60', linewidth=.9, zorder=2)
    lower.bar(range(7), changes, color=['#258476' if value >= 0 else '#bf5155' for value in changes], width=.65, zorder=3)
    lower.set_xticks(range(7), names)
    lower.set_ylabel('IR minus original RGB\n(absolute HOTA points)')
    lower.grid(axis='y', alpha=.2, zorder=0)
    bound = max(1.5, max(abs(value) for value in changes)*1.3)
    lower.set_ylim(-bound, bound)
    for index, value in enumerate(changes):
        lower.text(index, value + (.045*bound if value >= 0 else -.045*bound), f'{value:+.2f}',
                   ha='center', va='bottom' if value >= 0 else 'top', fontsize=9)
    lower.set_title('All seven complete videos; negative changes retained', loc='left', pad=13)
    fig.text(.085, .035, '7 full Test videos / 15,985 frames / seed 0. COMBINED is official, not a video-score mean.', fontsize=9)
    args.output_directory.mkdir(parents=True)
    args.preview_directory.mkdir(parents=True)
    svg = args.output_directory / 'full_Test_primary_and_per_video_changes.svg'
    preview = args.preview_directory / 'full_Test_primary_and_per_video_changes.png'
    fig.savefig(svg)
    fig.savefig(preview, dpi=170)
    plt.close(fig)
    csv_path = args.output_directory / 'plotted_Test_HOTA_points.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id', 'seq', 'HOTA'])
        writer.writeheader()
        writer.writerows(points)
    metadata = dict(status='COMPLETE_FIXED_PRIMARY_FULL_TEST_FIGURE_NOT_SELECTION_OR_INNOVATION_PROOF',
        report=str(args.report), report_sha256=sha(args.report), source_CSVs=sources,
        source_plot_script_sha256=sha(Path(__file__)), official_fields_rechecked=1128,
        points=24, videos=7, frames=15985, primary_id=ORDER[-1], original_baseline_id=ORDER[0],
        methods_selected_or_modified=False, new_TrackEval_or_NN_work=False,
        svg=dict(file=str(svg), sha256=sha(svg), bytes=svg.stat().st_size),
        private_PNG_preview=dict(file=str(preview), sha256=sha(preview), bytes=preview.stat().st_size),
        points_CSV=dict(file=str(csv_path), sha256=sha(csv_path), bytes=csv_path.stat().st_size))
    (args.output_directory / 'figure_sources.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(dict(status=metadata['status'], SVG=str(svg), private_preview=str(preview), points=24)), flush=True)


if __name__ == '__main__':
    main()
