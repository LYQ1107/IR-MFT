#!/usr/bin/env python3
"""Describe paired Dev identity errors; never fit or select a model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def summarize(rows):
    positive = sum(max(0, x['delta_dominant_tracker_matches']) for x in rows)
    largest = max(rows, key=lambda x: x['delta_dominant_tracker_matches'])
    return dict(
        GT_IDs=len(rows),
        better_dominant_GT_coverage=sum(x['delta_dominant_tracker_matches'] > 0 for x in rows),
        worse_dominant_GT_coverage=sum(x['delta_dominant_tracker_matches'] < 0 for x in rows),
        equal_dominant_GT_coverage=sum(x['delta_dominant_tracker_matches'] == 0 for x in rows),
        delta_total_dominant_tracker_matches=sum(x['delta_dominant_tracker_matches'] for x in rows),
        delta_total_HOTA_matched_GT_detections=sum(x['delta_matched_GT_detections'] for x in rows),
        sum_positive_dominant_match_changes=positive,
        largest_positive_change=dict(sequence=largest['sequence'], GT_ID=largest['GT_ID'],
            delta_dominant_tracker_matches=largest['delta_dominant_tracker_matches'],
            fraction_of_positive_dominant_match_changes=(
                max(0, largest['delta_dominant_tracker_matches']) / positive if positive else None)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-directory', required=True, type=Path)
    parser.add_argument('--execution-receipt', required=True, type=Path)
    parser.add_argument('--output-json', required=True, type=Path)
    args = parser.parse_args()
    audit_path = args.input_directory / 'complete.json'
    audit = read(audit_path)
    assert audit['status'] == 'COMPLETE_CPU_DEV_OFFICIAL_MATCH_IDENTITY_DIAGNOSTIC'
    assert audit['cases'] == 14 and audit['frames_per_case'] == 4834
    assert not audit['Test_results_read'] and not audit['Test_GT_read']
    receipt = read(args.execution_receipt)
    assert receipt['actual_CPU_child_exit_code'] == receipt['actual_tool_exit_code'] == 0
    source_path = ROOT / audit['source']
    assert sha256(source_path) == audit['source_sha256']
    source = read(source_path)
    cases = {x['id']: x for x in source['rows']}
    csv_path = args.input_directory / audit['identity_csv']['file']
    assert sha256(csv_path) == audit['identity_csv']['sha256']
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == audit['identity_csv']['rows'] == 294
    index = {(x['case'], x['sequence'], int(x['original_GT_ID'])): x for x in rows}
    assert len(index) == len(rows)
    addresses = {(x['sequence'], int(x['original_GT_ID']))
                 for x in rows if x['case'] == 'NP_temporal_ir_e06'}
    assert len(addresses) == 21
    for case in audit['fixed_cases']:
        assert {(x['sequence'], int(x['original_GT_ID']))
                for x in rows if x['case'] == case} == addresses
    pairs = [('NP_temporal_ir_e06', other) for other in audit['fixed_cases']
             if other != 'NP_temporal_ir_e06']
    pairs += [('NP_quality_ir_e06', other) for other in
              ['NP_quality_rgb_e02', 'NP_quality_gray_e05', 'NP_quality_rgb_e06']]
    comparisons = []
    for primary, control in pairs:
        paired = []
        for seq, gt_id in sorted(addresses):
            p, c = index[(primary, seq, gt_id)], index[(control, seq, gt_id)]
            assert p['GT_detections'] == c['GT_detections']
            assert p['GT_median_width'] == c['GT_median_width']
            assert p['GT_median_height'] == c['GT_median_height']
            p_quality, c_quality = p['GT_association_quality'], c['GT_association_quality']
            paired.append(dict(sequence=seq, GT_ID=gt_id, GT_detections=int(p['GT_detections']),
                primary_dominant_GT_coverage=float(p['dominant_tracker_GT_coverage']),
                control_dominant_GT_coverage=float(c['dominant_tracker_GT_coverage']),
                delta_dominant_GT_coverage=(float(p['dominant_tracker_GT_coverage'])
                                            - float(c['dominant_tracker_GT_coverage'])),
                delta_dominant_tracker_matches=int(p['dominant_tracker_matches']) - int(c['dominant_tracker_matches']),
                delta_matched_GT_detections=int(p['matched_GT_detections']) - int(c['matched_GT_detections']),
                primary_distinct_matched_tracker_IDs=int(p['distinct_matched_tracker_IDs']),
                control_distinct_matched_tracker_IDs=int(c['distinct_matched_tracker_IDs']),
                delta_GT_association_quality=(float(p_quality) - float(c_quality)
                                              if p_quality and c_quality else None)))
        comparisons.append(dict(primary=primary, control=control,
            official_combined_delta={k: cases[primary]['combined47'][k] - cases[control]['combined47'][k]
                                     for k in ['HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW']},
            alpha0_5_diagnostic_by_sequence={seq: summarize([x for x in paired if x['sequence'] == seq])
                                           for seq in audit['sequences']},
            alpha0_5_combined_diagnostic=summarize(paired), all_GT_paired_diagnostics=paired))
    out = dict(status='COMPLETE_DESCRIPTIVE_DEV_IDENTITY_ERROR_ANALYSIS',
        source_audit_sha256=sha256(audit_path), source_identity_CSV_sha256=sha256(csv_path),
        source_full_Dev_report_sha256=sha256(source_path),
        actual_CPU_execution_receipt_sha256=sha256(args.execution_receipt),
        source_code=str(Path(__file__).resolve().relative_to(ROOT)), source_code_sha256=sha256(__file__),
        comparisons=comparisons, fixed_seed=0, GPU_work=False, fitting=False,
        interpretation='Dominant-ID coverage and its concentration are posthoc alpha0.5 diagnostics. They do not decompose HOTA, show temporal error locations, establish independent contributions, or prove that a module caused a change.',
        historical_Test_previously_exposed=True, RGB_pretraining_saw_Dev=True,
        pseudoIR_derived_from_RGB=True, new_Test_results_read=False,
        new_selection_or_routing=False, supported_innovations=0, goal_met=False)
    assert not args.output_json.exists()
    args.output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=out['status'], comparisons=len(comparisons),
                         paired_rows=sum(len(x['all_GT_paired_diagnostics']) for x in comparisons), goal_met=False)))


if __name__ == '__main__':
    main()
