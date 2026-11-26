#!/usr/bin/env python3
"""Posthoc Dev diagnostics from the unchanged official HOTA match matrices.

No image/model loading, optimization, Test reading, or selection. The profile
hook observes HOTA's return locals; it does not replace its matching algorithm.
"""
import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV.json'
SOURCE_SHA = '63426a8689f27ce662557b2df1d737db1d55d18af7e716bdd1374bdc290d688e'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def observe_hota(metric, data, hota_path):
    """Capture the real evaluator's terminal local arrays without modifying it."""
    captured = []

    def profile(frame, event, arg):
        if (event == 'return' and frame.f_code.co_name == 'eval_sequence'
                and frame.f_code.co_filename == hota_path):
            local = frame.f_locals
            captured.append({key: local[key] for key in
                             ['matches_counts', 'gt_id_count', 'tracker_id_count']})

    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        result = metric.eval_sequence(data)
    finally:
        sys.setprofile(previous)
    assert len(captured) == 1, 'Need exactly one actual official HOTA evaluation'
    return result, captured[0]


def check_official(result, expected, metric, np):
    largest_error = 0.0
    for field in metric.fields:
        factor = 1 if field in metric.integer_array_fields else 100
        actual = float(np.mean(result[field])) * factor
        error = abs(actual - expected[field])
        assert error < 1e-8, (field, actual, expected[field], error)
        largest_error = max(largest_error, error)
    return largest_error


def check_detailed(result, detailed, metric, np):
    """Compare every alpha, rather than infer array equality from its mean."""
    largest_error = 0.0
    checks = 0
    for field in metric.float_array_fields + metric.integer_array_fields:
        for index, alpha in enumerate(metric.array_labels):
            column = f'{field}___{int(round(float(alpha) * 100))}'
            error = abs(float(result[field][index]) - float(detailed[column]))
            assert error < 1e-12, (column, error)
            largest_error = max(largest_error, error)
            checks += 1
    for field in metric.float_fields:
        error = abs(float(result[field]) - float(detailed[field]))
        assert error < 1e-12, (field, error)
        largest_error = max(largest_error, error)
        checks += 1
    assert checks == 12 * 19 + 3
    return largest_error, checks


def identity_rows(case_id, seq, dataset, raw, data, metric, result, captured, np):
    # Reconstruct original GT labels using the same kept-class rule as this
    # source-bound dataset. Assert every frame's labels AND boxes after mapping.
    assert dataset.do_preproc and dataset.benchmark != 'MOT15'
    kept_ids, kept_boxes = [], []
    for t in range(data['num_timesteps']):
        mask = ((raw['gt_extras'][t]['zero_marked'] != 0)
                & (raw['gt_classes'][t] == dataset.class_name_to_class_id['pedestrian']))
        kept_ids.append(raw['gt_ids'][t][mask])
        kept_boxes.append(raw['gt_dets'][t][mask])
    original_ids = np.unique(np.concatenate(kept_ids))
    assert len(original_ids) == data['num_gt_ids']
    boxes = [[] for _ in original_ids]
    frames = [[] for _ in original_ids]
    for t, (ids, bxs) in enumerate(zip(kept_ids, kept_boxes)):
        mapped = np.searchsorted(original_ids, ids)
        np.testing.assert_array_equal(mapped, data['gt_ids'][t])
        np.testing.assert_array_equal(bxs, data['gt_dets'][t])
        for g, box in zip(mapped, bxs):
            frames[g].append(t + 1)
            boxes[g].append(box)
    gt_count = captured['gt_id_count'][:, 0]
    pred_count = captured['tracker_id_count'][0]
    np.testing.assert_array_equal(gt_count, [len(f) for f in frames])
    assert len(captured['matches_counts']) == len(metric.array_labels) == 19
    for index, matches in enumerate(captured['matches_counts']):
        np.testing.assert_allclose(matches.sum(), result['HOTA_TP'][index], rtol=0, atol=0)
        numerator = (matches * matches / np.maximum(
            1, gt_count[:, None] + pred_count[None, :] - matches)).sum()
        np.testing.assert_allclose(numerator / max(1, matches.sum()),
                                   result['AssA'][index], rtol=0, atol=1e-14)
    alpha_index = int(np.flatnonzero(np.isclose(metric.array_labels, .5))[0])
    matches = captured['matches_counts'][alpha_index]
    out = []
    for g, original in enumerate(original_ids):
        counts = matches[g]
        matched = int(counts.sum())
        dominant = int(counts.max()) if len(counts) else 0
        box = np.asarray(boxes[g])
        association_numerator = float((counts * counts / np.maximum(
            1, gt_count[g] + pred_count - counts)).sum())
        out.append(dict(
            case=case_id, sequence=seq, original_GT_ID=int(original),
            GT_detections=int(gt_count[g]), first_GT_frame=frames[g][0],
            last_GT_frame=frames[g][-1], GT_median_width=float(np.median(box[:, 2])),
            GT_median_height=float(np.median(box[:, 3])),
            HOTA_alpha=.5, matched_GT_detections=matched,
            missed_GT_detections=int(gt_count[g]) - matched,
            GT_match_recall=matched / gt_count[g],
            distinct_matched_tracker_IDs=int(np.count_nonzero(counts)),
            dominant_tracker_matches=dominant,
            dominant_tracker_GT_coverage=dominant / gt_count[g],
            dominant_tracker_share_of_matches=dominant / matched if matched else None,
            GT_association_quality=association_numerator / matched if matched else None,
        ))
    assert sum(x['matched_GT_detections'] for x in out) == result['HOTA_TP'][alpha_index]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only diagnostic'
    assert sha256(SOURCE) == SOURCE_SHA
    source = read(SOURCE)
    assert source['stage'] == 6 and source['split'] == 'dev'
    assert source['all62_cases_included'] and source['supported_innovations'] == 0
    plan = read(ROOT / 'configs/perceptual_mft_20261122.json')
    protocol = read(ROOT / plan['protocol'])
    assert list(source['sequences']) == protocol['dev'] == ['BT-005', 'SN-013']
    assert source['frames'] == 4834
    frozen = source['registered_sources_verified_unchanged']
    for name, expected in frozen.items():
        assert sha256(ROOT / name) == expected, name
    assert not args.output_directory.exists(), 'Use a new, exclusive output directory'
    args.output_directory.mkdir(parents=True)
    started = time.monotonic()
    import numpy as np
    if 'float' not in np.__dict__:
        np.float = float
    if 'int' not in np.__dict__:
        np.int = int
    trackeval_root = Path(plan['evaluation']['TrackEval_root'])
    sys.path.insert(0, str(trackeval_root))
    from trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox
    from trackeval.metrics.hota import HOTA
    hota_file = inspect.getsourcefile(HOTA)
    dataset_file = inspect.getsourcefile(MotChallenge2DBox)
    implementation = {str(p): sha256(p) for p in [hota_file, dataset_file,
        trackeval_root / 'trackeval/datasets/_base_dataset.py']}
    rows = {row['id']: row for row in source['rows']}
    cases = sorted(set(source['six_own_rounded_official_Dev_best'].values())
                   | {f'NP_{fit}_e06' for fit in source['six_own_rounded_official_Dev_best']}
                   | {'NP_source_canonical_RGB', 'NP_source_fresh_RGB',
                      'NP_temporal_ir_e06_roll', 'NP_temporal_ir_e06_off'})
    all_identities, evaluations, inputs = [], [], {}
    metric = HOTA()
    for case_id in cases:
        case = rows[case_id]
        folder = Path(case['config']['stream_directory']) / case_id / 'track_results'
        csv_path = ROOT / case['official_CSV']['file']
        assert sha256(csv_path) == case['official_CSV']['sha256']
        inputs[str(csv_path)] = sha256(csv_path)
        with csv_path.open() as stream:
            detailed_rows = {x['seq']: x for x in csv.DictReader(stream)}
        assert set(detailed_rows) == set(protocol['dev']) | {'COMBINED'}
        dataset = MotChallenge2DBox(dict(
            TRACKERS_FOLDER=str(folder), TRACKERS_TO_EVAL=['QRTM_V1'],
            GT_FOLDER=str(Path(plan['evaluation']['GT_root']) / 'train'),
            OUTPUT_FOLDER=str(args.output_directory), BENCHMARK='MFT26IR',
            CLASSES_TO_EVAL=['pedestrian'], GT_LOC_FORMAT='{gt_folder}/{seq}/gt/gt.txt',
            SEQ_INFO=protocol['dev'], PRINT_RESULTS=False, PRINT_CONFIG=False,
            TIME_PROGRESS=False, DO_PREPROC=False, use_super_categories=False,
            USE_PARALLEL=False))
        sequence_results = {}
        for seq in protocol['dev']:
            mot = folder / f'{seq}.txt'
            gt = Path(plan['evaluation']['GT_root']) / 'train' / seq / 'gt/gt.txt'
            assert sha256(mot) == case['original_coverage'][seq]['output_sha256']
            assert sha256(gt) == protocol['derived_gt_sha256'][f'train/{seq}']
            for path in [mot, gt, gt.parent.parent / 'seqinfo.ini']:
                inputs[str(path)] = sha256(path)
            raw = dataset.get_raw_seq_data(seq)
            data = dataset.get_preprocessed_seq_data(raw, 'pedestrian')
            assert data['num_timesteps'] == source['sequences'][seq]
            result, captured = observe_hota(metric, data, hota_file)
            official = next(x for x in case['per_sequence47'] if x['seq'] == seq)
            error = check_official(result, official, metric, np)
            detailed_error, detailed_checks = check_detailed(result, detailed_rows[seq], metric, np)
            identities = identity_rows(case_id, seq, dataset, raw, data, metric,
                                       result, captured, np)
            all_identities.extend(identities)
            sequence_results[seq] = result
            evaluations.append(dict(case=case_id, sequence=seq,
                official_HOTA_parity_max_error=error,
                all19_detailed_CSV_max_error=detailed_error,
                detailed_CSV_numeric_checks=detailed_checks,
                all19_HOTA_arrays={f: result[f].tolist() for f in
                                  metric.float_array_fields + metric.integer_array_fields},
                official_metrics={f: official[f] for f in metric.fields},
                GT_IDs=len(identities), all19_match_matrix_accounting_passed=True))
        result = metric.combine_sequences(sequence_results)
        error = check_official(result, case['combined47'], metric, np)
        detailed_error, detailed_checks = check_detailed(result, detailed_rows['COMBINED'], metric, np)
        evaluations.append(dict(case=case_id, sequence='COMBINED',
            official_HOTA_parity_max_error=error,
            all19_detailed_CSV_max_error=detailed_error,
            detailed_CSV_numeric_checks=detailed_checks,
            all19_HOTA_arrays={f: result[f].tolist() for f in
                              metric.float_array_fields + metric.integer_array_fields},
            official_metrics={f: case['combined47'][f] for f in metric.fields}))
        print(json.dumps(dict(case=case_id, complete_Dev=True, official_parity=True,
                              HOTA=case['combined47']['HOTA'])), flush=True)
    for name, expected in {**inputs, **implementation}.items():
        assert sha256(name) == expected, name
    for name, expected in frozen.items():
        assert sha256(ROOT / name) == expected, name
    csv_path = args.output_directory / 'all_Dev_GT_identity_diagnostics.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_identities[0]))
        writer.writeheader()
        writer.writerows(all_identities)
    report = dict(status='COMPLETE_CPU_DEV_OFFICIAL_MATCH_IDENTITY_DIAGNOSTIC',
        UTC=datetime.now(timezone.utc).isoformat(), source=str(SOURCE.relative_to(ROOT)),
        source_sha256=SOURCE_SHA, source_code=str(Path(__file__).resolve().relative_to(ROOT)),
        source_code_sha256=sha256(__file__), split='dev', minimum_training_stage=6,
        fixed_cases=cases, cases=len(cases), sequences=source['sequences'],
        frames_per_case=source['frames'], alpha_for_identity_diagnostics=.5,
        official_19_alpha_fields_reproduced=True,
        official_implementation_hashes=implementation, input_hashes=inputs,
        unchanged_registered_sources=frozen, evaluations=evaluations,
        identity_csv=dict(file=csv_path.name, sha256=sha256(csv_path),
                          rows=len(all_identities)),
        diagnostic_definitions=dict(
            dominant_tracker_GT_coverage='max_p matched(g,p) / number_of_GT_detections(g)',
            GT_association_quality='sum_p matched(g,p)^2/(n_gt(g)+n_pred(p)-matched(g,p)) / sum_p matched(g,p)',
            distinct_matched_tracker_IDs='Number of distinct HOTA-matched prediction identities at IoU alpha=0.5; not CLEAR Frag or IDSW.'),
        limitations=[
            'Posthoc descriptive Dev diagnosis at alpha=0.5; not a new official metric or per-target HOTA.',
            'HOTA association matching is different from CLEAR matching. Counts are not ID switches or CLEAR fragmentations.',
            'Per-target association quality depends on full predicted track histories, including matches to other targets.',
            'Targets share a sequence and are not independent statistical replicates. No significance claims.',
            'Own-best checkpoints were already chosen by the published stage6 Dev rule; same-epoch6 cases are also included.',
            'No sequence/identity routing, new threshold tuning, architecture change, or final model choice.',
            'Historical Test exposure, RGB pretraining saw Dev, and RGB-derived pseudoIR limits remain.'],
        CPU_internal_wall_seconds=time.monotonic() - started,
        GPU_work=False, model_or_image_loading=False, fitting=False,
        Test_results_read=False, Test_GT_read=False, Test_based_selection=False,
        final_model_selected=False, supported_innovations=0, goal_met=False)
    dump(args.output_directory / 'complete.json', report)
    print(json.dumps(dict(status=report['status'], cases=len(cases),
        identity_rows=len(all_identities), goal_met=False)), flush=True)


if __name__ == '__main__':
    main()
