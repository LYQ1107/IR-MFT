#!/usr/bin/env python3
"""Read existing full Dev CSVs; preserve every frozen cue pair and domain."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.assess_native_remaining_test import paired_signature, off_signature

SPEC = ROOT/'configs/perceptual_mechanism_diagnosis_20261125.json'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def metrics(row):
    result = {key: 100*float(row[key+'___AUC']) for key in ['HOTA', 'DetA', 'AssA', 'LocA']}
    result.update({key: 100*float(row[key]) for key in ['IDF1', 'MOTA', 'CLR_Re', 'CLR_Pr']})
    for key in ['CLR_TP', 'CLR_FN', 'CLR_FP', 'IDSW', 'Frag', 'IDs', 'GT_IDs', 'GT_Dets', 'CLR_Frames']:
        value = float(row[key])
        assert value.is_integer()
        result[key] = int(value)
    return result


def decomposition(ir, rgb):
    values = []
    for threshold in range(5, 100, 5):
        def triple(row):
            return [float(row[f'{name}___{threshold}']) for name in ['HOTA', 'DetA', 'AssA']]
        hi, di, ai = triple(ir)
        hr, dr, ar = triple(rgb)
        assert abs(hi-math.sqrt(di*ai)) < 1e-10
        assert abs(hr-math.sqrt(dr*ar)) < 1e-10
        if min(hi, di, ai, hr, dr, ar) <= 0:
            continue
        hota = math.log(hi/hr)
        det, association = .5*math.log(di/dr), .5*math.log(ai/ar)
        assert abs(hota-det-association) < 1e-10
        values.append((hota, det, association))
    return dict(valid_thresholds=len(values),
        mean_log_HOTA_ratio=None if not values else sum(v[0] for v in values)/len(values),
        mean_half_log_DetA_ratio=None if not values else sum(v[1] for v in values)/len(values),
        mean_half_log_AssA_ratio=None if not values else sum(v[2] for v in values)/len(values),
        thresholds_with_larger_absolute_association_term=sum(abs(v[2]) > abs(v[1]) for v in values),
        scope='Exact thresholdwise algebra of official HOTA. Mean logs describe geometric ratios, '
            'not additive HOTA percentage points or causal effects. Do not reconstruct HOTA AUC from DetA/AssA AUC.')


def main():
    specification = read(SPEC)
    plan_path = ROOT/'configs/perceptual_mft_20261122.json'
    assert sha(plan_path) == specification['main_plan_sha256']
    plan = read(plan_path)
    protocol_path = ROOT/plan['protocol']
    assert sha(protocol_path) == specification['protocol_sha256']
    protocol = read(protocol_path)
    registry_path = ROOT/specification['dev']['source']
    assert sha(registry_path) == specification['dev']['source_sha256']
    registry = read(registry_path)
    assert len(registry['groups']) == 58 and registry['fixed_seed'] == 0
    records, buckets, controls = {}, {}, {}
    for group in registry['groups']:
        selected = group['selected']
        path = Path(selected['source_Dev_metrics_path'])
        assert sha(path) == selected['source_Dev_metrics_sha256']
        value = read(path)
        manifest = value['manifest']
        assert value['status'] == 'COMPLETE' and len(value['trackeval']) == 39
        assert manifest['split'] == 'dev' and manifest['frames'] == 4834
        assert manifest['sequences'] == protocol['dev']
        assert manifest['config']['native_seed'] == 0
        table = path.parent/'trackeval_output/pedestrian_detailed.csv'
        with table.open() as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            assert len(reader.fieldnames) == 276
        assert [row['seq'] for row in rows] == protocol['dev']+['COMBINED']
        for row in rows:
            hota = sum(float(row[f'HOTA___{t}']) for t in range(5, 100, 5))/19
            assert abs(hota-float(row['HOTA___AUC'])) < 1e-12
        assert abs(100*float(rows[-1]['HOTA___AUC'])-selected['HOTA']) <= .000501
        assert sum(int(float(row['CLR_Frames'])) for row in rows[:-1]) == 4834
        record = dict(group=group['group_id'], epoch=selected['epoch'],
            metrics_json=str(path), metrics_json_sha256=sha(path),
            official_CSV=str(table), official_CSV_sha256=sha(table),
            metrics={row['seq']: metrics(row) for row in rows})
        records[group['group_id']] = record, {row['seq']: row for row in rows}
        key, cue = paired_signature(selected['config'])
        assert cue not in buckets.setdefault(key, {})
        buckets[key][cue] = group
        if selected['config'].get('disable_native_increment'):
            controls.setdefault(off_signature(selected['config']), []).append(group)
    pairs, unmatched = [], []
    for _, members in sorted(buckets.items()):
        if set(members) != {'rgb', 'ir'}:
            unmatched.extend(row['group_id'] for row in members.values())
            continue
        ig, rg = members['ir'], members['rgb']
        ir, ir_csv = records[ig['group_id']]
        rgb, rgb_csv = records[rg['group_id']]
        matching_off = controls.get(off_signature(ig['selected']['config']), [])
        off = None
        if matching_off:
            candidates = [records[g['group_id']][0] for g in matching_off]
            assert all(c['metrics'] == candidates[0]['metrics'] for c in candidates)
            off = candidates[0]
        differences = {}
        for sequence in protocol['dev']+['COMBINED']:
            delta = {k: ir['metrics'][sequence][k]-rgb['metrics'][sequence][k]
                     for k in ir['metrics'][sequence]}
            off_delta = None if off is None else {k: ir['metrics'][sequence][k]-off['metrics'][sequence][k]
                                                  for k in ir['metrics'][sequence]}
            differences[sequence] = dict(IR_minus_RGB=delta, IR_minus_own_increment_off=off_delta,
                HOTA_threshold_decomposition=decomposition(ir_csv[sequence], rgb_csv[sequence]))
        signs = [differences[seq]['IR_minus_RGB']['HOTA'] for seq in protocol['dev']]
        pairs.append(dict(IR_group=ig['group_id'], RGB_group=rg['group_id'],
            frontend=ig['selected']['config']['detection_arm'], backend=ig['selected']['config']['backend'],
            IR=ir, RGB=rgb, own_increment_off=off, differences=differences,
            both_Dev_sequences_positive=all(v > 1e-10 for v in signs),
            opposite_Dev_sequence_signs=min(signs) < -1e-10 and max(signs) > 1e-10,
            both_sequences_positive_against_own_off=None if off is None else all(
                differences[seq]['IR_minus_own_increment_off']['HOTA'] > 1e-10 for seq in protocol['dev'])))
    assert len(pairs) == 23 and len(unmatched) == 12
    summary = dict(pairs=23, unmatched_groups=12,
        combined_Dev_IR_positive=sum(p['differences']['COMBINED']['IR_minus_RGB']['HOTA'] > 1e-10 for p in pairs),
        both_Dev_sequences_IR_positive=sum(p['both_Dev_sequences_positive'] for p in pairs),
        opposite_Dev_sequence_signs=sum(p['opposite_Dev_sequence_signs'] for p in pairs),
        original_Dev_own_off_available=sum(p['own_increment_off'] is not None for p in pairs),
        both_sequences_positive_against_RGB_and_own_off=sum(p['both_Dev_sequences_positive'] and
            p['both_sequences_positive_against_own_off'] is True for p in pairs))
    paths = ['tools/diagnose_native_dev_domains.py', 'tools/assess_native_remaining_test.py',
             'tools/native_remaining_test_backfill.py']
    result = dict(status='COMPLETE_EXISTING_FROZEN23_FULL_DEV_DOMAIN_DIAGNOSIS',
        specification_sha256=sha(SPEC), registry_sha256=sha(registry_path),
        source_hashes={name: sha(ROOT/name) for name in paths}, summary=summary,
        pairs=pairs, unmatched_groups=unmatched, frames_per_case=4834,
        actual_Test_accuracy_inputs=[], new_inference=False, new_fits=False, GPU_work=False,
        no_Test_selection=True, current_primary_unchanged=True,
        scope='Only original full Dev metric JSON and 276-columnCSV. All58frozen behavior groups retained. '
            'Different Dev-selected snapshot epochs are reported, not rematched after outcomes. '
            'Identity-vector families are not new fusion architectures; domain signs do not prove quality routing.',
        historical_Test_exposed=True, RGB_pretraining_saw_current_Dev=True,
        supported_innovations=0, goal_met=False)
    stem = ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261125_NATIVE_DEV_DOMAINS'
    assert not stem.with_suffix('.json').exists()
    stem.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], summary=summary)), flush=True)


if __name__ == '__main__':
    main()
