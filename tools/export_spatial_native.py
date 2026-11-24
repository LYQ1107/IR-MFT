#!/usr/bin/env python3
"""Publish every spatial curve, original official CSV and all47 fields."""
import csv
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.export_perceptual_metrics47 import official_schema, read_table
from tools.native_pyramid_experiment import read
from tools.observe_spatial_native import FRESH, require_test_freeze
from tools.run_pseudo_ir_comparison import sha256
from tools.spatial_native_study import STUDY, context, frozen_json, registered

DOCS = ROOT/'docs/fusion_research'


def small_json(path, value):
    assert len((json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode()) < 2**20
    frozen_json(path, value)


def publish_case(base, split, model, protocol, schema):
    identifier = model['id']
    source = Path(base)/'evaluation'/split/identifier/'metrics.json'
    value = read(source)
    manifest = value['manifest']
    assert value['status'] == 'COMPLETE' and manifest['study_sha256'] == sha256(STUDY)
    assert len(value['trackeval']) == 39
    assert manifest['config']['model'] == model and manifest['config']['id'] == identifier
    sequences = protocol['dev' if split == 'dev' else 'historical_test']
    assert manifest['sequences'] == sequences and manifest['split'] == split
    assert manifest['frames'] == {'dev': 4834, 'test': 15985}[split]
    if not model.get('baseline'):
        assert sha256(model['checkpoint']) == model['checkpoint_sha256']
    original = source.parent/'trackeval_output/pedestrian_detailed.csv'
    destination = DOCS/'perceptual_20261122_official_tables/native_spatial'/split/(identifier+'.csv')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert sha256(destination) == sha256(original)
    else:
        shutil.copy2(original, destination)
    assert destination.stat().st_size < 2**20
    rows, provenance = read_table(destination, schema, protocol)
    combined = next(r for r in rows if r['seq'] == 'COMBINED')
    assert all(row['split'] == split for row in rows)
    assert all(combined[k] == value['selection_metrics'][k] for k in ['HOTA', 'AssA', 'IDF1'])
    gates = {}
    for sequence, reference in manifest['sources'].items():
        assert sha256(reference['manifest_path']) == reference['manifest_sha256']
        producer = read(reference['manifest_path'])
        stream = producer['streams'][identifier]
        assert stream['frames'] == manifest['coverage'][sequence]['frames']
        assert sha256(stream['track_path']) == stream['track_sha256'] == manifest['coverage'][sequence]['track_sha256']
        gate = producer.get('gate_statistics', {}).get(identifier)
        if not model.get('baseline'):
            assert gate is not None and gate['frames'] == stream['frames']
        gates[sequence] = gate
    payload = dict(id=identifier, split=split, model=model, metrics39=value['trackeval'],
        combined47={s['metric']: combined[s['metric']] for s in schema}, per_sequence47=rows,
        official_CSV_file=str(destination.relative_to(ROOT)), official_CSV_sha256=sha256(original),
        metric_schema_provenance=provenance, source_metrics=str(source), source_metrics_sha256=sha256(source),
        coverage=manifest['coverage'], source_manifests=manifest['sources'], gate_statistics_by_sequence=gates)
    path = DOCS/'perceptual_native_spatial_20261124'/split/(identifier+'.json')
    small_json(path, payload)
    return payload, dict(id=identifier, file=str(path.relative_to(ROOT)), sha256=sha256(path))


def write47(path, rows, schema):
    fields = ['candidate', 'split', 'seq']+[s['metric'] for s in schema]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open() as stream:
            assert list(csv.DictReader(stream)) == [{k: str(r[k]) for k in fields} for r in rows]
        return
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)
    assert path.stat().st_size < 2**20


def scientific_effects(freeze, dev, test):
    roles = freeze['roles']
    primary = test[roles['primary_IR']]['combined47']
    selected = freeze['selection']['primary']
    dev_primary = selected['selection_metrics']
    comparisons = {role: primary['HOTA']-test[roles[role]]['combined47']['HOTA'] for role in
        ['source_RGB_fresh', 'source_RGB_canonical', 'matched_RGB', 'matched_gray',
         'global_Dev_selected_rgb', 'global_Dev_selected_gray', 'primary_IR_unavailable', 'primary_IR_image_roll']}
    necessary_cue_Test = all(value > 0 for value in comparisons.values())
    removal_effects = {}
    for label, choices in freeze['removals'].items():
        policies = {}
        for policy, row in choices.items():
            removed = test[roles['removal_'+label+'_'+policy]]['combined47']
            policies[policy] = dict(Dev_HOTA_delta=dev_primary['HOTA']-row['selection_metrics']['HOTA'],
                Test_HOTA_delta=primary['HOTA']-removed['HOTA'],
                Test_IDF1_delta=primary['IDF1']-removed['IDF1'],
                Test_AssA_delta=primary['AssA']-removed['AssA'], Test_IDSW_delta=primary['IDSW']-removed['IDSW'])
        necessary = necessary_cue_Test and all(p['Dev_HOTA_delta'] > 0 and p['Test_HOTA_delta'] > 0 for p in policies.values())
        removal_effects[label] = dict(policies=policies, necessary_numeric_conditions=necessary,
            independently_validated_innovation=False)
    # Every Dev comparison is identified before Test, including global cue controls.
    dev_cue = {}
    for role in ['source_RGB_fresh', 'source_RGB_canonical', 'matched_RGB', 'matched_gray',
                 'global_Dev_selected_rgb', 'global_Dev_selected_gray', 'primary_IR_unavailable', 'primary_IR_image_roll']:
        dev_cue[role] = dev_primary['HOTA']-dev[roles[role]]['combined47']['HOTA']
    necessary_cue_Dev = all(v > 0 for v in dev_cue.values())
    for value in removal_effects.values():
        value['necessary_numeric_conditions'] &= necessary_cue_Dev
    return dict(Test_HOTA_cue_deltas=comparisons, Dev_HOTA_cue_deltas=dev_cue,
        necessary_all_strong_cue_controls_both_Dev_Test=necessary_cue_Test and necessary_cue_Dev,
        removal_effects=removal_effects,
        scope='Unrounded official combined HOTA percentage points. Positive controls/removals are necessary, '
            'not proof of novelty; fixed single seed, historical Test exposure and pretrained Dev exposure remain.')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plan, study, protocol, calibration, work, checkpoint, base = context()
    registration = registered(work)
    state = read(base/'state.json')
    assert state['status'] == 'COMPLETE_SPATIAL_EXECUTION_PENDING_SCIENTIFIC_EXPORT'
    specification = base/'specifications/test_frozen.json'
    freeze = require_test_freeze(base, specification)
    complete_dev = read(base/'complete_Dev.json')
    assert complete_dev['complete_all9_normal_and_required_removal_Dev'] and not complete_dev['Test_accuracy_used']
    runtime = read(base/'individual_runtime.json')
    assert runtime['status'] == 'COMPLETE_SPATIAL_INDIVIDUAL_SOURCE_MATCHED_PREFIX_RUNTIME'
    assert runtime['Test_selection_sha256'] == sha256(base/'Test_selection_frozen.json')
    assert {r['id'] for r in runtime['rows']} == {m['id'] for m in freeze['models']}
    schema, _ = official_schema(plan['evaluation']['TrackEval_root'])
    objects, references = {'dev': {}, 'test': {}}, {'dev': [], 'test': []}
    assert all(sha256(c['config']) == c['config_sha256'] for c in freeze['cases'])
    for split, models in [('dev', complete_dev['rows']), ('test', [read(c['config'])['model'] for c in freeze['cases']])]:
        for row in models:
            if split == 'dev':
                assert sha256(row['config']) == row['config_sha256']
            model = read(row['config'])['model'] if split == 'dev' else row
            if model['id'] in objects[split]:
                continue
            value, reference = publish_case(base, split, model, protocol, schema)
            objects[split][model['id']] = value
            references[split].append(reference)
        metric_rows = [dict(r, candidate=identifier) for identifier, value in objects[split].items() for r in value['per_sequence47']]
        write47(DOCS/'perceptual_native_spatial_20261124'/(split+'_metrics47.csv'), metric_rows, schema)
    effects = scientific_effects(freeze, objects['dev'], objects['test'])
    primary = objects['test'][freeze['roles']['primary_IR']]['combined47']
    fresh = objects['test'][FRESH['id']]['combined47']
    canonical = objects['test'][freeze['roles']['source_RGB_canonical']]['combined47']
    required = max(34.529, fresh['HOTA']+5, canonical['HOTA']+5)
    costs = {}
    for phase, entries in state['phases'].items():
        if not (phase.startswith('TRAIN_') or phase == 'REMOVAL_TRAIN'):
            continue
        for fit_id, entry in entries.items():
            assert entry['status'] == 'COMPLETE' and entry['exit_code'] == 0
            costs.setdefault(fit_id, []).append(dict(phase=phase, lower_seconds=entry['worker_wall_lower_seconds'],
                upper_seconds=entry['worker_wall_upper_seconds'], observed_peak_RSS_bytes=entry.get('observed_peak_RSS_bytes'),
                log=entry['log'], physical_gpu=entry['slot'], start_ticks=entry['start_ticks']))
    report = dict(status='COMPLETE_NINE_ARM_SPATIAL_STUDY_REQUIRES_SCIENTIFIC_ASSESSMENT',
        study_sha256=sha256(STUDY), registry_sha256=sha256(base/'pipeline_registered.json'),
        Test_selection_sha256=sha256(base/'Test_selection_frozen.json'), Dev_cases=references['dev'], Test_cases=references['test'],
        roles=freeze['roles'], primary_Test47=primary, primary_minus_fixed_RGB_pp=primary['HOTA']-29.529,
        fixed_target_HOTA=34.529, fair_required_HOTA=required, numeric_fixed_target_met=primary['HOTA'] >= 34.529,
        numeric_fair_direct_RGB_plus5_met=primary['HOTA'] >= required, effect_analyses=effects,
        runtime=runtime, fit_launch_wall_cost_bounds=costs,
        fit_cost_scope='Launch-to-exit observation bounds from5-second polling; includes worker setup/loading. '
            'Per-fit stages retained separately; concurrent job times are not study wall time. Observed RSS is not exact peak.',
        original_primary_frozen_from_complete_Dev=True, Test_based_promotion=False,
        historical_Test_exposed=True, original_RGB_pretraining_saw_current_Dev=True,
        four_directions_complete=False, supported_innovations=0, goal_met=False)
    stem = DOCS/'PERCEPTUAL_MFT_20261124_NATIVE_SPATIAL_FORMAL'
    small_json(stem.with_suffix('.json'), report)
    text = ('# 新版 IR 空间融合完整研究\n\n'
        f"固定Dev主模型完整七段Test HOTA为{primary['HOTA']:.6f}，相对原始29.529为{primary['HOTA']-29.529:+.6f}点。"
        f"原始目标34.529；完整精度直接RGB公平+5门槛为{required:.6f}。\n\n"
        '所有九组正式正常模型和必要消融均保留完整Test；未据Test改选主模型。'
        '47字段、逐序列值和原276列CSV逐项保留，见JSON引用。'
        '去除与强RGB/灰度、错位/缺失比较仍须结合效应、先验工作和数据限制判断，不能自动计为创新。\n\n'
        '独立成本为七段各32帧、8帧预热24帧计时的B1前缀抽样，排除权重加载和离线DiffV2IR生成，非完整15985帧FPS。'
        '本研究不单独证明四方向全部完成；整体Goal尚未完成。\n')
    if stem.with_suffix('.md').exists():
        assert stem.with_suffix('.md').read_text() == text
    else:
        stem.with_suffix('.md').write_text(text)


if __name__ == '__main__':
    main()
