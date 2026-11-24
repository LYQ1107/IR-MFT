#!/usr/bin/env python3
"""Publish every frozen neck72 case; preserve the outstanding full-system costs."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.neck72_experiment import registered, config_for, STUDY, DOCS, ALGORITHMS
from tools.neck72_runtime import read
from tools.export_perceptual_metrics47 import official_schema, read_table
from tools.run_pseudo_ir_comparison import dump, sha256


def export_case(work, case, split, protocol, schema):
    folder = work / ('neck72_'+split) / case['id']
    path = folder / 'metrics.json'; value = read(path); manifest = value['manifest']
    assert value['status'] == 'COMPLETE' and len(value['trackeval']) == 39
    assert manifest['study_sha256'] == sha256(STUDY) and manifest['config'] == case['config']
    assert manifest['split'] == split and manifest['frames'] == (4834 if split == 'dev' else 15985)
    assert not manifest['GT_inference_input'] and not manifest['future_frame_input'] and not manifest['previous_outputs_rewritten']
    assert manifest['tracking_and_ID_counter_reset_each_sequence']
    source = folder / 'trackeval_output/pedestrian_detailed.csv'
    published = DOCS / 'perceptual_20261122_official_tables/neck72' / split / (case['id']+'.csv')
    published.parent.mkdir(parents=True, exist_ok=True)
    if published.exists():
        assert sha256(published) == sha256(source)
    else:
        shutil.copy2(source, published)
    values, provenance = read_table(published, schema, protocol)
    combined = next(row for row in values if row['seq'] == 'COMBINED')
    sequences = protocol['dev' if split == 'dev' else 'historical_test']
    assert manifest['sequences'] == sequences
    for seq in sequences:
        current = folder / 'track_results' / (seq+'.txt')
        assert sha256(current) == manifest['coverage'][seq]['output_sha256']
    return dict(case=case, metrics39=value['trackeval'], unrounded_combined_HOTA=combined['HOTA'],
        official47_by_sequence={row['seq']: {k: v for k, v in row.items()
            if k not in ['candidate', 'split', 'seq']} for row in values},
        official_CSV=str(published.relative_to(ROOT)), official_CSV_sha256=sha256(published),
        metrics_sha256=sha256(path), manifest=manifest,
        CSV_source_columns=provenance['source_columns'])


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plan, protocol, work, study, registration = registered()
    state = read(work / 'neck72_study_state.json')
    assert state['status'] == 'COMPLETE_NECK72_ALL_SELECTED_TEST_PENDING_FULL_SYSTEM_COST_AUDIT'
    freeze_path = work / 'neck72_Test_selection_frozen.json'; freeze = read(freeze_path)
    assert freeze['study_sha256'] == sha256(STUDY) and freeze['complete_selected_Dev_before_Test']
    assert not freeze['Test_accuracy_read_for_selection']
    schema, _ = official_schema(plan['evaluation']['TrackEval_root'])
    destination = DOCS / 'perceptual_neck72_20261124'
    assert not destination.exists(), 'Preserve complete prior scientific exports'
    destination.mkdir()
    curve_records = {}
    for head in study['heads']:
        for algorithm in ALGORITHMS:
            key = head['id']+'_'+algorithm; curve_records[key] = []
            for point in freeze['curves'][key]:
                case = config_for(study, head, point['epoch'], algorithm)
                assert sha256(point['metrics_path']) == point['metrics_sha256']
                record = export_case(work, case, 'dev', protocol, schema)
                path = destination / 'all_normal_Dev_snapshots' / (case['id']+'.json')
                dump(path, record); assert path.stat().st_size < 1024*1024
                curve_records[key].append(dict(epoch=point['epoch'], HOTA=point['HOTA'],
                    record=str(path.relative_to(ROOT)), record_sha256=sha256(path),
                    official_CSV=record['official_CSV'], official_CSV_sha256=record['official_CSV_sha256']))
    rows, dev, test = [], {}, {}
    for case in freeze['cases']:
        records = {}
        for split in ['dev', 'test']:
            record = export_case(work, case, split, protocol, schema)
            records[split] = record
        assert sha256(work / 'neck72_dev' / case['id'] / 'metrics.json') == freeze['Dev_metrics'][case['id']]
        record_path = destination / (case['id']+'.json')
        dump(record_path, records); assert record_path.stat().st_size < 1024*1024
        a, b = records['dev'], records['test']; dev[case['id']], test[case['id']] = a, b
        rows.append(dict(case=case, Dev_HOTA=a['unrounded_combined_HOTA'],
            Test_HOTA=b['unrounded_combined_HOTA'], Test_metrics39=b['metrics39'],
            record=str(record_path.relative_to(ROOT)), record_sha256=sha256(record_path),
            official_CSV=b['official_CSV'], official_CSV_sha256=b['official_CSV_sha256']))
    roles = freeze['roles']; primary = test[roles['primary']]
    baseline = test['N72_rgb_ByteTrack']
    assert baseline['metrics39']['HOTA'] == 29.529
    original = read(work / 'native_common_test_backfill_test/NC36_rgb_ByteTrack/metrics.json')
    assert baseline['metrics39'] == original['trackeval']
    assert baseline['official_CSV_sha256'] == sha256(work / 'native_common_test_backfill_test/NC36_rgb_ByteTrack/trackeval_output/pedestrian_detailed.csv')
    removal_deltas = {role: dict(case=key,
        Dev_delta_pp=dev[roles['primary']]['unrounded_combined_HOTA']-dev[key]['unrounded_combined_HOTA'],
        Test_delta_pp=primary['unrounded_combined_HOTA']-test[key]['unrounded_combined_HOTA'])
        for role, key in roles.items() if role.startswith('primary_')}
    components = []
    for split, sequences in [('dev', protocol['dev']), ('test', protocol['historical_test'])]:
        for seq in sequences:
            for cue in ['rgb', 'ir']:
                p = work / 'native_neck_seed0_eval_features' / split / seq / cue / 'seed_0/complete.json'
                value = read(p)
                components.append(dict(split=split, sequence=seq, cue=cue, frames=value['frames'],
                    shared_pixel_production_seconds=value['seconds'],
                    producer_peak_GPU_GiB=value['peak_gpu_gib'], producer_sha256=sha256(p),
                    scope='Shared encoding of both fixed proposal frontends, not individual-method FPS '
                          'or full-system GPU peak. Source detector/point production/IR generation excluded.'))
    fits = {}
    for head in study['heads']:
        p = Path(head['output']) / 'complete.json'; value = read(p)
        assert value['seed'] == 0 and value['epochs'] == freeze['common_fit_epochs']
        fits[head['id']] = dict(complete_sha256=sha256(p), executed_query_epochs=value['epochs'],
            final_fit_invocation_seconds=value['seconds'], fit_peak_GPU_GiB=value['peak_gpu_gib'],
            cost_scope='Last fit invocation only; retain all controller phase logs for cumulative cost.',
            encoder_epochs=72, original_encoder_retrained=False)
    report = dict(status='COMPLETE_NECK72_SELECTED_ALL7TEST_REQUIRES_FULL_SYSTEM_COST_AUDIT',
        rows=rows, cases=len(rows), all_normal_Dev_curves=curve_records, full_Test_frames_each=15985, roles=roles,
        primary=roles['primary'], primary_unrounded_Test_HOTA=primary['unrounded_combined_HOTA'],
        primary_delta_vs_original_RGB_pp=primary['unrounded_combined_HOTA']-baseline['unrounded_combined_HOTA'],
        primary_delta_vs_paired_extra_RGB_pp=primary['unrounded_combined_HOTA']-test[roles['primary_paired_extra_RGB']]['unrounded_combined_HOTA'],
        removal_deltas=removal_deltas, head_fits=fits, pixel_production_components=components,
        selection_frozen_sha256=sha256(freeze_path), study_sha256=sha256(STUDY),
        full_system_runtime_complete=False, component_costs_are_not_full_system_FPS=True,
        full_system_cost_audit_still_required=True, source_detector_and_feature_extraction_not_benchmarked_here=True,
        no_Test_based_primary_selection=True, fixed_seed=0, additional_seed_work=False,
        intervention_scope=study['intervention_scope'], limitations=study['limitations'],
        scientific_assessment_required=True, supports_new_architecture_count=False,
        supported_innovations=0, goal_met=False)
    stem = DOCS / 'PERCEPTUAL_MFT_20261124_NECK72_SELECTED_TEST'
    assert not stem.with_suffix('.json').exists() and not stem.with_suffix('.md').exists()
    dump(stem.with_suffix('.json'), report)
    lines = ['# neck72固定seed0完整Test', '',
        '每个冻结对象均覆盖7序列/15985帧；负结果全部保留，没有按Test改变主方法。', '',
        '| 条件 | Dev HOTA | Test HOTA | Test IDF1 | Test DetA | Test AssA | Test MOTA | Test IDSW |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        v = row['Test_metrics39']
        lines.append(f'| {row["case"]["id"]} | {row["Dev_HOTA"]:.4f} | {row["Test_HOTA"]:.4f} | '
            f'{v["IDF1"]:.3f} | {v["DetA"]:.3f} | {v["AssA"]:.3f} | {v["MOTA"]:.3f} | {v["IDSW"]:.0f} |')
    lines += ['', study['intervention_scope'], '',
        '各条件JSON含全部39摘要及每序列/COMBINED的47字段，官方276列CSV未经改写。'
        'replay_FPS为带只读观察器的缓存CPU回放；像素生产秒数同时服务两前端。'
        '它们均不等于完整系统FPS，完整运行成本仍未完成。表征修复不增加架构/创新计数，Goal未完成。']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    subprocess.run([sys.executable, 'tools/export_perceptual_metrics47.py'], cwd=ROOT,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
    print(report['status'], len(rows), flush=True)


if __name__ == '__main__':
    main()
