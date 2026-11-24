#!/usr/bin/env python3
"""Publish complete native study tables/provenance; never raw tracks or weights."""
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.native_pyramid_design import FRESH
from tools.native_pyramid_experiment import STUDY, context, read, registry
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.summarize_pseudo_ir_comparison import detailed_metrics


def metric_file(work, split, key):
    return work / 'native_pyramid_evaluation' / split / key / 'metrics.json'


def export_row(work, split, key, tables):
    path = metric_file(work, split, key)
    data = read(path)
    assert data['status'] == 'COMPLETE'
    manifest = data['manifest']
    assert manifest['frames'] == {'dev': 4834, 'test': 15985}[split]
    assert len(data['trackeval']) == 39 and manifest['study_sha256'] == sha256(STUDY)
    csv_path = path.parent / 'trackeval_output/pedestrian_detailed.csv'
    with csv_path.open(newline='') as stream: raw = list(csv.DictReader(stream))
    assert len(raw) == len(manifest['sequences'])+1 and len(raw[0]) == 276
    output = tables / split / (key+'.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): assert sha256(output) == sha256(csv_path)
    else: shutil.copy2(csv_path, output)
    combined = next(r for r in raw if r['seq'] not in manifest['sequences'])
    return dict(id=key, split=split, config=manifest['config'], config_sha256=manifest['config_sha256'],
        metrics=data['trackeval'], per_sequence=detailed_metrics(csv_path, manifest['sequences'], data['trackeval']),
        unrounded_combined_HOTA=100*float(combined['HOTA___AUC']),
        metrics_file=str(path), metrics_sha256=sha256(path),
        official_CSV_file=str(output.relative_to(ROOT)), official_CSV_sha256=sha256(csv_path),
        frames=manifest['frames'], sequences=manifest['sequences'], coverage=manifest['coverage'],
        source_manifest_references=manifest['sources'])


def main():
    plan, study, protocol, calibration, work, checkpoint = context()
    registered = registry(work)
    state = read(work / 'native_pyramid_study_state.json')
    assert state['status'] == 'COMPLETE_NATIVE_STUDY_REQUIRES_SCIENTIFIC_ASSESSMENT'
    freeze = read(work / 'native_pyramid_Test_selection_frozen.json')
    assert freeze['complete_Dev_before_Test'] and freeze['study_sha256'] == sha256(STUDY)
    dev_path = work / 'native_pyramid_complete_Dev.json'
    assert sha256(dev_path) == freeze['complete_Dev_sha256']
    dev = read(dev_path)
    runtime_path = work / 'native_pyramid_individual_runtime.json'
    runtime = read(runtime_path)
    assert runtime['status'] == 'COMPLETE_NATIVE_INDIVIDUAL_RUNTIME'
    assert runtime['specification_sha256'] == freeze['specification_sha256']
    stem = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261124_NATIVE_PYRAMID_FORMAL'
    assert not stem.with_suffix('.json').exists() and not stem.with_suffix('.md').exists()
    tables = ROOT / 'docs/fusion_research/perceptual_20261122_official_tables/native_pyramid'
    test = {r['id']: export_row(work, 'test', r['id'], tables) for r in freeze['cases']}
    all_dev = {r['id']: export_row(work, 'dev', r['id'], tables) for r in dev['rows']}
    roles = freeze['roles']
    primary, rgb, fresh, matched, gray = (test[roles[k]] for k in
        ['primary_IR', 'source_RGB_canonical', 'source_RGB_fresh', 'matched_RGB', 'matched_gray'])
    assert rgb['metrics']['HOTA'] == 29.529
    off, wrong = test[roles['primary_IR_unavailable']], test[roles['primary_IR_image_roll']]
    assert off['metrics'] == fresh['metrics'] and off['official_CSV_sha256'] == fresh['official_CSV_sha256']
    assert all(off['coverage'][s]['output_sha256'] == fresh['coverage'][s]['output_sha256'] for s in protocol['historical_test'])
    required_target = max(34.529, fresh['unrounded_combined_HOTA']+5)
    target_met = primary['unrounded_combined_HOTA'] >= required_target
    primary_dev = freeze['selection']['primary']
    wrong_dev = all_dev[primary_dev['id']+'_roll']
    current_input_effect = (primary['unrounded_combined_HOTA'] > wrong['unrounded_combined_HOTA'] and
        all_dev[primary_dev['id']]['unrounded_combined_HOTA'] > wrong_dev['unrounded_combined_HOTA'])
    strong_controls = (primary['unrounded_combined_HOTA'] > max(matched['unrounded_combined_HOTA'], gray['unrounded_combined_HOTA']))
    contributions = {}
    for removal, choices in freeze['removals'].items():
        differences = {}
        for policy, choice in choices.items():
            other = test[roles['removal_'+removal+'_'+policy]]
            differences[policy] = dict(
                Dev_delta_pp=all_dev[primary_dev['id']]['unrounded_combined_HOTA']-all_dev[choice['id']]['unrounded_combined_HOTA'],
                Test_delta_pp=primary['unrounded_combined_HOTA']-other['unrounded_combined_HOTA'],
                test_case=other['id'], Dev_case=choice['id'])
        candidate = strong_controls and current_input_effect and all(
            value['Dev_delta_pp'] > 0 and value['Test_delta_pp'] > 0 for value in differences.values())
        contributions[removal] = dict(comparisons=differences, positive_removal_evidence_candidate=candidate,
            independent_innovation_established=False)
    report = dict(status='COMPLETE_NATIVE_PYRAMID_FORMAL_STUDY_REQUIRES_RESEARCH_AUDIT',
        study_sha256=sha256(STUDY), registry_sha256=sha256(work / 'native_pyramid_pipeline_registered.json'),
        complete_Dev_sha256=sha256(dev_path), frozen_Test_selection_sha256=sha256(work / 'native_pyramid_Test_selection_frozen.json'),
        complete_state_sha256=sha256(work / 'native_pyramid_study_state.json'),
        Dev_rows=list(all_dev.values()), Test_rows=list(test.values()), roles=roles,
        fixed_baseline_HOTA=29.529, fixed_target_HOTA=34.529, fresh_B1_fair_target_HOTA=required_target,
        numeric_target_met=target_met, primary_minus_fixed_RGB_HOTA=primary['metrics']['HOTA']-29.529,
        primary_minus_fresh_RGB_HOTA=primary['unrounded_combined_HOTA']-fresh['unrounded_combined_HOTA'],
        primary_minus_matched_RGB_HOTA=primary['unrounded_combined_HOTA']-matched['unrounded_combined_HOTA'],
        primary_minus_matched_gray_HOTA=primary['unrounded_combined_HOTA']-gray['unrounded_combined_HOTA'],
        current_IR_image_effect_supported=current_input_effect, primary_exceeds_strong_controls=strong_controls,
        unavailable_full_tracks_all39_CSV_exact_to_fresh_RGB=True,
        removal_evidence=contributions,
        individual_runtime=runtime, source_hashes=registered['source_hashes'],
        complete_Train_epochs_each=freeze['complete_epochs_each'], fixed_seed=0, additional_seed_work=False,
        Test_based_selection=False, historical_Test_previously_exposed=True,
        supported_innovations=0, goal_met=False,
        remaining_acceptance='Numerical and removal evidence alone do not complete the goal. Audit clear2–3independent mechanisms, single/cumulative combinations, four current-IR architectures, all formal-candidate coverage, prior-art boundaries, full cost/limitations and final report before any goal completion.',
        limitations=study['limitations'], runtime_scope=runtime['sample_scope'])
    dump(stem.with_suffix('.json'), report)
    lines = ['# 原生辅助金字塔完整正式实验', '',
        '主模型由完整Dev选择，全部正式条件执行完整7Test/15985帧，未按Test改选。保留两种RGB基线、匹配RGB/灰度、实际IR图像roll、模态缺失及触发的同预算去除条件。', '',
        '| 比较角色 | HOTA | IDF1 | DetA | AssA | MOTA | IDSW |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for role, key in roles.items():
        m = test[key]['metrics']
        lines.append(f"| {role} | {m['HOTA']:.3f} | {m['IDF1']:.3f} | {m['DetA']:.3f} | {m['AssA']:.3f} | {m['MOTA']:.3f} | {m['IDSW']:.0f} |")
    lines += ['', f"固定主模型相对规范RGB：{report['primary_minus_fixed_RGB_HOTA']:+.3f} HOTA点；相对新B1RGB：{report['primary_minus_fresh_RGB_HOTA']:+.3f}。",
        f"同时核对固定34.529与更强B1RGB+5的未舍入目标，数值目标通过：{target_met}；这不自动构成整体Goal完成。", '',
        '缺失辅助的完整轨迹、39摘要和整个官方CSV精确等于新B1RGB。所有Dev/Test摘要、逐序列值、276列官方CSV、来源版本和检查点摘要均保留；未发布图像、特征、权重或原始轨迹。', '',
        '逐模型成本包含原图读取、预处理、RGB编码、教师头、辅助编码/融合、所需光流、NMS和ByteTrack。成本只测每段Test前32帧、8帧预热后24帧，共168测量帧；不是完整Test逐模型FPS，也不包括已离线生成的DiffV2IR。GPU可能与其他用户共享，未用新种子重复估计波动。', '',
        '去除正差值仅是贡献候选证据，仍须按原任务要求核对单模块/累积组合、2–3项明确独立机制、当前IR四类架构、全部正式候选范围与先验工作。此导出不自动计创新或宣布Goal完成。历史Test已暴露、RGB预训练见过Dev、伪IR由RGB生成等限制继续适用。']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'], Test_cases=len(test), Dev_cases=len(all_dev), numeric_target_met=target_met,
                          goal_met=False)), flush=True)


if __name__ == '__main__': main()
