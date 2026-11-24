#!/usr/bin/env python3
"""Publish all completed frozen CGR cases, source/cost proofs and47field views."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.correlation_recovery_workflow import DOCS, SPEC, WORKFLOW, context, read, registered
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.summarize_pseudo_ir_comparison import detailed_metrics


def export_row(work, split, case):
    path = work / ('correlation_recovery_'+split) / case['id'] / 'metrics.json'
    value = read(path)
    manifest = value['manifest']
    assert value['status'] == 'COMPLETE' and manifest['workflow_sha256'] == sha256(WORKFLOW)
    assert manifest['frames'] == (4834 if split == 'dev' else 15985) and len(value['trackeval']) == 39
    source = path.parent / 'trackeval_output/pedestrian_detailed.csv'
    with source.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(manifest['sequences'])+1 and len(rows[0]) == 276
    combined = next(row for row in rows if row['seq'] == 'COMBINED')
    output = DOCS / 'perceptual_20261122_official_tables/correlation_recovery' / split / (case['id']+'.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        assert sha256(output) == sha256(source)
    else:
        shutil.copy2(source, output)
    gates = {}
    for sequence, meta in manifest['sources'].items():
        assert sha256(meta['pixel_manifest']) == meta['pixel_manifest_sha256']
        producer = read(meta['pixel_manifest'])
        if not case.get('baseline'):
            gate = producer['gate_statistics'][case['id']]
            assert all(row['frames'] == manifest['coverage'][sequence]['frames'] for row in gate)
            gates[sequence] = gate
    return dict(id=case['id'], config=case, split=split, metrics=value['trackeval'],
        unrounded_combined_HOTA=100*float(combined['HOTA___AUC']),
        per_sequence=detailed_metrics(source, manifest['sequences'], value['trackeval']),
        frames=manifest['frames'], sequences=manifest['sequences'], source_metrics_sha256=sha256(path),
        official_CSV=str(output.relative_to(ROOT)), official_CSV_sha256=sha256(source),
        source_references=manifest['sources'], coverage=manifest['coverage'], gate_statistics=gates)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plan, spec, protocol, parent, calibration, work = context()
    workflow = registered()
    state = read(work / 'correlation_recovery_workflow_state.json')
    assert state['status'] == 'COMPLETE_CGR_BACKFILL_REQUIRES_SCIENTIFIC_ASSESSMENT'
    freeze = read(work / 'correlation_recovery_Test_selection_frozen.json')
    assert freeze['workflow_sha256'] == sha256(WORKFLOW)
    assert freeze['original_selection_unchanged'] and not freeze['Test_accuracy_read_for_selection']
    runtime = read(work / 'correlation_recovery_individual_runtime.json')
    assert runtime['workflow_sha256'] == sha256(WORKFLOW) and len(runtime['rows']) == 12
    stem = DOCS / 'PERCEPTUAL_MFT_20261124_CGR_FORMAL_BACKFILL'
    assert not stem.with_suffix('.json').exists() and not stem.with_suffix('.md').exists()
    dev = {row['id']: export_row(work, 'dev', row) for row in spec['cases']}
    test = {row['id']: export_row(work, 'test', row) for row in spec['cases']}
    roles = spec['roles']
    primary, baseline, fresh = [test[roles[key]] for key in ['normal_ir', 'canonical_B4', 'fresh_B4']]
    assert baseline['metrics']['HOTA'] == 29.529
    off = test[roles['unavailable']]
    assert off['metrics'] == fresh['metrics'] and off['official_CSV_sha256'] == fresh['official_CSV_sha256']
    assert all(off['coverage'][seq]['output_sha256'] == fresh['coverage'][seq]['output_sha256']
               for seq in protocol['historical_test'])
    removals = {}
    for variant in ['aligned', 'blanket_counterfactual', 'no_curriculum']:
        removals[variant] = {}
        for role in ['paired', 'own_Dev_best']:
            key = roles[variant+'_'+role]
            removals[variant][role] = dict(case=key,
                Dev_delta_pp=dev[roles['normal_ir']]['unrounded_combined_HOTA']-dev[key]['unrounded_combined_HOTA'],
                Test_delta_pp=primary['unrounded_combined_HOTA']-test[key]['unrounded_combined_HOTA'])
    differences = {seq: read(work / 'correlation_recovery_test/sequence_manifests' / (seq+'.json'))['source_tail_differences']
                   for seq in protocol['historical_test']}
    report = dict(status='COMPLETE_ALL6_TRAINED_CGR_FITS_FROZEN12_FULL7TEST_BACKFILL',
        workflow_sha256=sha256(WORKFLOW), source_specification_sha256=sha256(SPEC),
        completed_state_sha256=sha256(work / 'correlation_recovery_workflow_state.json'),
        original48Dev_curves_in_frozen_specification=True, actual_Dev=list(dev.values()), Test_rows=list(test.values()),
        roles=roles, fixed_seed=0, additional_seed_work=False, actual_GPU_probe_proofs={gpu:
            sha256(work / 'correlation_recovery_formal/engineering' / ('gpu'+gpu) / 'complete.json') for gpu in ['4', '5']},
        complete_Dev_proof_sha256=sha256(work / 'correlation_recovery_full_Dev_proof.json'),
        runtime=runtime, source_tail_differences=differences, module_comparisons=removals,
        primary_minus_canonical_RGB_HOTA=primary['unrounded_combined_HOTA']-baseline['unrounded_combined_HOTA'],
        primary_minus_fresh_RGB_HOTA=primary['unrounded_combined_HOTA']-fresh['unrounded_combined_HOTA'],
        primary_minus_matched_RGB_HOTA=primary['unrounded_combined_HOTA']-test[roles['normal_rgb']]['unrounded_combined_HOTA'],
        primary_minus_matched_gray_HOTA=primary['unrounded_combined_HOTA']-test[roles['normal_gray']]['unrounded_combined_HOTA'],
        numeric_target_met=primary['unrounded_combined_HOTA'] >= max(34.529, fresh['unrounded_combined_HOTA']+5),
        current_global_visual_primary_unchanged=True, goal_met=False, supported_innovations=0,
        scope='Formal coverage of six previously trained217619parameter CGR conditions, not a new architecture. '
            'Each24cached epochs/6912updates on1152unique Trainframes; matchedRGB/gray used privileged '
            'IR-conditioned Train eligibility. Frozen models/readouts, ownDev-best and paired removals. '
            'All7Test/15985frames per condition; actualIR image roll/off. No Test-based promotion. '
            'CGR original curriculum Dev evidence remains negative. HistoricalTest exposed, RGB pretraining '
            'saw currentDev, pseudoIR derives fromRGB. Full goal/contribution audit still required.')
    dump(stem.with_suffix('.json'), report)
    lines = ['# 六个既有CGR训练条件的完整正式补测', '', report['scope'], '',
        '| 冻结比较角色 | HOTA | IDF1 | DetA | AssA | MOTA | IDSW |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for role, key in roles.items():
        m = test[key]['metrics']
        lines.append(f"| {role} | {m['HOTA']:.3f} | {m['IDF1']:.3f} | {m['DetA']:.3f} | {m['AssA']:.3f} | {m['MOTA']:.3f} | {m['IDSW']:.0f} |")
    lines += ['', f"冻结CGR主IR相对规范RGB为{report['primary_minus_canonical_RGB_HOTA']:+.3f}点，未据本批Test更换全局主方法或模型。", '',
        '全部条件保留39摘要、逐序列结果、完整276列官方CSV及47字段视图，缺失与新鲜RGB的完整轨迹、39摘要及CSV精确一致；历史规范RGB的完整官方CSV也精确复现。', '',
        '原Dev逐条件延续视频间编号以复现旧轨迹；新Test各视频编号从1开始，跟踪/关联证据始终在视频开始重置，没有事后重新编号。所有方法在同一视频共享同一张物理卡，最多同时4/5两卡。', '',
        runtime['scope'], '',
        '门控统计只是实现诊断，不计科学贡献。CGR是既有局部相关架构的训练变化，2–3项贡献、当前IR四类方向、单模块/累积组合和完整研究Goal仍须另外核实。']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    subprocess.run([sys.executable, 'tools/export_perceptual_metrics47.py'], cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
    print(json.dumps(dict(status=report['status'], cases=12, full_Test_frames_each=15985,
        numeric_target_met=report['numeric_target_met'], goal_met=False)), flush=True)


if __name__ == '__main__':
    main()
