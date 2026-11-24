#!/usr/bin/env python3
"""Freeze the six completed CGR fits from their original complete Dev curves.

This is a source/coverage audit and registration, not an inference controller.
It neither launches GPU work nor makes the legacy GPU2/3 extractors launchable.
The current visual primary and every running pipeline remain unchanged.
"""
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from tools.run_pseudo_ir_comparison import dump, sha256

DOCS = ROOT / 'docs/fusion_research'
SPEC = ROOT / 'configs/perceptual_correlation_recovery_backfill_20261124.json'
REPAIR = ROOT / 'configs/perceptual_correlation_recoverability_20261123.json'
STAGES = ['correlation_recoverability_pilot_dev', 'correlation_recoverability_staged_dev']
EXPECTED = {'rgb_full_s0', 'gray_full_s0', 'ir_full_s0', 'ir_aligned_s0',
            'ir_blanket_counterfactual_s0', 'ir_no_curriculum_s0'}


def read(path):
    return json.loads(Path(path).read_text())


def best(rows):
    # A new coverage registration, fixed before any CGR Test output. Full Dev
    # only; equal-HOTA ties favor earlier snapshots, then raw, then case ID.
    return min(rows, key=lambda row: (-row['HOTA'], row['epoch'],
        row['output_policy'] != 'raw', row['source_Dev_case']))


def main():
    assert not SPEC.exists(), 'Do not overwrite frozen choices'
    report_path = DOCS / 'PERCEPTUAL_MFT_20261124_CORRELATION_RECOVERY_BACKFILL_REGISTRATION.json'
    markdown_path = report_path.with_suffix('.md')
    assert not report_path.exists() and not markdown_path.exists()
    plan_path = ROOT / 'configs/perceptual_mft_20261122.json'
    assert sha256(plan_path) == '388e636d45840f3da8eae44cde2efc4731eeceadc7c169a39f19bf30488ecc03'
    plan = read(plan_path)
    work = Path(plan['work_root'])
    protocol = read(ROOT / plan['protocol'])
    assert sha256(ROOT / plan['protocol']) == 'fe3db8e991facfcd74981f76aa5c2271c18a9554386501eb2c2dd30a12bf6c60'
    assert sha256(REPAIR) == '23689be1d7e5e1b9584be2c97368421c041180e7c6a9baee5d4c8591985f8e58'
    policy_path = ROOT / 'configs/perceptual_single_seed_policy_20261123.json'
    policy = read(policy_path)
    assert policy['fixed_seed'] == 0 and not policy['additional_seed_inference']
    groups = {name: [] for name in sorted(EXPECTED)}
    artifacts, checkpoints, completed = {}, {}, {}
    verified_hashes = {}

    def pin(path):
        path = Path(path)
        if str(path) not in verified_hashes:
            verified_hashes[str(path)] = sha256(path)
        return verified_hashes[str(path)]

    for stage in STAGES:
        for path in sorted((work / stage).glob('*/metrics.json')):
            value = read(path)
            manifest = value['manifest']
            assert value['status'] == 'COMPLETE' and manifest['split'] == 'dev'
            assert manifest['frames'] == 4834 and manifest['sequences'] == protocol['dev']
            assert len(value['trackeval']) == 39
            assert not manifest['GT_inference_input'] and not manifest['future_frame_input']
            assert not manifest['previous_outputs_rewritten']
            config = manifest['config']
            if not config.get('checkpoint'):
                assert config['baseline']
                continue
            checkpoint = Path(config['checkpoint'])
            assert checkpoint.parent.parent == work / 'correlation_recoverability_fit'
            assert checkpoint.parent.name in EXPECTED and config['seed'] == 0
            assert config['fusion_plan_sha256'] == sha256(REPAIR)
            for seq in protocol['dev']:
                track = path.parent / 'track_results' / (seq + '.txt')
                assert pin(track) == manifest['coverage'][seq]['output_sha256']
                source = manifest['sources'][seq]
                assert pin(source['path']) == source['sha256']
            row = dict(source_Dev_case=config['id'], source_Dev_metrics=str(path),
                source_Dev_metrics_sha256=pin(path), HOTA=value['trackeval']['HOTA'],
                official_CSV_sha256=pin(path.parent / 'trackeval_output/pedestrian_detailed.csv'),
                cue=config['cue'], variant=config['variant'], seed=0, epoch=config['epoch'],
                output_policy=config['output_policy'], checkpoint=str(checkpoint),
                checkpoint_sha256=pin(checkpoint))
            groups[checkpoint.parent.name].append(row)
    assert sum(map(len, groups.values())) == 48
    for name, curve in groups.items():
        assert {(row['epoch'], row['output_policy']) for row in curve} == {
            (epoch, output) for epoch in [6, 12, 18, 24] for output in ['priority', 'raw']}
        path = work / 'correlation_recoverability_fit' / name / 'complete.json'
        info = read(path)
        assert info['status'] == 'COMPLETE_TRAIN_ONLY_ADAPTER' and info['epochs'] == 24
        assert info['successful_steps_this_invocation'] == 6912 and len(info['history']) == 24
        assert info['seed'] == 0 and info['config_sha256'] == sha256(REPAIR)
        assert info['trainable_parameters'] == 217619
        assert info['Dev_fit_rows'] == 0 and not info['Test_read']
        assert info['Train_videos'] == protocol['train']
        assert info['source_RGB_sha256'] == 'fc27f6bbfb961d409247a765c8704b2dd83121a0160df672933b81d36aa27e46'
        for source in info['Train_sources'].values():
            assert source['frames'] == 192 and pin(source['manifest']) == source['sha256']
        for source in info['eligibility_sources'].values():
            assert pin(source['manifest']) == source['sha256']
        assert pin(ROOT / 'tools/train_recoverable_correlation.py') == info['source_sha256']
        assert pin(ROOT / 'yolox/models/local_correlation_repair.py') == info['adapter_source_sha256']
        assert pin(ROOT / 'yolox/models/frozen_head_repair.py') == info['loss_source_sha256']
        assert pin(ROOT / 'yolox/models/recoverable_curriculum.py') == info['curriculum_source_sha256']
        completed[name] = dict(complete_path=str(path), complete_sha256=pin(path),
            epochs=24, updates=6912, trainable_parameters=217619,
            unique_Train_frames=1152, sampling='192 fixed frames per video, repeated24times',
            control_disclosure=info['control_disclosure'])
        artifacts[str(path)] = pin(path)
    selected = {name: best(rows) for name, rows in groups.items()}
    primary = selected['ir_full_s0']
    role_rows = {f'normal_{cue}': selected[f'{cue}_full_s0'] for cue in ['rgb', 'gray', 'ir']}
    for variant in ['aligned', 'blanket_counterfactual', 'no_curriculum']:
        curve = groups[f'ir_{variant}_s0']
        role_rows[variant + '_own_Dev_best'] = best(curve)
        paired = [row for row in curve if (row['epoch'], row['output_policy']) == (
            primary['epoch'], primary['output_policy'])]
        assert len(paired) == 1
        role_rows[variant + '_paired'] = paired[0]
    cases, roles, identities = [], {}, {}
    for role, row in role_rows.items():
        identity = (row['checkpoint'], row['output_policy'])
        if identity not in identities:
            name = f"CRB_{row['cue']}_{row['variant']}_e{row['epoch']:02d}_{row['output_policy']}"
            identities[identity] = name
            cases.append(dict(id=name, source=row, intervention='normal'))
        roles[role] = identities[identity]
    for intervention in ['unavailable', 'IR_image_spatial_roll']:
        name = 'CRB_IR_' + intervention
        roles[intervention] = name
        cases.append(dict(id=name, source=primary, intervention=intervention))
    for convention in ['fresh_B4', 'canonical_B4']:
        name = 'CRB_source_RGB_' + convention
        roles[convention] = name
        cases.append(dict(id=name, baseline=True, convention=convention))
    for case in cases:
        row = case.get('source')
        if row is None or row['checkpoint'] in checkpoints:
            continue
        saved = torch.load(row['checkpoint'], map_location='cpu', weights_only=False)
        meta = saved['metadata']
        assert saved['completed_epochs'] == row['epoch'] and saved['config_sha256'] == sha256(REPAIR)
        assert (meta['cue'], meta['variant'], meta['seed']) == (row['cue'], row['variant'], 0)
        assert sum(t.numel() for t in saved['adapter'].values()) == 217619
        assert all(torch.isfinite(t).all() for t in saved['adapter'].values())
        checkpoints[row['checkpoint']] = row['checkpoint_sha256']
    visual_specs = ['perceptual_CGA_formal_test_20261123.json',
        'perceptual_structural_nocycle_test_20261123.json',
        'perceptual_structural_factorial_test_20261123.json']
    inspected = {}
    for name in visual_specs:
        path = ROOT / 'configs' / name
        model_paths = {str(Path(row['checkpoint']).parent) for row in read(path)['models'] if row.get('checkpoint')}
        assert not model_paths.intersection({str(work / 'correlation_recoverability_fit' / name) for name in EXPECTED})
        inspected[name] = pin(path)
    report = dict(status='FROZEN_CGR6_FROM_COMPLETE_ORIGINAL48DEV_NOT_LAUNCH_READY',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), cases=cases, roles=roles,
        fixed_seed=0, additional_seed_work=False, GPU_work=False, new_fit_or_inference=False,
        source_plan_sha256=pin(plan_path), repair_plan_sha256=pin(REPAIR),
        single_seed_policy_sha256=pin(policy_path), completed_fits=completed,
        original_Dev_curves=groups, checkpoint_sources=checkpoints,
        artifact_sources=artifacts, completed_visual_Test_specs_inspected=inspected,
        required_sequences=protocol['historical_test'], required_frames_each=15985,
        selection='Original complete4834frame Dev only. Highest HOTA, earlier epoch, raw then ID ties. '
            'Keep each of6fits own best;3removals also paired to fullIR epoch/output policy. '
            'Identical checkpoint/output roles share a case. No Test accuracy read or global primary change.',
        training_scope='Same217619parameter local-correlation architecture as parentCGA, with curriculum '
            'and validity-mask training changes; not another independently validated architecture. '
            'Only1152uniqueTrainframes, not24completeTrainepochs. MatchedRGB/gray had privileged '
            'IR-conditioned Train eligibility; RGB-only inference does not erase that training disclosure.',
        execution_status='REGISTRATION_ONLY_PENDING_GPU4_5_PIPELINE_AND_ENGINEERING',
        execution_requirements=['Do not use legacy extractors on physical2/3 or spoof their CUDA mapping.',
            'Implement isolated physical4/5 inference after the existing fullTrain/legacy/nativepyramid work; no interruption.',
            'Verify all selected original Dev tracks,39summary fields and entire officialCSV exactly.',
            'Validate actualsource/zero/missing and fullTest provenance before reading new Test accuracy.',
            'Report every frozen case on all7Test, all47metric views, sourceproof, gate statistics and actual per-method costs.'],
        storage='Reuse sourceRGB/IR files and frozenweights.64GiB work cap,40GiB independent free reserve. '
            'No image/FPN copies or largefiles in Git.',
        source_sha256=sha256(__file__), Test_accuracy_read=False,
        full_formal_candidate_coverage=False, goal_met=False, supported_innovations=0)
    dump(SPEC, report)
    dump(report_path, dict(status=report['status'], specification=str(SPEC.relative_to(ROOT)),
        specification_sha256=sha256(SPEC), completed_fits=6, original_Dev_rows=48,
        comparison_roles=len(roles), distinct_Test_cases=len(cases),
        selected_Dev={name: {key: row[key] for key in ['source_Dev_case', 'HOTA', 'epoch', 'output_policy']}
                      for name, row in selected.items()}, execution_status=report['execution_status'],
        source_sha256=sha256(__file__), GPU_work=False, Test_accuracy_read=False,
        current_primary_unchanged=True, goal_met=False, supported_innovations=0))
    lines = ['# 六个CGR训练条件：完整Test缺口及冻结对象', '',
        '本次只读原始训练/完整Dev、核验实际检查点和来源。没有GPU工作、新训练、推理或Test准确率读取，当前主方法与正在运行的队列未变。', '',
        '| 已完成训练条件 | 自身完整Dev最佳 | epoch | 输出 |',
        '| --- | ---: | ---: | --- |']
    for name, row in sorted(selected.items()):
        lines.append(f"| {name} | {row['HOTA']:.3f} | {row['epoch']} | {row['output_policy']} |")
    lines += ['', f'六个条件均执行24次1152帧缓存遍历、6912更新、217619参数；每个有四个快照/两种读出的8项完整Dev，共48项。已完成的三份视觉Test规格没有这些训练目录。现按原Dev冻结{len(roles)}个比较角色、{len(cases)}个不重复Test对象，包含自身最佳与三个配对去除、真实IR错位/关闭以及新鲜/规范RGB。', '',
        '完整IR的Dev为47.628，直接RGB为47.579，去课程则49.186；这不支持课程贡献。仍保留并计划补齐完整Test，不据负结果删除正式条件，也不将这套同架构训练变化算作新的融合架构。', '',
        '训练仅1152幅独立画面，不能称24个完整Train epoch。RGB/灰度也使用IR决定的训练可恢复性选择，属于有额外训练信息的强对照；推理不读IR不能抹去该限制。', '',
        '当前仅登记，未启动或冒充可执行队列。旧提取器写死GPU2/3，需要独立的4/5卡实现及真实来源、零/缺失回退、原Dev全轨迹/39摘要/整个CSV复现门槛；待现有全量Train、结构补测和原生金字塔流程结束后执行，不能抢占或停止健康作业。最终仍须全7序列/15985帧、47字段及门控/成本，当前目标与贡献均未完成。', '',
        '完整冻结配置：`configs/perceptual_correlation_recovery_backfill_20261124.json`。已有历史Test暴露、RGB预训练见过Dev以及伪IR由RGB生成的限制继续适用。']
    markdown_path.write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=report['status'], completed_fits=6, original_Dev_rows=48,
        comparison_roles=len(roles), distinct_Test_cases=len(cases), GPU_work=False, goal_met=False)))


if __name__ == '__main__':
    main()
