#!/usr/bin/env python3
"""Describe completed minimum6 Dev evidence without changing any selection."""
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs/fusion_research'
SOURCE = DOCS / 'PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV.json'
EXPECTED = '63426a8689f27ce662557b2df1d737db1d55d18af7e716bdd1374bdc290d688e'
STEM = DOCS / 'PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV_ANALYSIS_ZH'
FIELDS = ['HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW']
SEQUENCES = ['BT-005', 'SN-013', 'COMBINED']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only analysis required')
    require(sha(SOURCE) == EXPECTED, 'Exact completed62-case audit required')
    data = read(SOURCE)
    require(data['status'] == 'COMPLETE_INDEPENDENT_MINIMUM6_FULL_DEV_AUDIT_NOT_FINAL_STUDY' and
            data['frames'] == 4834 and data['stage'] == 6 and data['Test_results_read'] is False and
            len(data['rows']) == 62 and data['unavailable_all47_per_sequence_full_CSV_full_MOT_exact_to_fresh_RGB'],
            'Full original Dev and missing-modality checks are required')
    require(not any(STEM.with_suffix(suffix).exists() for suffix in ['.json', '.md']),
            'Preserve earlier scientific analysis')
    lookup = {row['id']: row for row in data['rows']}
    own = data['six_own_rounded_official_Dev_best']
    require(len(lookup) == 62 and len(own) == 6, 'All registered curves and six own choices required')
    bindings = []
    for row in data['rows']:
        path = ROOT / row['official_CSV']['file']
        require(sha(path) == row['official_CSV']['sha256'], 'Raw official CSV changed')
        bindings.append(row['official_CSV'])

    def metrics(key):
        return {field: lookup[key]['combined47'][field] for field in FIELDS}

    def point(key, sequence):
        return next(row for row in lookup[key]['per_sequence47'] if row['seq'] == sequence)

    def rank(key):
        row = lookup[key]
        values, config = row['original_official_metrics39'], row['config']
        return (-values['HOTA'], -values['AssA'], -values['IDF1'], config['epoch'],
                0 if config['mode'] == 'quality' else 1)

    # A descriptive comparison of already registered own-Dev-best checkpoints.
    # It is not a new freeze, Test choice, budget decision, or fit authorization.
    strongest = {cue: min([own[mode+'_'+cue] for mode in ['quality', 'temporal']], key=rank)
                 for cue in ['rgb', 'gray', 'ir']}
    baselines = ['NP_source_canonical_RGB', 'NP_source_fresh_RGB']
    all_six = [dict(arm=arm, case=key, epoch=lookup[key]['config']['epoch'], metrics=metrics(key))
               for arm, key in own.items()]
    comparisons, per_sequence, curves, interactions = [], [], [], []
    for mode in ['quality', 'temporal']:
        ir = own[mode+'_ir']
        controls = dict(canonical_RGB=baselines[0], fresh_B1_RGB=baselines[1],
            same_mode_own_best_RGB=own[mode+'_rgb'], same_mode_own_best_gray=own[mode+'_gray'],
            strongest_RGB_across_modes=strongest['rgb'], strongest_gray_across_modes=strongest['gray'],
            same_checkpoint_IR_image_roll=ir+'_roll', same_checkpoint_IR_unavailable=ir+'_off')
        for label, control in controls.items():
            comparisons.append(dict(mode=mode, IR_case=ir, control_label=label, control_case=control,
                delta={field: metrics(ir)[field]-metrics(control)[field] for field in FIELDS}))
            for sequence in SEQUENCES:
                left, right = point(ir, sequence), point(control, sequence)
                per_sequence.append(dict(mode=mode, sequence=sequence, IR_case=ir,
                    control_label=label, control_case=control,
                    delta={field: left[field]-right[field] for field in FIELDS}))
        for cue in ['rgb', 'gray', 'ir']:
            curves.append(dict(mode=mode, cue=cue, points=[dict(epoch=epoch,
                case=f'NP_{mode}_{cue}_e{epoch:02d}', metrics=metrics(f'NP_{mode}_{cue}_e{epoch:02d}'))
                for epoch in range(1, 7)]))
    for epoch in range(1, 7):
        changes = {cue: metrics(f'NP_temporal_{cue}_e{epoch:02d}')['HOTA']-
                        metrics(f'NP_quality_{cue}_e{epoch:02d}')['HOTA'] for cue in ['rgb', 'gray', 'ir']}
        interactions.append(dict(epoch=epoch, temporal_minus_quality_HOTA=changes,
            IR_relative_to_RGB_difference_in_differences=changes['ir']-changes['rgb'],
            IR_relative_to_gray_difference_in_differences=changes['ir']-changes['gray'],
            scope='Descriptive same-epoch architecture/cue interaction, not an isolated module ablation or causal estimate.'))
    temporal = own['temporal_ir']

    def delta(control_label, mode='temporal'):
        return next(row['delta'] for row in comparisons if row['mode'] == mode and row['control_label'] == control_label)

    report = dict(status='COMPLETE_MINIMUM6_DEV_DESCRIPTIVE_ANALYSIS_NOT_TEST_OR_INNOVATION_PROOF',
        source_report=str(SOURCE.relative_to(ROOT)), source_report_sha256=EXPECTED,
        source_CSV_bindings=bindings, source_analysis_tool=str(Path(__file__).relative_to(ROOT)),
        source_analysis_tool_sha256=sha(__file__), split='dev', frames=4834, videos=['BT-005', 'SN-013'],
        cases=62, all_six_own_registered_Dev_best=all_six,
        descriptive_strongest_controls_across_modes=strongest,
        IR_own_best_comparisons=comparisons, per_sequence_comparisons=per_sequence,
        full_normal_curves=curves, all_six_same_epoch_interactions=interactions,
        original_frozen_budget_decision=data['original_frozen_budget_decision'],
        positive_combination_evidence_candidate=delta('same_mode_own_best_RGB')['HOTA'] > 0 and
            delta('same_mode_own_best_gray')['HOTA'] > 0 and delta('same_checkpoint_IR_image_roll')['HOTA'] > 0,
        strongest_RGB_comparison_must_not_be_omitted=True,
        isolated_temporal_module_contribution_established=False,
        convergence_established=False, source_47_metrics_or_official_evaluation_recomputed=False,
        original_frozen_choices_or_budget_changed=False, new_Test_case_selected=False,
        Test_results_read=False, additional_seed_work=False, GPU_work=False,
        supported_innovations=0, goal_met=False,
        limitations=['Only two complete Dev videos; historical Test has previously been exposed.',
            'Original RGB pretraining saw current Dev. PseudoIR is derived from RGB, not a physical thermal sensor.',
            'The favorable same-mode comparison is larger than the comparison against the strongest RGB architecture.',
            'Curves are nonmonotone; videos and snapshots are not independent random repetitions.',
            'Architecture/cue interactions and image-roll responses do not replace matched module removals and full Test.'])
    require(sha(SOURCE) == EXPECTED and all(sha(ROOT/row['file']) == row['sha256'] for row in bindings),
            'Input evidence changed during analysis')
    with STEM.with_suffix('.json').open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')

    lines = ['# 前六轮完整Dev：有组合收益候选，独立创新仍待验证', '',
        '以下数值全部来自两个完整Dev视频、4834帧。全部62项条件及原65个生产／规范源／官方评估worker均完成并实际退出0，独立47字段审计也实际退出0。'
        '这是最低六轮阶段结果，不是完整Test、最终模型选择或收敛结论。', '',
        f"时序IR（{temporal}）HOTA为{metrics(temporal)['HOTA']:.6f}，对同模式各自Dev最佳RGB为"
        f"{delta('same_mode_own_best_RGB')['HOTA']:+.6f}点，对灰度为{delta('same_mode_own_best_gray')['HOTA']:+.6f}。"
        f"对规范直接RGB为{delta('canonical_RGB')['HOTA']:+.6f}，对新鲜B1 RGB为{delta('fresh_B1_RGB')['HOTA']:+.6f}。", '',
        f"跨两种结构的最强RGB是{strongest['rgb']}，HOTA为{metrics(strongest['rgb'])['HOTA']:.6f}；"
        f"时序IR只高{delta('strongest_RGB_across_modes')['HOTA']:+.6f}点。"
        '因此不能只展示较大的同模式差值来宣称突破。该跨结构比较是额外强对照，不替代原预登记的同模式比较或改变选择。', '',
        '| 完整Dev条件 | 检查点 | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label, key in [('规范直接RGB', baselines[0]), ('新鲜B1 RGB', baselines[1])]+[(row['arm'],row['case']) for row in all_six]:
        values = metrics(key)
        lines.append('| '+label+' | '+key+' | '+' | '.join(f'{values[field]:.3f}' if field != 'IDSW'
                     else str(values[field]) for field in FIELDS)+' |')
    lines += ['', f"质量模式IR相对本模式RGB为{delta('same_mode_own_best_RGB','quality')['HOTA']:+.6f}点，未超过对应RGB强对照。"
        '负结果与全部正常曲线继续保留，不能只把时序IR列出来。', '',
        '| 时序IR−对照，完整视频 | 同模式RGB | 同模式灰度 | 跨结构最强RGB | 同检查点图像roll |',
        '| --- | ---: | ---: | ---: | ---: |']
    for sequence in SEQUENCES:
        values = [next(row['delta']['HOTA'] for row in per_sequence if row['mode']=='temporal' and
            row['sequence']==sequence and row['control_label']==label) for label in
            ['same_mode_own_best_RGB','same_mode_own_best_gray','strongest_RGB_across_modes','same_checkpoint_IR_image_roll']]
        lines.append('| '+sequence+' | '+' | '.join(f'{value:+.6f}' for value in values)+' |')
    lines += ['', '时序IR相对同模式RGB在两段都为正，相对灰度在BT-005为负；相对跨结构最强RGB则两段方向反转。'
        '不能把汇总正差值解释成每个场景都稳定获益，也不能用序列标签设计Test路由。', '',
        f"时序IR对同模式RGB的AssA差为{delta('same_mode_own_best_RGB')['AssA']:+.6f}、IDF1为"
        f"{delta('same_mode_own_best_RGB')['IDF1']:+.6f}，而DetA为{delta('same_mode_own_best_RGB')['DetA']:+.6f}、"
        f"MOTA为{delta('same_mode_own_best_RGB')['MOTA']:+.6f}，IDSW多{delta('same_mode_own_best_RGB')['IDSW']:.0f}次。"
        '当前正向主要体现在关联指标，并非所有指标均提升；这不是对HOTA的精确因果分解或对记忆模块的独立归因。', '',
        f"时序IR的正确图像相对同权重真实图像roll为{delta('same_checkpoint_IR_image_roll')['HOTA']:+.6f}点，"
        '两段视频各自也为正。全部12个缺失条件的47字段、逐序列、完整CSV及MOT均精确回到新鲜B1 RGB。'
        '这些结果支持继续验证当前IR输入的组合收益，不代表额外物理热信息或新颖性证明。', '',
        '| 同轮时序结构−质量结构，HOTA差 | RGB | 灰度 | IR | IR−RGB增量差 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for item in interactions:
        values = item['temporal_minus_quality_HOTA']
        lines.append('| '+str(item['epoch'])+' | '+' | '.join(f'{values[cue]:+.6f}' for cue in ['rgb','gray','ir'])+
                     f" | {item['IR_relative_to_RGB_difference_in_differences']:+.6f} |")
    lines += ['', '第6轮中，时序结构给IR和RGB带来的增量分别约+0.412和+0.413点，二者差约−0.001。'
        '当前不能把时序结构收益说成IR独有。结构之间还同时改变记忆、运动对齐与内部置信递归；'
        '上表只是描述性交互，不能替代独立模块去除或统计因果估计。跨轮交互方向也不稳定。', '',
        '原冻结预算决定已授权六组统一从6延长到9轮：质量灰度后期峰值比前三轮高0.947，质量IR高1.642，时序IR高2.707，均超过原0.2点阈值。'
        '训练沿用单一12轮学习率时间轴，不重置学习率，不增加随机种子；RGB、灰度和IR全部共同延长。'
        '训练曲线与指标均非单调，当前不宣称收敛。后续是否到12轮仍按同一完整Dev规则决定。', '',
        '独立创新计数仍0。当前只有需要继续验证的多模态组合候选；'
        '最终仍须完整预算选择、触发的同预算去除／组合实验、全部正常条件含负条件的7段15985帧Test、'
        '逐方法成本与先验工作边界核对。Test目标34.529及2–3项独立贡献尚未实现。'
        '历史Test暴露、RGB预训练见Dev、伪IR由RGB生成、单seed0等限制保持。', '',
        '![完整六组Dev曲线](perceptual_20261126_native_minimum6_dev_figures/all_six_normal_Dev_curves.svg)', '',
        '![实际IR干预](perceptual_20261126_native_minimum6_dev_figures/all_IR_image_interventions.svg)', '',
        '![逐视频同轮对照](perceptual_20261126_native_minimum6_dev_figures/matched_control_effects_by_Dev_video.svg)', '',
        '[全部62项47字段](PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV.json) · '
        '[原始47字段总表](PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV.csv) · '
        '[完整描述性比较](PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV_ANALYSIS_ZH.json)', '']
    with STEM.with_suffix('.md').open('x') as handle:
        handle.write('\n'.join(lines))
    print(json.dumps(dict(status=report['status'], stage6_temporal_IR_HOTA=metrics(temporal)['HOTA'],
        delta_same_mode_RGB=delta('same_mode_own_best_RGB')['HOTA'],
        delta_strongest_RGB_across_modes=delta('strongest_RGB_across_modes')['HOTA'],
        original_budget_extends_all_six_to=9, supported_innovations=0, goal_met=False)),flush=True)


if __name__=='__main__':
    main()
