#!/usr/bin/env python3
"""Describe a completed native Dev stage; preserve original choices and metrics."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW']
LENGTHS = {'BT-005': 1834, 'SN-013': 3000, 'COMBINED': 4834}
MODES = ('quality', 'temporal')
CUES = ('rgb', 'gray', 'ir')
CANONICAL = 'NP_source_canonical_RGB'
FRESH = 'NP_source_fresh_RGB'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank(row):
    values, config = row['original_official_metrics39'], row['config']
    return (-values['HOTA'], -values['AssA'], -values['IDF1'],
            config['epoch'], 0 if config['mode'] == 'quality' else 1)


def expected_ids(stage):
    result = {CANONICAL, FRESH}
    for mode in MODES:
        for cue in CUES:
            for epoch in range(1, stage + 1):
                key = f'NP_{mode}_{cue}_e{epoch:02d}'
                result.add(key)
                if cue == 'ir':
                    result.update((key + '_off', key + '_roll'))
    return result


def completed_report(path, digest, stage):
    require(sha(path) == digest, 'Completed report SHA differs')
    data = json.loads(path.read_text())
    status = ('COMPLETE_INDEPENDENT_MINIMUM6_FULL_DEV_AUDIT_NOT_FINAL_STUDY'
              if stage == 6 else 'COMPLETE_INDEPENDENT_EXTENDED_FULL_DEV_AUDIT_NOT_FINAL_STUDY')
    require(data['stage'] == stage and data['status'] == status and
            data['split'] == 'dev' and data['frames'] == 4834,
            'Only an actually completed matching full Dev stage is accepted')
    require(data['Test_results_read'] is False and data['fixed_seed'] == 0 and
            data['additional_seed_work'] is False and
            data['unavailable_all47_per_sequence_full_CSV_full_MOT_exact_to_fresh_RGB'],
            'Original fixed-seed Dev and complete unavailable-modality audit required')
    lookup = {row['id']: row for row in data['rows']}
    require(len(lookup) == len(data['rows']) == 10 * stage + 2 and
            set(lookup) == expected_ids(stage), 'Missing, extra or duplicate full-stage case')
    phases = data['original_stage_worker_completion']
    require(isinstance(phases, dict) and phases, 'Original worker receipts missing')
    for jobs in phases.values():
        require(isinstance(jobs, dict) and jobs, 'Original phase is empty')
        require(all(job['status'] == 'COMPLETE' and type(job.get('exit_code')) is int and
                    job['exit_code'] == 0 for job in jobs.values()),
                'An original stage worker lacks actual numeric exit0')
    if stage > 6:
        require(data['all_registered_cases_included'] and
                data['every_stage_recorded_original_image_byte_digest_matches_reference'],
                'Complete extended cases and original-image digest checks required')
    schema = data['official_fields']
    require(len(schema) == len(set(schema)) == 47 and set(FIELDS) <= set(schema),
            'Exact official47 schema required')
    bindings = []
    for row in lookup.values():
        require(row['split'] == 'dev' and set(row['combined47']) == set(schema),
                'A row is not complete official47 Dev')
        require(all(math.isfinite(row['combined47'][field]) for field in schema),
                'Nonfinite official metric')
        points = row['per_sequence47']
        require(len(points) == 3 and {p['seq'] for p in points} == set(LENGTHS),
                'Both complete Dev videos and official combined row required')
        for point in points:
            require(point['split'] == 'dev' and point['CLR_Frames'] == LENGTHS[point['seq']] and
                    set(schema) <= set(point) and
                    all(math.isfinite(point[field]) for field in schema),
                    'Incomplete per-sequence official47 metrics')
        binding = row['official_CSV']
        require(sha(ROOT / binding['file']) == binding['sha256'], 'Original official CSV changed')
        bindings.append(binding)
    own = data['six_own_rounded_official_Dev_best']
    require(set(own) == {mode + '_' + cue for mode in MODES for cue in CUES},
            'Six original own-Dev choices required')
    for mode in MODES:
        for cue in CUES:
            candidates = [lookup[f'NP_{mode}_{cue}_e{epoch:02d}']
                          for epoch in range(1, stage + 1)]
            for row in candidates:
                config = row['config']
                require(config['mode'] == mode and config['cue'] == cue and
                        config['intervention'] == 'normal' and config['seed'] == 0 and
                        config['removal'] is None,
                        'A normal curve contains another treatment')
            require(own[mode + '_' + cue] == min(candidates, key=rank)['id'],
                    'Original rounded-official own-Dev choice differs')
    fresh = lookup[FRESH]
    for mode in MODES:
        for epoch in range(1, stage + 1):
            off = lookup[f'NP_{mode}_ir_e{epoch:02d}_off']
            require(off['combined47'] == fresh['combined47'] and
                    off['per_sequence47'] == fresh['per_sequence47'],
                    'Unavailable modality does not reproduce fresh RGB metrics')
    return data, lookup, own, schema, bindings


def describe(data, lookup, own, schema):
    stage = data['stage']

    def metrics(key):
        return {field: lookup[key]['combined47'][field] for field in FIELDS}

    def point(key, sequence):
        return next(p for p in lookup[key]['per_sequence47'] if p['seq'] == sequence)

    strongest = {cue: min((own[mode + '_' + cue] for mode in MODES),
                          key=lambda key: rank(lookup[key])) for cue in CUES}
    all_six = [dict(arm=arm, case=key, epoch=lookup[key]['config']['epoch'], metrics=metrics(key))
               for arm, key in own.items()]
    comparisons, per_sequence, curves, interactions = [], [], [], []
    for mode in MODES:
        ir = own[mode + '_ir']
        controls = dict(canonical_RGB=CANONICAL, fresh_B1_RGB=FRESH,
                        same_mode_own_best_RGB=own[mode + '_rgb'],
                        same_mode_own_best_gray=own[mode + '_gray'],
                        strongest_RGB_across_modes=strongest['rgb'],
                        strongest_gray_across_modes=strongest['gray'],
                        same_checkpoint_IR_image_roll=ir + '_roll',
                        same_checkpoint_IR_unavailable=ir + '_off')
        for label, control in controls.items():
            left, right = lookup[ir]['combined47'], lookup[control]['combined47']
            comparisons.append(dict(mode=mode, IR_case=ir, control_label=label, control_case=control,
                delta={field: left[field] - right[field] for field in FIELDS},
                delta47={field: left[field] - right[field] for field in schema}))
            for sequence in LENGTHS:
                left, right = point(ir, sequence), point(control, sequence)
                per_sequence.append(dict(mode=mode, sequence=sequence, IR_case=ir,
                    control_label=label, control_case=control,
                    delta={field: left[field] - right[field] for field in FIELDS},
                    delta47={field: left[field] - right[field] for field in schema}))
        for cue in CUES:
            curves.append(dict(mode=mode, cue=cue, points=[dict(epoch=epoch,
                case=f'NP_{mode}_{cue}_e{epoch:02d}',
                metrics=metrics(f'NP_{mode}_{cue}_e{epoch:02d}'))
                for epoch in range(1, stage + 1)]))
    for epoch in range(1, stage + 1):
        changes = {cue: metrics(f'NP_temporal_{cue}_e{epoch:02d}')['HOTA'] -
                        metrics(f'NP_quality_{cue}_e{epoch:02d}')['HOTA'] for cue in CUES}
        interactions.append(dict(epoch=epoch, temporal_minus_quality_HOTA=changes,
            IR_relative_to_RGB_difference_in_differences=changes['ir'] - changes['rgb'],
            IR_relative_to_gray_difference_in_differences=changes['ir'] - changes['gray'],
            scope='Descriptive same-epoch architecture/cue interaction, not an isolated module ablation or causal estimate.'))
    return dict(all_six_own_registered_Dev_best=all_six,
                descriptive_strongest_controls_across_modes=strongest,
                IR_own_best_comparisons=comparisons, per_sequence_comparisons=per_sequence,
                full_normal_curves=curves, same_epoch_architecture_cue_interactions=interactions)


def markdown(report, lookup):
    stage = report['stage']
    lines = [f'# 完整{stage}轮Dev：多模态对照与证据边界', '',
        f"两个完整Dev视频、4834帧，全部{report['cases']}项条件。"
        '以下只描述已完成官方报告，沿用原六组各自Dev最佳；不改变预算、最终选择或实验授权。', '',
        '| 条件 | 原Dev最佳检查点 | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    entries = [('规范直接RGB', CANONICAL), ('新鲜B1 RGB', FRESH)]
    entries += [(row['arm'], row['case']) for row in report['all_six_own_registered_Dev_best']]
    for label, key in entries:
        values = lookup[key]['combined47']
        rendered = [str(values[field]) if field == 'IDSW' else f'{values[field]:.6f}'
                    for field in FIELDS]
        lines.append('| ' + label + ' | ' + key + ' | ' + ' | '.join(rendered) + ' |')
    strongest = report['descriptive_strongest_controls_across_modes']
    lines += ['', f"跨结构RGB强对照为{strongest['rgb']}，灰度强对照为{strongest['gray']}。"
        '这些是按原始rounded-official规则对既有各组Dev最佳做的描述性比较，未建立新的全局选模策略。', '',
        '| IR−对照：模式 | 视频 | 同结构RGB | 同结构灰度 | 跨结构最强RGB | 同检查点错位IR | 同检查点缺失IR |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    labels = ['same_mode_own_best_RGB', 'same_mode_own_best_gray', 'strongest_RGB_across_modes',
              'same_checkpoint_IR_image_roll', 'same_checkpoint_IR_unavailable']
    for mode in MODES:
        for sequence in LENGTHS:
            values = [next(row['delta']['HOTA'] for row in report['per_sequence_comparisons']
                           if row['mode'] == mode and row['sequence'] == sequence and
                           row['control_label'] == label) for label in labels]
            lines.append('| ' + mode + ' | ' + sequence + ' | ' +
                         ' | '.join(f'{value:+.6f}' for value in values) + ' |')
    lines += ['', '差值为IR减对照的原始点差。正负方向均完整保留；两段视频、各轮检查点和表中比较不是独立随机重复。'
        '汇总收益不代表每个场景稳定改善，不能按序列身份设计Test路由。IDSW等指标的改善方向与HOTA不同，'
        '完整JSON同时保留全部47字段的有符号差值，不统一标成“越大越好”。', '',
        f'全部{2 * stage}个缺失模态条件由原独立审计核对全部47字段、逐视频、完整CSV与MOT精确回到新鲜B1 RGB；'
        '本工具重新核对绑定的CSV哈希及报告数值，不重新运行模型、官方评估或MOT比较。'
        '正确IR与错位IR比较支持的是当前输入干预分析，不替代模块去除或独立贡献证明。', '',
        '| 同轮时序−质量结构，HOTA点差 | RGB | 灰度 | IR | IR−RGB增量差 | IR−灰度增量差 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for item in report['same_epoch_architecture_cue_interactions']:
        changes = item['temporal_minus_quality_HOTA']
        values = [changes[cue] for cue in CUES] + [
            item['IR_relative_to_RGB_difference_in_differences'],
            item['IR_relative_to_gray_difference_in_differences']]
        lines.append('| ' + str(item['epoch']) + ' | ' +
                     ' | '.join(f'{value:+.6f}' for value in values) + ' |')
    lines += ['', '结构间同时改变记忆、运动对齐及内部置信递归。上表是同轮架构与输入的描述性交互，'
        '不能单独归因为记忆模块，也不是统计因果估计。完整正常曲线与全部正负条件保留在源报告和分析JSON。', '',
        '原冻结共同预算决定保持不变：', '', '```json',
        json.dumps(report['original_frozen_budget_decision'], ensure_ascii=False, indent=2), '```', '',
        '当前分析不声明收敛或独立创新。仍须按原计划完成最终预算、被触发的同预算模块去除、'
        '全部七段15985帧正式Test和逐方法成本，验证至少+5点HOTA与2–3项独立贡献。'
        '上述数字均为Dev，不能与Test基线29.529跨划分相减。历史Test已暴露、RGB预训练见Dev、'
        '辅助IR由RGB生成、单seed0等限制保持。', '',
        f"[完整47字段源报告]({Path(report['source_report']).name})。"
        f"源报告SHA256：`{report['source_report_sha256']}`。分析JSON保留全部CSV绑定与分析源代码SHA。", '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev-report', type=Path, required=True)
    parser.add_argument('--expect-report-sha256', required=True)
    parser.add_argument('--stage', type=int, choices=(6, 9, 12), required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only analysis required')
    require(not args.output_directory.exists(), 'Preserve an existing analysis directory')
    source = args.dev_report.resolve()
    require(source.is_relative_to(ROOT), 'Use the original repository Dev audit report')
    started = time.monotonic()
    data, lookup, own, schema, bindings = completed_report(
        source, args.expect_report_sha256, args.stage)
    audit_source = ROOT / data['source_audit_tool']
    require(sha(audit_source) == data['source_audit_tool_sha256'], 'Original audit source changed')
    result = describe(data, lookup, own, schema)
    report = dict(status='COMPLETE_NATIVE_STAGE_DEV_DESCRIPTIVE_ANALYSIS_NOT_TEST_OR_INNOVATION_PROOF',
        stage=args.stage, cases=len(lookup), split='dev', frames=4834, videos=['BT-005', 'SN-013'],
        source_report=str(source.relative_to(ROOT)), source_report_sha256=args.expect_report_sha256,
        source_CSV_bindings=bindings, source_analysis_tool=str(Path(__file__).resolve().relative_to(ROOT)),
        source_analysis_tool_sha256=sha(__file__), original_audit_source=str(audit_source.relative_to(ROOT)),
        original_audit_source_sha256=data['source_audit_tool_sha256'], official_fields=schema,
        **result, original_frozen_budget_decision=data['original_frozen_budget_decision'],
        source_stage_decision_sha256=data['original_stage_decision_sha256'],
        source_original_completed_workers=sum(len(jobs) for jobs in data['original_stage_worker_completion'].values()),
        source_47_metrics_or_official_evaluation_recomputed=False,
        original_frozen_choices_or_budget_changed=False, new_Test_case_selected=False,
        strongest_RGB_comparison_must_not_be_omitted=True,
        isolated_temporal_module_contribution_established=False, convergence_established=False,
        Test_results_read=False, fixed_seed=0, additional_seed_work=False, GPU_work=False,
        NN_forward_passes=0, backward_passes=0, optimizer_updates=0,
        source_images_weights_MOT_or_feature_banks_read=False,
        supported_innovations=0, goal_met=False,
        limitations=['Only two full Dev videos; historical Test was previously exposed.',
            'RGB pretraining saw Dev. PseudoIR is derived from RGB rather than an independent sensor.',
            'Signed point differences and architecture/cue interactions do not establish independent contributions.',
            'CSV hashes and prior audited metrics are checked; official metrics and MOT are not independently recomputed.'])
    text = markdown(report, lookup)
    require(sha(source) == args.expect_report_sha256 and
            sha(audit_source) == data['source_audit_tool_sha256'] and
            all(sha(ROOT / row['file']) == row['sha256'] for row in bindings),
            'Input evidence changed during analysis')
    report.update(seconds=time.monotonic() - started,
                  peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    args.output_directory.mkdir(exist_ok=False, parents=True)
    (args.output_directory / 'complete.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (args.output_directory / 'analysis.md').write_text(text)
    print(json.dumps(dict(status=report['status'], actual_stage=args.stage, actual_cases=len(lookup),
        original_choices_preserved=True, source_CSVs_checked=len(bindings),
        seconds=report['seconds'], GPU_work=False, Test_results_read=False,
        supported_innovations=0, goal_met=False)), flush=True)


if __name__ == '__main__':
    main()
