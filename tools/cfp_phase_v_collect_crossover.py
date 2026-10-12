#!/usr/bin/env python3
"""Compile complete controlled crossover, output and error evidence."""
import json
from pathlib import Path
import statistics

from cfp.provenance import sha256, write_json


FIELDS = ('HOTA', 'AssA', 'DetA', 'IDF1', 'MOTA', 'IDSW', 'CLR_FP', 'CLR_FN', 'Frag', 'MT', 'ML', 'AssRe', 'AssPr', 'DetRe', 'DetPr', 'LocA')


def table(rows):
    lines = ['| Cell | ' + ' | '.join(FIELDS) + ' |', '|---|' + '|'.join('---:' for _ in FIELDS) + '|']
    for name, mean, std in rows:
        lines.append('| ' + name + ' | ' + ' | '.join(f'{mean[k]:.3f}±{std[k]:.3f}' for k in FIELDS) + ' |')
    return '\n'.join(lines)


def main():
    root = Path('/data3/liuyeqiang/IR-MFT_assets/cfp/phase5')
    campaign_path = root / 'crossover_campaign_v1.json'
    output_path = root / 'output_diagnosis_three_seed_v2/manifest.json'
    campaign, outputs = [json.loads(p.read_text()) for p in (campaign_path, output_path)]
    if campaign['status'] != 'COMPLETE' or outputs['status'] != 'COMPLETE' or len(campaign['runs']) != 36:
        raise ValueError('All36 controlled runs and output diagnoses required')
    report = dict(status='FAILURE_DIAGNOSED', campaign_sha256=sha256(campaign_path), output_diagnosis_sha256=sha256(output_path),
                  cells={}, output_policies=outputs['selected_means'], events={}, scientific_goal='IN_PROGRESS')
    selected, sequence_rows = [], []
    for association, lifecycle, label in [('bot', 'bot', 'A: BoT＋BoT'), ('gru', 'bot', 'B: GRU＋BoT'), ('bot', 'cfp', 'C: BoT＋CFP'), ('gru', 'cfp', 'D: GRU＋CFP')]:
        settings = []
        for candidate in range(3):
            names = [f'cross_{association}_{lifecycle}_s{seed}_c{candidate}_v1' for seed in range(3)]
            values = [campaign['runs'][name]['metrics'] for name in names]
            per_sequence = {seq: dict(mean={key: statistics.mean(v[seq][key] for v in values) for key in FIELDS},
                                      sample_std={key: statistics.stdev(v[seq][key] for v in values) for key in FIELDS})
                            for seq in ('BT-005', 'SN-013', 'COMBINED_SEQ')}
            settings.append(dict(candidate=candidate, runs=names, per_sequence=per_sequence))
        choice = max(settings, key=lambda value: tuple(value['per_sequence']['COMBINED_SEQ']['mean'][key] for key in ('HOTA', 'AssA', 'IDF1')))
        report['cells'][association + '_' + lifecycle] = dict(settings=settings, selected=choice)
        item = choice['per_sequence']['COMBINED_SEQ']
        selected.append((label, item['mean'], item['sample_std']))
        for seq in ('BT-005', 'SN-013'):
            item = choice['per_sequence'][seq]
            sequence_rows.append((label + '/' + seq, item['mean'], item['sample_std']))
    for name in ('events_bot_bot_s0_c0_v1', 'events_gru_cfp_s0_c1_v1', 'events_train_bot_bot_s0_c0_v1', 'events_train_gru_cfp_s0_c1_v1'):
        path = root / name / 'manifest.json'
        events = json.loads(path.read_text())
        if events['status'] != 'COMPLETE':
            raise ValueError('Full error audit not complete')
        report['events'][name] = dict(manifest_sha256=sha256(path), partition=events['partition'],
                                    videos={seq: {key: video[key] for key in ('known_reference_queries', 'unknown_reference_queries', 'categories', 'official_switch_count', 'full_proposal_recall_given_visible')} for seq, video in events['videos'].items()})
    means = {key: value['selected']['per_sequence']['COMBINED_SEQ']['mean'] for key, value in report['cells'].items()}
    effects = {}
    for key in FIELDS:
        a, b, c, d = [means[cell][key] for cell in ('bot_bot', 'gru_bot', 'bot_cfp', 'gru_cfp')]
        effects[key] = dict(gru_minus_bot_given_bot_lifecycle=b-a, gru_minus_bot_given_cfp_lifecycle=d-c,
                           bot_minus_cfp_given_bot_association=a-c, bot_minus_cfp_given_gru_association=b-d,
                           interaction=(b-a)-(d-c))
    report['conditional_effects'] = effects
    write_json('docs/research/PHASE_V_LIFECYCLE_MEASURED.json', report)
    output_rows = [(key, value['COMBINED_SEQ']['mean'], value['COMBINED_SEQ']['sample_std']) for key, value in outputs['selected_means'].items()]
    error_lines = ['| Retrospective category | BoT BT | GRU BT | BoT SN | GRU SN |', '|---|---:|---:|---:|---:|']
    bot = report['events']['events_bot_bot_s0_c0_v1']['videos']
    gru = report['events']['events_gru_cfp_s0_c1_v1']['videos']
    for category in bot['BT-005']['categories']:
        values = [v[seq]['categories'].get(category, {'events': 0})['events'] for seq in ('BT-005', 'SN-013') for v in (bot, gru)]
        error_lines.append('| ' + category + ' | ' + ' | '.join(map(str, values)) + ' |')
    text = f'''# Association / lifecycle crossover results

状态：**FAILURE_DIAGNOSED**。Phase V科学目标仍IN_PROGRESS。36组完整两视频实验（4组合×3设置×3种子）及18组输出干预已完成；没有训练新模型。两个原始对角组合的全部18组运行／36个视频文件均与原输出逐字节一致。100原测试＋5适配测试＋3错误诊断测试通过，新增GT诊断不进入推理代码。

## 显式交换边界与公平预算

关联策略负责成本公式、外观记忆表示和对应接受尺度（BoT .8、GRU .97）；各家族保留既有三个设置。生命周期负责xywh/xyah Kalman、预测、出生确认、Lost/Removed/重激活、重复抑制和输出过滤。低分IoU .5／未确认 .7、置信 .58/.1/.68、37帧lost期限及面积>100相同。BoT外观EMA只写高分观测，GRU历史包含接受的低分观测；这是评分记忆的显式差异，没有暗中统一或称之为纯输出干预。

原BoT更新逻辑有MIT许可注明的独立适配文件及两个评分接口；所有参考依赖哈希验证，原参考仓库不修改。输入共用同一V3及冻结鱼类RGB特征，512维归一化坐标一致，GRU仅读最多8个严格过去的锚点／近期token。对角完整字节回归和独立观测器测试先通过才解释交叉结果。每个家族按三种子平均HOTA选择一个设置，不按最好种子选择。

## 四组合三种子均值及样本标准差

{table(selected)}

每个视频的相同选择配置：

{table(sequence_rows)}

## HOTA与DetA差距来源

BoT生命周期提高DetA：在BoT关联下约2.102个百分点，在GRU关联下约1.820。HOTA却有显著交互：BoT关联换到BoT生命周期提高约2.833，GRU关联换过去反而降低约0.189；GRU相对BoT的关联差，在BoT生命周期下为−0.785，在CFP下为＋2.237。因此不能给出“关联占X%、生命周期占Y%”的加性归因，也不能把简单交换认为身份恢复创新。seed0交叉GRU＋BoT达到53.171，但三种子仅51.696，不能凭单种子领先。

进一步在相同内部匹配／身份编号下只改变输出策略；三个规则先已从原运行保存，seed0线索出现后冻结三种子扩展。没有改变任何内部框预测、匹配、记忆或历史文件。measurement严格输出本帧实际匹配检测框，没有GT或未来信息。此为开发诊断，后续作为更强基线，而非创新归因。

{table(output_rows)}

GRU＋CFP当前检测框输出：HOTA53.591±{outputs['selected_means']['gru_cfp_measurement']['COMBINED_SEQ']['sample_std']['HOTA']:.3f}、AssA39.139、DetA73.457、IDF162.701、MOTA93.672、IDSW87.333；比原GRU增加HOTA约1.706、DetA约2.024。BoT输出过滤效应很小，直接测量框反而略退化。结果指向CFP xyah平滑框输出的定位损失，不能把指标的AssA/IDSW变化当作内部身份已经被修复：身份分配文件键完全相同，评测随IoU阈值和匹配变化。后续RIS/JEPA必须超过这个53.591的更强普通GRU输出基线，并保留原BoT52.482，才能证明独立身份创新。

## 真实决策与后验错误分析

实际在线三阶段成本／几何矩阵、轨迹池状态、候选索引、严格过去的接受观测、分配和unmatched集合在每个运行的压缩日志中。事后才接入独立当前框IoU身份标签及官方CLEAR观察事件。两种标签分开：CLEAR有连续性奖励，不能作为在线输入或独立物理监督。

{chr(10).join(error_lines)}

这里的类别是操作定义、可重叠的帧/轨迹事件；appearance/crossing/duplicate/termination/birth带proxy的项是候选解释，不能当成已证明的独立根因。未知/歧义标签排除，未归为错误。长期污染是真实过去观测混入不同物理标签，但多数历史只是事后参考；first observation没有被当作永久真值。重复框重叠不自动证明真实重复鱼；官方FP共现另存。每类别的事件、受影响帧数和轨迹、候选召回、margin以及原矩阵索引完整保留。类别不互斥，不把HOTA变化强行分配给某类。

BT-005第1049帧附近的全部真实在线矩阵另存，各算法可逐阶段对比，未据此调规则。同样分析已完成六个训练视频的BoT和GRU实际接受历史；训练视频内也存在高置信错误、交叉错误、错误重激活、几何候选排除和污染，后续纠错训练使用这些分布，开发案例只用于报告。训练日志未经SGD改变、无测试集选择。训练官方CLEAR尚未重算时相应计数为null，不能误写成零。

资源：本阶段约1.1GiB新增索引／压缩决策／官方评估文本，未生成图像或特征库。最低40GiB余量，最多四GPU；交叉和事后分析在CPU运行，其耗时绝非端到端FPS。固定GT的SN-013四个歧义框处理不变；检测器训练暴露开发、两开发视频不足跨场景泛化及历史测试曾调参的限制均保留。

复现：tools/cfp_phase_v_crossover_campaign.py（36组）；tools/cfp_phase_v_output_diagnosis.py（18组输出规则）；tools/cfp_analyze_crossover_events.py（固定轨迹事后分析）。完整配置、每视频指标、源快照和哈希：PHASE_V_LIFECYCLE_MEASURED.json及本机phase5各manifest。后续阶段是可恢复身份状态，RIS/JEPA尚未训练，不能声称研究目标完成。
'''
    Path('docs/research/LIFECYCLE_CROSSOVER_RESULTS.md').write_text(text)
    state_path = Path('docs/research/PHASE_V_STATE.json')
    state = json.loads(state_path.read_text())
    state.update(status='FAILURE_DIAGNOSED', stage='PHASE_II_RIS_DATA_AND_TRAINING',
                 strongest_plain_gru_measurement_hota=outputs['selected_means']['gru_cfp_measurement']['COMBINED_SEQ']['mean']['HOTA'])
    state['completed'] += ['36 complete three-seed lifecycle crossover runs', '18 fixed-output policy evaluations', 'Six training-video real accepted-history traces for BoT/GRU', 'Physical event taxonomy and exact dev CLEAR co-observation']
    state['remaining'] = [item for item in state['remaining'] if item != 'Lifecycle crossover and event attribution']
    state['completed'] = list(dict.fromkeys(state['completed']))
    write_json(state_path, state)
    print('Complete lifecycle evidence compiled; RIS/JEPA remain active research.')


if __name__ == '__main__':
    main()
