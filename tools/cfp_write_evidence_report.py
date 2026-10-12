#!/usr/bin/env python3
"""Compile completed measured artifacts into the final bounded research report."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from cfp.provenance import sha256, write_json


def table(rows, fields):
    header = '| 方法 | ' + ' | '.join(fields) + ' |\n'
    header += '|---|' + '|'.join('---:' for _ in fields) + '|\n'
    for label, mean, std in rows:
        values = []
        for field in fields:
            value = f'{mean[field]:.3f}'
            if std is not None:
                value += f'±{std[field]:.3f}'
            values.append(value)
        header += '| ' + label + ' | ' + ' | '.join(values) + ' |\n'
    return header


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets-root', type=Path, required=True)
    p.add_argument('--online-report', type=Path, required=True)
    args = p.parse_args()
    root = args.assets_root
    online = json.loads(args.online_report.read_text())
    if online['status'] != 'COMPLETE' or online['three_seed_current_frame']['runtime_variant'] != 'split_cu117_b8':
        raise ValueError('Do not finalize before corrected full-video collection')
    parity = {}
    for seed in range(3):
        name = f'online_anchor_gru_seed{seed}_split_full_v1'
        parity[name] = {}
        for seq in ('BT-005', 'SN-013'):
            current = root / 'phase4' / name / (seq + '.txt')
            reference = root / 'phase3' / f'plim_v3_dev_gru_predictive_anchor_seed{seed}_w0.3' / (seq + '.txt')
            parity[name][seq] = current.read_bytes() == reference.read_bytes()
            if not parity[name][seq]:
                raise ValueError('Corrected complete online output differs from its canonical reference')
    selections = {}
    sources = {}
    for key, filename in [('baseline', 'baseline_selection_post_neck_v1.json'), ('first', 'plim_first_three_seed_selection_v1.json'),
                          ('anchor', 'plim_anchor_three_seed_selection_v3.json'), ('log', 'plim_log_time_three_seed_selection_v4.json'),
                          ('ir', 'plim_modality_ir_three_seed_selection_v1.json'), ('rgb_ir', 'plim_modality_rgb_ir_three_seed_selection_v1.json'),
                          ('two_rgb', 'plim_modality_two_rgb_three_seed_selection_v1.json')]:
        path = Path('configs/cfp') / filename
        record = json.loads(path.read_text())
        if record.get('final_test_frozen'):
            raise ValueError('This bounded negative report must not invent a final test freeze')
        selections[key] = record['families']
        sources[str(path)] = sha256(path)
    rows = []
    def selected(label, group, family):
        entry = selections[group][family]['selected']
        rows.append((label, entry['mean'], entry['sample_std']))
    selected('V3＋BoT-SORT＋RGB ReID', 'baseline', 'bot_sort')
    selected('原锚点普通 RGB GRU', 'anchor', 'plim_v3_dev_gru_association_anchor')
    selected('原锚点预测 RGB GRU', 'anchor', 'plim_v3_dev_gru_predictive_anchor')
    selected('原锚点预测 RGB Transformer', 'anchor', 'plim_v3_dev_transformer_predictive_anchor')
    selected('对数时间预测 RGB GRU', 'log', 'plim_v4_dev_gru_predictive_log_anchor')
    for view, label in [('ir', 'IR'), ('rgb_ir', 'RGB＋IR'), ('two_rgb', '双 RGB')]:
        selected(label + '预测 GRU', view, f'plim_mod_dev_{view}_gru_predictive_anchor')
        selected(label + '普通 GRU', view, f'plim_mod_dev_{view}_gru_association_anchor')
    joint = json.loads(Path('docs/research/phase4_joint_measured_seed0_20261012.json').read_text())
    joint_rows = [(name.removeprefix('joint_v1_dev_').removesuffix('_seed0'), values, None) for name, values in joint['rows'].items()]
    intervention_rows = []
    cost_rows = []
    for name, entry in online['runs'].items():
        if name.startswith('online_ir_intervention_'):
            intervention_rows.append((name.removeprefix('online_ir_intervention_').removesuffix('_seed0_v1'), entry['metrics']['COMBINED_SEQ'], None))
        else:
            child_peak = json.loads((root / 'phase4' / name / 'manifest.json').read_text())['detector_runtime']['peak_gpu_memory_bytes']
            cost_rows.append(f"| {name} | {entry['aggregate_fps']:.3f} | {entry['aggregate_latency_p50_p95_ms'][0]:.1f} | {entry['aggregate_latency_p50_p95_ms'][1]:.1f} | {entry['peak_gpu_memory_bytes']/2**30:.3f} | {child_peak/2**30:.3f} |")
    free = shutil.disk_usage(root).free / 2**30
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    state = dict(status='GOAL_NOT_MET_WITH_EVIDENCE', task_scope='Bounded original research hypotheses evaluated; no validated final innovative method or SOTA claim.',
                 original_goal='Reliable synthetic-modality identity association for online multi-object fish tracking',
                 reasons=['CMUE has no validated independent gain; final gate accepts zero IR rows',
                          'Paired forecasting helps its ordinary paired control but is not stably superior to strong RGB/twoRGB capacity controls',
                          'No candidate improves strongest native baseline HOTA across the complete selected three-seed development protocol',
                          'Corrected end-to-end runtime is causal and reproducible but does not meet camera-rate real-time performance'],
                 selection_sources=sources, corrected_online_report_sha256=sha256(args.online_report),
                 full_trajectory_parity=parity, final_test_frozen=False, new_historical_test_evaluation=False,
                 limitations=['Two development videos/21 identities; dense49-fish MSK scene excluded',
                              'Historical detector trained on all8 original training videos including dev',
                              'Biological identity overlap across videos is unverified',
                              'Native person-weight/own-detector ProtocolB and external fresh blind validation unperformed',
                              'Group interaction and joint SGD not introduced without controlled core usefulness evidence'],
                 free_gib=free, minimum_free_gib=30, source_commit=commit)
    write_json('docs/research/RESEARCH_OUTCOME_20261012.json', state)
    report = f'''# CFP-FishMOT bounded research evidence

状态：**GOAL_NOT_MET_WITH_EVIDENCE**，2026-10-12。在线工程与复现验证完成；当前证据不支持原计划要求的独立 CMUE／合成模态贡献和最强基线上的稳定总体提升。本结论限于已记录的模型、数据与受控预算，不代表所有合成模态方法均无效。

## 研究目的与实验边界

研究合成伪红外何时帮助鱼类身份关联、潜在身份预测能否恢复遮挡身份，以及记忆写入能否避免污染。IR 来自 RGB 的规则响应和伪彩映射，没有真实温度或独立传感器信息。完整视频隔离：六个身份训练视频、BT-005／SN-013 两个开发视频，开发集4,834帧、21个身份。V3 曾在全部八个原训练视频上训练；此划分只隔离新增身份模块，不是整个检测系统的独立盲测。原始标注保留，SN-013 四个歧义框按冻结协议在派生 GT 中忽略。

历史七序列15,985帧复现逐检测数组、逐轨迹文件精确一致：HOTA31.091、AssA19.619、DetA49.575、IDF135.044、MOTA52.325、IDSW560。历史测试被调过参数，只作参考，不能与下面开发集数字直接比较。15个参考项目做了固定 commit 的源码审计，并非全部训练复现。

## 同检测输入的三种子完整比较（Protocol A）

每个家族固定三个参数候选，以三个种子的 combined-dev 平均 HOTA 选择配置；± 为样本标准差，不是独立视频置信区间。所有使用相同 V3 因果检测输入。原生外部关联器可因坐标输出与生命周期不同改变 DetA，不能把所有 HOTA 差异解释为纯身份贡献。

{table(rows, ['HOTA','AssA','DetA','IDF1','MOTA','IDSW'])}
ByteTrack 同一开发协议：HOTA49.297、AssA34.012、IDF158.877、MOTA92.529、IDSW86。BoT 的 HOTA 最强，但 IDF1 较低、切换较多；普通 RGB GRU／配对预测有身份指标与检测指标的权衡，没有全面领先。

第一轮十二个 RGB 时序模型／90组跟踪结果完成；实际污染历史第二轮四模型／12组完成；第三轮可观测初始锚点十二模型／99组完成；年龄饱和驱动的单次对数时间修复十二模型／36组完成；计划内 IR-only／配对 IR／双 RGB 十八模型／54组完成。原锚点 Transformer 的 seed0 HOTA53.055 没有稳定复现；三种子均值50.606。对数时间修复未带来提升，不保留。配对预测较普通配对关联有收益，但 IR 配对仅比同容量双 RGB 高约0.212个百分点，且与更便宜的普通单 RGB 锚点 GRU几乎相同，不能据此证明 IR 特有优势。双 RGB 控制循环复用已有三个冻结编码器，跨控制管线的编码器重叠已披露。

## CMUE、联合关联和独立记忆写入

三轮 seed0 CMUE 修复及匹配控制完成：第一轮 HOTA50.977，去掉身份排序的常数 IR 距离控制50.953，差0.024且切换更少；候选上下文遗漏修复后和排序／符号监督第三轮仍没有独立收益。第三轮校准符号 AUC0.405，实际部署接受 IR 为零。GT 仅作训练／事后分析，在线评分不读标签。

九组冻结组合和记忆开关（seed0，未做联合 SGD）：

{table(joint_rows, ['HOTA','AssA','IDF1','MOTA','IDSW'])}
拒绝 IR／IR 不可用时，联合评分和原 PLIM 逐值一致；PLIM-only 的完整文件与原实验逐字节一致。普通／质量／固定 IR 融合退化；独立拒写也退化。拒写后出现最长127帧记忆间隔及具体交叉失败，但只有少量超过原训练间隔，不能把它当作唯一根因。不能把“可靠写入”作为已验证改进。

## 真实像素干预：固定检测框和 RGB 输入

现场编码当前裁剪，检测框保持规范缓存；此部分速度排除预处理生成和检测，不能称为端到端 FPS。

{table(intervention_rows, ['HOTA','AssA','IDF1','MOTA','IDSW'])}
正确 IR 好于置零、重复 RGB、打乱空间结构的 IR，说明在该固定融合模型内部，未破坏的辅助表征有价值；但它仍低于不加 IR 的强 PLIM 基线。CMUE 对正确／置零 IR 都完全拒绝，轨迹相同。不能把坏输入的退化当成相对强 RGB 基线的新增模态优势。

## 完整当前帧流程与运行栈修复（Protocol B）

最初共同使用 cu118/cuDNN8.7／检测批次1 的三个完整运行均值 HOTA48.901，作为保留的数值差异诊断。规范检测器使用 cu117/cuDNN8.5，身份编码器训练／导出使用 cu118；最初入口遗漏了这一差别，随后通过受控当前帧对照修复。cu117批次8在两个视频各128帧上的原始框／分数／NMS零差异，批次1不能复现规范输入。

现在检测器在独立 cu117 worker 中运行，身份编码器和预测器在 cu118 中运行；双方只传当前 RGB／响应数组，检测填充是当前帧的八份副本。生成器只用当前和过去帧，在响应生成后更新 EMA，所有序列采用同一配置。两套运行栈间传递无像素／提案精度损失，没有未来图像或 GT。三个种子的完整两视频轨迹均与原规范缓存关联逐字节一致，全部生成像素也与逐帧审计的因果数据精确一致。

| 运行 | 全流程 FPS | p50 ms | p95 ms | 父进程峰值分配 GiB | 检测进程峰值分配 GiB |
|---|---:|---:|---:|---:|---:|
{chr(10).join(cost_rows)}

总延迟包括解码、因果响应／JPEG、两种环境间传递、检测／NMS、裁剪、编码、预测／关联及输出。worker 子阶段与 roundtrip 重叠，不能重复相加。两个进程的独立峰值不能当作同时整卡显存，也不含全部驱动上下文。GPU0／1／2 各运行一个修正管线，主机仍与其他工作共享。这是真实完整视频测量，不能称为25FPS实时系统。参数：V3检测器186,199,381；单编码器24,607,808；RGB GRU231,040；配对GRU362,624；效用估计器5,697。算子 profiler 只计支持的卷积／矩阵乘等 FLOPs，检测单份1.400e12、编码单裁剪5.340e9；编码补齐128、检测重复8的成本不可省略，CPU生成／解码／Hungarian等未纳入该 FLOP计数。

## 失败诊断与复现

主要证据：pre-neck 窄锥特征的27组结果能由相同常数身份向量逐文件复现，早期增益来自几何接受规则；post-neck修复恢复表征区分，但不建立独立 IR 收益。第一轮效用监督遗漏148个正确几何候选，已单独修复并保留旧结果。实际历史存在较重身份污染；原始锚点有77%–99%的年龄编码饱和，针对性对数编码并未改善结果。单个有利种子不能替代完整均值。真实 CLEAR 交叉轨迹和错误写入图例保留在 FAILURE_ANALYSIS.md 的源记录中。

100项实现测试通过。ReID 恢复验证已有记录；本轮真实中断 predictor／utility 于 epoch3，再恢复到epoch8，权重、优化器状态、每轮损失精确一致，utility 校准也一致。此结论针对记录的已完成 epoch 和源码／运行栈，不保证任意半写入文件或未来版本。每次运行都保留小型源码快照、配置／输入／权重哈希、命令、每视频官方指标与失败日志，大型数据不进 Git。

## 对原任务书研究问题的回答

| 问题 | 当前可支持的回答 |
|---|---|
| V3和真实HOTA/AssA/DetA是否恢复 | 已精确恢复，见历史参考段；开发指标读官方输出，未用HOTA反推AssA |
| IR何时有效、何时有害 | 正确结构优于破坏结构；但在强RGB预测上固定混合通常退化，尚无稳定的条件收益选择器 |
| CMUE是否优于普通质量门控、反事实监督是否独立有效 | 当前三轮和联合控制均不支持，第三轮全部回退RGB |
| PLIM是否优于EMA/普通GRU/运动 | 部分预测模型高于简单记忆；配对预测高于同容量普通配对GRU，但普通单RGB与双RGB控制使IR特有贡献不清楚 |
| 是否减少相似鱼交叉IDSW | 一些配置总切换下降，原锚点RGB预测均值83.667对ByteTrack86；但不是所有视频/种子一致，身份盲几何控制也解释部分事件 |
| 哪个创新点提高HOTA/AssA最多 | 未证实独立创新优势，不能将原生关联器、表示导出或候选规则收益归给CMUE/预测损失 |
| MOTA和IDF1/HOTA是否权衡 | 完整表报告；普通RGB/配对预测IDF1较高而HOTA低于BoT，原生输出坐标/生命周期影响DetA |
| 稳定性和每序列是否一致 | 单种子53.055不稳定；所有三设置/三种子/每视频结果保留，开发视频数少，不能宣称普适 |
| 速度、显存和计算成本是否实测 | 已报告当前帧完整视频总延迟及父/检测进程分配峰值，非缓存重放FPS；有限算子FLOPs明确覆盖范围 |
| 与强基线是否公平 | ProtocolA共用提案和鱼类特征/开发预算；披露原生规则与坐标差别，未声称复现其人类权重/原检测完整系统 |
| 失败是否定位、是否完全在线 | 有具体负结果/源快照/图例；修正管线只使用当前/过去，完整像素/轨迹一致性通过 |
| 创新是否充分、能否支撑独立方法论文 | 目前不足；可复现负结果和平台证据成立，原计划的稳定CMUE+IR独立贡献未达到 |
'''
    report += f'''
## 结论、边界与资源

在预先固定的两视频开发协议和受控迭代内，原科学成功条件未满足；没有宣布 SOTA 或完整创新方法成功。保留所有负结果、有效控制和可复现在线入口。密集场景补充交叉验证、真正新的外部盲测、原生完整系统 ProtocolB、联合 SGD 和可选群体交互没有执行，不能标为已完成；没有核心独立效用证据时，未强行添加交互模块或继续扩大参数搜索。历史七序列未用于新模型选择，也没有新方法最终盲测声明。

当前剩余约{free:.1f}GiB，最低保留30GiB。本轮不新增图像/裁剪/特征银行，仅保存小模型、轨迹、指标与日志；新增受控产物约1.1GiB，原数据、权重、检测缓存、历史记录保留。最多使用三张物理GPU，符合四卡上限。代码在research/cfp-fishmot-v1；源码提交{commit[:7]}，本轮 GitHub 同步遭TLS/EOF/SSH超时，最后确认的远端为6ffaaba，不能声称新提交已推送。

主要入口：tools/cfp_run_online.py；正确配置configs/cfp/online_anchor_gru_seed{{0,1,2}}_split_v1.json。训练/重放/评估命令和哈希在experiment_registry及对应不可变manifest中。摘要状态：RESEARCH_OUTCOME_20261012.json；完整机器报告：{args.online_report}。
'''
    Path('docs/research/FINAL_RESEARCH_REPORT.md').write_text(report)
    write_json(root / 'current_research_state_20261012.json', state)
    print('Compiled bounded outcome:', state['status'], 'full trajectory parity', parity)


if __name__ == '__main__':
    main()
