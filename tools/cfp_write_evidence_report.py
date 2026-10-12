#!/usr/bin/env python3
"""Compile completed measured artifacts into the final bounded research report."""
import argparse
import json
from pathlib import Path
import shutil
import statistics
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
    p.add_argument('--confirmed-remote-head')
    p.add_argument('--push-log', type=Path)
    args = p.parse_args()
    root = args.assets_root
    online = json.loads(args.online_report.read_text())
    if online['status'] != 'COMPLETE' or online['three_seed_current_frame']['runtime_variant'] != 'split_cu117_b8':
        raise ValueError('Do not finalize before corrected full-video collection')
    diagnostic_path = root / 'phase4/terminal_detector_batch_diagnostic_v1/manifest.json'
    diagnostic = json.loads(diagnostic_path.read_text())
    if diagnostic['status'] != 'COMPLETE' or diagnostic['terminal_batch_size'] != 2:
        raise ValueError('Completed terminal batch diagnosis required')
    for frame in (2997, 2998, 2999, 3000):
        rows = diagnostic['frames'][str(frame)]
        if rows['8']['raw_equal'] != (frame < 2999) or rows['2']['raw_equal'] != (frame >= 2999):
            raise ValueError('Unexpected terminal detector parity')
    parity = {}
    for seed in range(3):
        name = f'online_anchor_gru_seed{seed}_split_full_v1'
        parity[name] = {}
        for seq in ('BT-005', 'SN-013'):
            current = root / 'phase4' / name / (seq + '.txt')
            reference = root / 'phase3' / f'plim_v3_dev_gru_predictive_anchor_seed{seed}_w0.3' / (seq + '.txt')
            actual, expected = current.read_text().splitlines(), reference.read_text().splitlines()
            if len(actual) != len(expected) or [x.split(',')[:2] for x in actual] != [x.split(',')[:2] for x in expected]:
                raise ValueError('Online frame/identity decisions differ from canonical replay')
            differences = [dict(row=i + 1, online=a, canonical=b) for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
            if any(seq != 'SN-013' or int(x['online'].split(',')[0]) not in (2999, 3000) for x in differences):
                raise ValueError('Unexplained difference outside the diagnosed terminal batch')
            parity[name][seq] = dict(byte_equal=current.read_bytes() == reference.read_bytes(),
                                    frame_identity_equal=True, differing_rows=differences,
                                    online_sha256=sha256(current), canonical_sha256=sha256(reference))
        measured = json.loads((root / 'phase4' / (name + '_evaluation') / 'metrics.json').read_text())['metrics']
        original = json.loads((root / 'phase3' / f'plim_v3_dev_gru_predictive_anchor_seed{seed}_w0.3_evaluation/metrics.json').read_text())['metrics']
        for seq in measured:
            for field in ('HOTA', 'AssA', 'DetA', 'IDF1', 'MOTA', 'IDSW'):
                if measured[seq][field] != original[seq][field]:
                    raise ValueError('Principal official metrics differ')
        parity[name]['metric_differences'] = {seq: {key: dict(online=value, canonical=original[seq][key])
                                                   for key, value in values.items() if value != original[seq][key]}
                                             for seq, values in measured.items()}
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
    selected('V3＋HybridSORT＋RGB ReID', 'baseline', 'hybrid_sort')
    selected('RGB EMA', 'baseline', 'rgb')
    selected('IR EMA', 'baseline', 'ir')
    selected('固定 RGB／IR EMA', 'baseline', 'fixed')
    selected('原锚点普通 RGB GRU', 'anchor', 'plim_v3_dev_gru_association_anchor')
    selected('原锚点预测 RGB GRU', 'anchor', 'plim_v3_dev_gru_predictive_anchor')
    selected('原锚点预测 RGB Transformer', 'anchor', 'plim_v3_dev_transformer_predictive_anchor')
    selected('对数时间预测 RGB GRU', 'log', 'plim_v4_dev_gru_predictive_log_anchor')
    for view, label in [('ir', 'IR'), ('rgb_ir', 'RGB＋IR'), ('two_rgb', '双 RGB')]:
        selected(label + '预测 GRU', view, f'plim_mod_dev_{view}_gru_predictive_anchor')
        selected(label + '普通 GRU', view, f'plim_mod_dev_{view}_gru_association_anchor')
    sequence_rows = []
    for label, group, family, phase in [('BoT-SORT', 'baseline', 'bot_sort', 'phase1'),
                                       ('RGB预测GRU', 'anchor', 'plim_v3_dev_gru_predictive_anchor', 'phase3'),
                                       ('RGB＋IR预测GRU', 'rgb_ir', 'plim_mod_dev_rgb_ir_gru_predictive_anchor', 'phase3')]:
        choice = selections[group][family]['selected']
        runs = choice.get('run_ids', choice.get('runs'))
        measurements = [json.loads((root / phase / (run + '_evaluation') / 'metrics.json').read_text())['metrics'] for run in runs]
        for seq in ('BT-005', 'SN-013'):
            fields = ('HOTA', 'AssA', 'DetA', 'IDF1', 'MOTA', 'IDSW')
            mean = {key: statistics.mean(x[seq][key] for x in measurements) for key in fields}
            std = {key: statistics.stdev(x[seq][key] for x in measurements) for key in fields}
            sequence_rows.append((label + '／' + seq, mean, std))
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
    sync = dict(status='NETWORK_UNAVAILABLE', last_confirmed_remote='6ffaababd26e5a2e2628dd02a03514ac5ecb7702')
    sync_text = '本轮 GitHub 同步遭TLS/EOF/SSH超时，最后确认的远端为6ffaaba，不能声称新提交已推送。'
    if args.confirmed_remote_head:
        confirmed = subprocess.check_output(['git', 'rev-parse', args.confirmed_remote_head], text=True).strip()
        actual_remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/research/cfp-fishmot-v1'],
                                               text=True, timeout=30).split()[0]
        if args.push_log is None or confirmed != actual_remote:
            raise ValueError('Confirmed remote must match the live remote branch')
        subprocess.check_call(['git', 'merge-base', '--is-ancestor', confirmed, commit])
        if '-> research/cfp-fishmot-v1' not in args.push_log.read_text():
            raise ValueError('Successful push log required')
        sync = dict(status='PUSH_CONFIRMED', last_confirmed_remote=confirmed,
                    push_log=str(args.push_log), push_log_sha256=sha256(args.push_log))
        sync_text = f'此前TLS/EOF/SSH故障后连接恢复，已成功推送；本报告生成时确认远端为{args.confirmed_remote_head[:7]}，成功推送日志及哈希保留在机器状态中。'
    state = dict(status='GOAL_NOT_MET_WITH_EVIDENCE', task_scope='Bounded original research hypotheses evaluated; no validated final innovative method or SOTA claim.',
                 original_goal='Reliable synthetic-modality identity association for online multi-object fish tracking',
                 reasons=['CMUE has no validated independent gain; final gate accepts zero IR rows',
                          'Paired forecasting helps its ordinary paired control but is not stably superior to strong RGB/twoRGB capacity controls',
                          'No candidate improves strongest native baseline HOTA across the complete selected three-seed development protocol',
                          'Corrected end-to-end runtime is causal and reproducible but does not meet camera-rate real-time performance'],
                 selection_sources=sources, corrected_online_report_sha256=sha256(args.online_report),
                 full_trajectory_parity=parity, terminal_batch_diagnostic_sha256=sha256(diagnostic_path),
                 final_test_frozen=False, new_historical_test_evaluation=False,
                 limitations=['Two development videos/21 identities; dense49-fish MSK scene excluded',
                              'Historical detector trained on all8 original training videos including dev',
                              'Biological identity overlap across videos is unverified',
                              'Native person-weight/own-detector ProtocolB and external fresh blind validation unperformed',
                              'Group interaction and joint SGD not introduced without controlled core usefulness evidence'],
                 free_gib=free, minimum_free_gib=30, source_commit=commit, github_sync=sync)
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

相同选择配置的每序列三种子结果，不能用 combined 的提升替代每视频一致性证据：

{table(sequence_rows, ['HOTA','AssA','DetA','IDF1','MOTA','IDSW'])}

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

现在检测器在独立 cu117 worker 中运行，身份编码器和预测器在 cu118 中运行；双方只传当前 RGB／响应数组，检测填充是当前帧的八份副本。生成器只用当前和过去帧，在响应生成后更新 EMA，所有序列采用同一配置。两套运行栈间传递无像素／提案精度损失，没有未来图像或 GT。全部生成像素与逐帧审计的因果数据精确一致，全部输出行的帧号／身份号一致，三个种子的每序列及 combined HOTA、AssA、DetA、IDF1、MOTA、IDSW 均完全一致。

逐字节边界：BT-005 全部文件和 SN-013 前2,998帧完全一致；SN-013 最后两帧每种子有8行框坐标／置信度差异。缓存跨视频的4,834帧按8分批，最后未补齐批次只有2帧，在线始终采用8份当前帧。末端受控检查表明第2,997／2,998帧仅批次8精确复现、第2,999／3,000帧仅批次2精确复现，解释了差异。原始在线文件未改写；不能声称全部轨迹逐字节相同。定位敏感的LocA/MOTP/sMOTA等有约1e-5个百分点差异，机器状态保留全部逐行和指标差异。

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
| 1. V3是否准确复现 | 历史参考输入／输出精确恢复；新严格在线入口的末端批次边界单独披露 |
| 2. 真实AssA、DetA、HOTA | 最强开发基线37.498／73.518／52.482；RGB＋IR预测37.715／71.447／51.885，均为官方三种子均值 |
| 3. IR何时有效 | 固定融合内部正确结构优于破坏结构；配对预测优于普通配对GRU，但未证明强RGB之上的独立增益 |
| 4. IR何时有害 | 强RGB预测上固定混合及打乱／置零辅助分支退化；没有可靠条件选择器 |
| 5. CMUE优于质量门控吗 | 第三轮联合数值高于质量融合，却接受IR为零，与PLIM完全相同，因此不是CMUE增益 |
| 6. 反事实监督有独立增益吗 | 三轮和常数／排序控制均不支持，不能归因反事实监督 |
| 7. PLIM优于EMA/普通GRU/运动吗 | 部分预测高于简单记忆和ByteTrack；普通单RGBGRU更强，未普遍优于所有控制 |
| 8. 显著减少交叉IDSW吗 | RGB预测总切换均值83.667对ByteTrack86；开发视频少且事件不一致，未证明统计显著交叉恢复提升 |
| 9. 哪个创新点提高HOTA最多 | 配对预测相对普通配对控制约2.420个百分点，但强基线和双RGB控制限制创新归因 |
| 10. 哪个创新点提高AssA最多 | 配对预测相对普通配对控制约3.465个百分点；未证明IR特有贡献，CMUE/拒写无独立收益 |
| 11. MOTA受影响吗 | BoT93.552、配对预测92.611；完整表披露坐标/生命周期导致的检测与关联权衡 |
| 12. IDF1升而HOTA不升吗 | 相对BoT配对预测IDF1由57.477至62.242，HOTA由52.482至51.885，确有权衡 |
| 13. 三种子稳定吗 | 原锚点Transformer53.055单种子未复现，平均50.606±2.214；配对51.885±1.384，不能声称稳定领先 |
| 14. 每序列一致吗 | 上表逐视频完整报告；收益/退化和事件不完全一致，不以combined数字替代 |
| 15. 真实FPS/显存/延迟 | 完整RGB预测管线约1.20–1.23FPS，父/worker分配峰值及p50/p95见成本表；配对只完成前缀工程验证，不冒充完整配对速度 |
| 16. 对HybridSORT等公平吗 | ProtocolA共用提案/鱼类特征/开发预算，HybridSORT46.657；原生输出规则不同，人类权重/原检测完整ProtocolB未复现 |
| 17. 真实失败原因 | pre-neck窄锥、遗漏候选、历史污染、时间饱和、运行栈/末端批次等分别诊断和修复；时间修复无收益，未知标签不等于错误 |
| 18. 完全在线吗 | 修正入口只用当前/过去，不读GT；完整像素/身份编号一致，末端数值差异明确定位 |
| 19. 与已有模块的区别成立吗 | 反事实收益/潜在预测/独立写入有可运行设计，但目前控制实验不足以确立独立有效方法贡献 |
| 20. 足够支撑独立方法论文吗 | 不足；可复现负结果和平台证据成立，原计划稳定CMUE＋IR独立贡献未达到 |
'''
    report += f'''
## 结论、边界与资源

在预先固定的两视频开发协议和受控迭代内，原科学成功条件未满足；没有宣布 SOTA 或完整创新方法成功。保留所有负结果、有效控制和可复现在线入口。密集场景补充交叉验证、真正新的外部盲测、原生完整系统 ProtocolB、联合 SGD 和可选群体交互没有执行，不能标为已完成；没有核心独立效用证据时，未强行添加交互模块或继续扩大参数搜索。历史七序列未用于新模型选择，也没有新方法最终盲测声明。

当前剩余约{free:.1f}GiB，最低保留30GiB。本轮不新增图像/裁剪/特征银行，仅保存小模型、轨迹、指标与日志；新增受控产物约1.1GiB，原数据、权重、检测缓存、历史记录保留。最多使用三张物理GPU，符合四卡上限。代码在research/cfp-fishmot-v1；源码提交{commit[:7]}。{sync_text}

主要入口：tools/cfp_run_online.py；正确配置configs/cfp/online_anchor_gru_seed{{0,1,2}}_split_v1.json。训练/重放/评估命令和哈希在experiment_registry及对应不可变manifest中。摘要状态：RESEARCH_OUTCOME_20261012.json；完整机器报告：{args.online_report}。
'''
    Path('docs/research/FINAL_RESEARCH_REPORT.md').write_text(report)
    write_json(root / 'current_research_state_20261012.json', state)
    print('Compiled bounded outcome:', state['status'], 'identity parity verified; terminal batch differences retained')


if __name__ == '__main__':
    main()
