#!/usr/bin/env python3
"""Plot and describe the already-completed Dev-only domain diagnosis."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    stem = ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261125_NATIVE_DEV_DOMAINS'
    result = json.loads(stem.with_suffix('.json').read_text())
    assert result['status'] == 'COMPLETE_EXISTING_FROZEN23_FULL_DEV_DOMAIN_DIAGNOSIS'
    assert result['actual_Test_accuracy_inputs'] == [] and not result['new_inference']
    assert not stem.with_suffix('.md').exists() and not stem.with_suffix('.png').exists()
    sequences = ['BT-005', 'SN-013', 'COMBINED']
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 10), constrained_layout=True)
    for axis, metric in zip(axes, ['HOTA', 'AssA']):
        values = np.array([[p['differences'][seq]['IR_minus_RGB'][metric] for seq in sequences]
                           for p in result['pairs']])
        limit = max(float(np.abs(values).max()), .001)
        display = axis.imshow(values, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
        axis.set_xticks(range(3), sequences)
        axis.set_yticks(range(23), [p['IR_group'].split('_')[0] for p in result['pairs']])
        axis.set_title(f'IR minus matched RGB: {metric} (points)')
        for (i, j), value in np.ndenumerate(values):
            axis.text(j, i, f'{value:+.2f}', ha='center', va='center', fontsize=8,
                      color='white' if abs(value) > .6*limit else 'black')
        fig.colorbar(display, ax=axis, shrink=.55, label='Absolute percentage points')
    fig.suptitle('Frozen seed-0 Dev cue pairs: domain-dependent effects\n'
                 '23 behavior pairs in one family; 2 full sequences, 4,834 frames', fontsize=12)
    fig.savefig(stem.with_suffix('.png'), dpi=150)
    plt.close(fig)
    summary = result['summary']
    lines = ['# 新版IR身份分支：完整Dev的逐序列诊断', '',
        f"原先58组行为配置中的23组RGB/IR配对，汇总Dev HOTA较高的有{summary['combined_Dev_IR_positive']}组；"
        f"两个Dev序列都较高的只有{summary['both_Dev_sequences_IR_positive']}组，"
        f"有{summary['opposite_Dev_sequence_signs']}组收益方向随序列反转。", '',
        '进一步要求两个序列都胜过匹配RGB及自身增量关闭对照，满足的为0组。'
        '这是完整Dev上的必要数值检查，不能当成创新性证明或总体因果结论。', '',
        '本次只读取原先冻结的Dev快照、完整4834帧指标和276列官方CSV，'
        '没有重新推理、拟合、搜索阈值或读取Test准确率。原始双Dev选择不改，'
        '12个未配对的IR干预行为仍保留。23对不是23种架构，也不是独立重复实验。', '',
        '![Dev中IR相对匹配RGB的逐序列差值](PERCEPTUAL_MFT_20261125_NATIVE_DEV_DOMAINS.png)', '',
        '颜色与文字均为IR−RGB绝对百分点；图中保留两位小数，不表示统计显著性。'
        '完整精度、确切快照、逐项配置、来源与关闭对照见相邻JSON。', '',
        '| IR配置 | BT-005 ΔHOTA | SN-013 ΔHOTA | COMBINED ΔHOTA | 两段均正 |',
        '| --- | ---: | ---: | ---: | --- |']
    for row in result['pairs']:
        differences = [row['differences'][seq]['IR_minus_RGB']['HOTA'] for seq in sequences]
        lines.append('| '+row['IR_group']+' | '+' | '.join(f'{v:+.6f}' for v in differences)+
                     ' | '+('是' if row['both_Dev_sequences_positive'] else '否')+' |')
    row = next(p for p in result['pairs'] if p['IR_group'] == 'NR36_native_query_rgb_ir')
    lines += ['', '## NR36的反转来自哪些指标', '',
        '该分支两个序列的DetA变化都远小于AssA变化：', '',
        '| Dev序列 | ΔHOTA | ΔDetA | ΔAssA | ΔIDF1 | ΔIDSW | ΔFrag |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seq in sequences:
        delta = row['differences'][seq]['IR_minus_RGB']
        lines.append(f"| {seq} | "+' | '.join(f'{delta[k]:+.6f}' for k in ['HOTA', 'DetA', 'AssA', 'IDF1'])+
                     f" | {delta['IDSW']:+d} | {delta['Frag']:+d} |")
    lines += ['',
        'BT-005的匹配RGB HOTA51.180303，而IR43.934719；SN-013则为RGB36.598172、'
        'IR39.196347。汇总DetA略增而AssA下降，不能靠汇总IDSW接近就认定身份保持接近。', '',
        '脚本还逐个IoU阈值核验 `log(HOTA_IR/HOTA_RGB) = '
        '0.5 log(DetA_IR/DetA_RGB) + 0.5 log(AssA_IR/AssA_RGB)`。'
        'COMBINED的19个阈值中，关联项绝对值都更大。这里只是官方指标的代数分解；'
        '没有用DetA与AssA的AUC均值重建HOTA，也不把平均对数项称为百分点或因果贡献。', '',
        '## 对下一步研究的约束', '',
        '这支持继续区分检测变化和身份变化，并要求分序列、普通RGB记忆、相同质量权重、'
        '自身关闭和真实错位／缺失控制；不能据序列标签设计Test路由，'
        '也不能据此证明某一种质量门控已经有效。', '',
        'NR24在两个序列都略胜匹配RGB，但相对自身关闭并非两段都正；NR07汇总仍低于'
        '自身关闭。NR38和NR41的IR汇总均47.348568，对较弱RGB43.570603为正，'
        '仍低于原始RGB Dev约47.579，且缺少严格关闭对照。不能从这四项选择新主模型。', '',
        f"关闭对照本次可从{summary['original_Dev_own_off_available']}项原始完整Dev读取，"
        '比先前只在回放补测目录核对的8项多；这是找回原始已有证据，没有新增推理。'
        '未知关闭结果仍保留未知。', '',
        '并行的72帧Train诊断预登记见'
        '[mechanism specification](../../configs/perceptual_mechanism_diagnosis_20261125.json)。'
        '它检查全Train拟合后的实际视觉修正，不是上述描述向量分支的新融合架构，'
        '也不替代尚待完成的完整Dev/Test。', '',
        '历史Test已暴露、原RGB预训练见过当前Dev、辅助表示源于RGB的限制继续保留。'
        '当前独立多模态贡献0、+5目标未达，Goal保持active。', '',
        '复现：先在CPU环境运行 `tools/diagnose_native_dev_domains.py`，'
        '再运行 `tools/report_native_dev_domains.py`。输入与输出已存在时脚本拒绝覆盖。']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='COMPLETE_DEV_DOMAIN_REPORT_AND_FIGURE',
        input_sha256=hashlib.sha256(stem.with_suffix('.json').read_bytes()).hexdigest(),
        figure_bytes=stem.with_suffix('.png').stat().st_size)), flush=True)


if __name__ == '__main__':
    main()
