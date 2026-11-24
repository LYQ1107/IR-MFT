#!/usr/bin/env python3
"""Publish complete scalar diagnostics in bounded files, never feature arrays."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    spec_path = ROOT/'configs/perceptual_mechanism_diagnosis_20261125.json'
    specification = json.loads(spec_path.read_text())
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    path = Path(plan['work_root'])/'structural_train_mechanism_20261125.json'
    value = json.loads(path.read_text())
    assert value['status'] == 'COMPLETE_READ_ONLY_STRUCTURAL_FULL_TRAIN_MECHANISM_DIAGNOSIS'
    assert value['specification_sha256'] == sha(spec_path)
    assert value['Train_unique_frames'] == 72 and value['Train_domains'] == 6
    assert value['cases'] == specification['train']['cases']
    assert not value['Dev_read'] and not value['Test_read'] and not value['model_fit']
    assert all(sha(ROOT/name) == digest for name, digest in value['source_hashes'].items())
    assert len(value['current_raw_source_images_matched_to_completed_IR_fit']) == 144
    docs = ROOT/'docs/fusion_research'
    stem = docs/'PERCEPTUAL_MFT_20261125_STRUCTURAL_TRAIN_MECHANISM'
    directory = docs/'perceptual_20261125_train_mechanism'
    assert not stem.with_suffix('.json').exists() and not stem.with_suffix('.md').exists()
    directory.mkdir(exist_ok=False)
    files, copied = [], []
    for case in value['cases']:
        identifier = case['id']
        assert identifier.replace('_', '').isalnum()
        rows = [r for r in value['pixel_and_anchor_statistics'] if r['case'] == identifier]
        support = [r for r in value['GT_anchor_support'] if r['case'] == identifier]
        assert len(support) == 7 and set(r['scope'] for r in support) == set(
            [r['sequence'] for r in specification['train']['sample']]+['ALL_TRAIN_SAMPLE'])
        item = dict(case=case, pixel_and_anchor_statistics=rows, GT_anchor_support=support,
                    scope=value['scope'], parent_full_result_sha256=sha(path))
        target = directory/(identifier+'.json')
        target.write_text(json.dumps(item, ensure_ascii=False, indent=2)+'\n')
        assert target.stat().st_size < 2**20
        files.append(dict(case=identifier, file=str(target.relative_to(docs)), sha256=sha(target),
                          statistic_rows=len(rows), bytes=target.stat().st_size))
        copied.extend(rows)
    assert copied == [r for case in value['cases'] for r in value['pixel_and_anchor_statistics'] if r['case'] == case['id']]
    assert len(copied) == len(value['pixel_and_anchor_statistics'])
    index = {k: v for k, v in value.items() if k not in ['pixel_and_anchor_statistics', 'GT_anchor_support']}
    index.update(public_scalar_case_files=files, public_all_scalar_rows_preserved=True,
        parent_full_result_sha256=sha(path), exporter_source_sha256=sha(Path(__file__)))
    stem.with_suffix('.json').write_text(json.dumps(index, ensure_ascii=False, indent=2)+'\n')
    assert stem.with_suffix('.json').stat().st_size < 2**20
    def mean(case, region, variable):
        found = [r for r in value['pixel_and_anchor_statistics'] if r['case'] == case and
                 r['scope'] == 'ALL_TRAIN_SAMPLE' and r['region'] == region and r['variable'] == variable]
        assert len(found) == 1
        return found[0]['mean']
    def fmt(number):
        return 'NA' if number is None else f'{number:.6f}'
    lines = ['# 完整Train拟合后的结构融合：72帧机制诊断', '',
        '六段Train各12帧，共72帧，检查已完成三遍全Train的RGB、灰度、IR111、IR011，'
        '并保留IR111第一／二／三遍快照。没有更新参数、运行GPU或读取Dev/Test。'
        '这些是训练集样本的机制量，不是完整Train统计或泛化、检测Recall、HOTA。', '',
        '| 模型 | P3弱RGB框内结构置信 | P3背景结构置信 | P3弱RGB框内修正/RGB RMS | 前景锚点概率变化 | 固定0.6支持增加/减少 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for case in value['cases']:
        identifier = case['id']
        support = next(r for r in value['GT_anchor_support'] if r['case'] == identifier and r['scope'] == 'ALL_TRAIN_SAMPLE')
        counts = support['thresholds']['0.6']
        values = [mean(identifier, 'P3_foreground_teacher_lt06', 'structural_confidence'),
            mean(identifier, 'P3_background', 'structural_confidence'),
            mean(identifier, 'P3_foreground_teacher_lt06', 'correction_RGB_RMS'),
            mean(identifier, 'anchors_foreground', 'probability_delta')]
        lines.append('| '+identifier+' | '+' | '.join(fmt(v) for v in values)+
                     f" | +{counts['gained']} / −{counts['lost']} |")
    lines += ['',
        '弱RGB框内指Train框覆盖的网格中心且教师概率<0.6；背景为有效缩放图像内、框外网格，'
        '不含padding。框不是像素分割，框重叠不是实测遮挡。统计按位置／锚点数加权；'
        '逐域、P3/P4/P5、单框／重叠框、高／低教师概率、矩与极值全部保留在六份标量JSON。', '',
        '固定0.6支持指：每个GT在原RGB解码且IoU≥0.5的稠密锚点中，最高语义概率是否越过0.6。'
        '没有NMS或ByteTrack，重复锚点保留；没有足够几何重叠的GT也原样计入。'
        '它不是正式召回率或跟踪提升，也不能据此调阈值。0.1与0.7支持量同样完整保留。', '',
        f"CPU FP32用时{value['seconds']:.3f}秒，进程峰值RSS{value['peak_process_RSS_GiB']:.3f}GiB。"
        '原head及六组adapter的参数、BN和梯度未变，最终语义输出的几何与原RGB严格相同；'
        '局部高教师置信位置的特征修正为零，RMS界逐项核实。该保护不意味着卷积后语义完全不变。', '',
        '所有原有缓存数组完整SHA256核对。144个当前原始RGB/IR文件的字节摘要与已完成IR拟合'
        '输入清单匹配；这里没有重新编码这些图像。历史GPU B1缓存等价证明覆盖12帧，'
        f"其中{value['current_sample_frames_already_in_historical_GPU_parity']}帧属于本次样本；"
        '没有将该局部证明冒充72帧新编码或CPU/GPU等价。', '',
        '完整标量索引及来源见相邻JSON，每份公开文件小于1MiB；没有保存或上传图像、权重、'
        'FPN、原始轨迹或大数组。该诊断不能决定Test模型、证明原创性或替代本轮完整Dev。'
        '独立贡献0、+5目标未达，Goal保持active。', '',
        '复现：CPU运行 `tools/diagnose_structural_train_mechanism.py`，完成后运行 '
        '`tools/export_structural_train_mechanism.py`；已有产物不覆盖。']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='EXPORTED_COMPLETE_TRAIN_SCALAR_DIAGNOSIS', cases=6,
                         statistic_rows=len(copied), largest_case_bytes=max(r['bytes'] for r in files))), flush=True)


if __name__ == '__main__':
    main()
