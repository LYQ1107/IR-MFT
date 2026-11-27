#!/usr/bin/env python3
"""Export the negative frozen-token pilot and its pre-GPU registration failure."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
WORK=Path('/var/tmp/liuyeqiang_ir_mft_perceptual_20261122')
DOC=ROOT/'docs/fusion_research'
DEST=DOC/'perceptual_20261127_frozen_tokens'
NAME='PERCEPTUAL_MFT_20261127_FROZEN_TOKENS'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def main():
    base=WORK/'frozen_tokens_20261127';reg=read(base/'registered.json');result=read(base/'complete_oracle_feature_diagnostic.json')
    for n in ['run_r1','audit_r1']:
        e=read(base/(n+'.execution.json'));assert e['actual_child_exit_code']==0 and e['source_unchanged']
        for path,h in e['source_hashes'].items():assert sha(ROOT/path)==h
    assert not result['representation_gate_passed']and not result['IR_identity_gate_passed']
    assert not DEST.exists()and not(DOC/(NAME+'.json')).exists()and not(DOC/(NAME+'.md')).exists()
    files=[];copies={}
    def copy(source,relative):
        source=Path(source);target=DEST/relative;assert source.is_file()and source.stat().st_size<512*1024
        assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);assert sha(target)==sha(source)
        name=str(target.relative_to(ROOT));files.append(name);copies[name]=dict(original_path=str(source),bytes=source.stat().st_size,sha256=sha(source),byte_exact=True)
    for n in ['CPU_regressions.json','registered.json','state.json','run_r1.execution.json','GPU_4.json','GPU_5.json',
        'analysis_registered.json','audit_r1.execution.json','audit_r1.stdout.txt','complete_oracle_feature_diagnostic.json']:
        copy(base/n,n)
    for p in sorted(base.glob('*.supervisor.py')):copy(p,p.name)
    for p in sorted((base/'private_bank').glob('*.json')):copy(p,Path('descriptor_receipts')/p.name)
    for p in sorted((base/'logs').glob('*.log')):copy(p,Path('worker_logs')/p.name)
    for p in sorted((base/'registration_failure_r1').iterdir()):
        assert p.is_file();copy(p,Path('registration_failure_r1')/p.name)
    compressed=sum(read(p)['bytes']for p in (base/'private_bank').glob('*.json'))
    summary=dict(result,private_descriptor_payload_bytes=reg['private_descriptor_payload_bytes'],compressed_bank_bytes_not_uploaded=compressed,
        fixed_model_source_commit=reg['source_vendor']['commit'],fixed_source='https://github.com/facebookresearch/dinov2',
        fixed_checkpoint_sha256=reg['checkpoint_sha256'],new_weight_downloads_or_copies=0,all_GPU_jobs_and_controller_actual_exit0=True,
        CPU_analysis_actual_exit0=True,initial_registration_actual_exit1_before_GPU_preserved=True,
        original_RGB_tracking_baseline_HOTA=29.528970687393972,minimum_tracking_target_HOTA=34.52897068739397,
        no_new_Dev_Test_or_tracking_HOTA=True,no_frozen_DINO_fullTrain_expansion=True,goal_status='active')
    p=DOC/(NAME+'.json');write(p,summary);files.append(str(p.relative_to(ROOT)))
    a=result['equal_video_known_top1_accuracy']
    lines=['# Frozen DINOv2：实际 Train proposals 的表征筛选负结果','',
        f'在相同 384 个均匀抽样 Train 帧、6,752 个原始 proposals 的 oracle 身份诊断中，旧 RGB ReID 的等视频 Top1 为 **{100*a["legacy_RGB"]:.6f}%**，DINO RGB patch mean 为 **{100*a["DINO_RGB_mean"]:.6f}%**，下降 **{-result["DINO_RGB_minus_legacy_RGB_pp"]:.6f}** 个点。RGB+新版 IR 为 **{100*a["DINO_RGB_IR_mean"]:.6f}%**，仅比 DINO RGB 高 **{result["DINO_paired_minus_DINO_RGB_pp"]:.6f}** 点，比 RGB+灰度低 **{-result["DINO_paired_minus_DINO_gray_pp"]:.6f}** 点。两个预注册门槛均未通过，因此不扩大该固定表征配置到全 Train 或 Dev／Test。','',
        '这些数值是 Train 中使用 GT 标识 gallery 的身份检索诊断，不是在线关联准确率，也不是 HOTA。原始 RGB 跟踪 baseline 继续固定为 29.528971，+5 目标继续为 34.528971。该诊断不增加任何多模态创新点。','',
        '## 真实执行和公平范围','',
        '使用 [DINOv2 官方代码](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8) 的 ViT-B/14，固定代码提交及现有 346,378,731 字节 checkpoint 的 SHA256，冻结全部参数。权重没有重新下载或复制。新编码器与旧 ResNet50 ReID 不同，预训练与计算预算也不同；它只改变我们候选方法的外观表征，没有改动原始 RGB tracking baseline。每种新模态控制共享同一 DINO backbone。','',
        'GPU4／5 各运行一个实际图像作业，全部作业与控制器实际退出 0。三种输入为 RGB、当前固定新版 DiffV2IR、当前 RGB 灰度；所有原始 proposals 在 GT 读取前提取。224 方形输入保留长宽比，padding 为归一化中性值，CLS 与按有效像素面积加权的 16×16 patch token mean 均归一化。每段第一帧重复 FP32 提取相同，模型权重与 buffers 前后相同，FP16 最终描述子量化误差小于 1e-3。没有保存 patch token、crop、图像或 FPN。','',
        f'六种最终描述子理论 payload 为 {reg["private_descriptor_payload_bytes"]:,} 字节，实际压缩 NPZ 共 {compressed:,} 字节，均留在本机。Git 只保留轻量来源、数量、字节哈希及误差收据。','',
        'CPU 消费者实际退出 0，独立重建了原始 detector 与交付图像包哈希。每段视频前 75% 的可靠已知正例中，各身份取最早三个观测建立 gallery；有三次早期观测但没有后期 query 的身份也保留为干扰项。后 25% 的可靠已知正例作为 query，要求距 gallery 最后观测至少 25 帧，固定 gallery-ID 数值顺序处理 1e-12 近似平局。共 '+str(result['queries'])+' 个 query。这里的 gallery 使用 Train GT 身份，不能作为可部署的在线身份缓存；所有 unknown proposals 仍被图像编码，仅在这个身份诊断指标中排除，不作背景。没有参数拟合或随机种子重复。','',
        '## 全部等视频结果','', '| 表征 | Top1 (%) |','|---|---:|']
    for name,value in a.items():lines.append(f'| {name} | {100*value:.6f} |')
    lines+=['','CLS 只是预先记录的辅助结果；没有因 primary mean 失败而切换成新的正式选择。重复 RGB 控制与单 RGB 分数的误差小于 1e-12，预测一致性逐段保存。全部逐序列指标和 query／gallery 数量见 JSON。','',
        '## 保留的注册错误与来源限制','',
        '首次注册把上一轮 64 个均匀抽样帧误写为前 64 帧，计数校验实际退出 1，在任何新 GPU 输入或特征之前停止。失败时的源码、配置和退出记录全部保留。随后在新 GPU 工作前修正为旧注册中同一组 384 个稀疏帧，6,752 个 proposals 计数通过；没有用新特征结果改变抽帧。','',
        '旧 ReID encoder 见过 Train 身份，DINO 外部预训练与 MFT 的图像重叠没有建立，因此这不是新盲测或严格跨域评估。该有限诊断仅排除当前固定冻结描述子方案，不给其他模型或后续训练下结论。原始 Test 已历史暴露的限制保持。','',
        f'完整数值见 [{NAME}.json]({NAME}.json)，退出记录、负结果及原始失败快照见 [{DEST.name}/]({DEST.name}/)。复现顺序为 `run_mft_frozen_tokens.py register/run` → `analyse_mft_frozen_tokens.py`；当前 sampler 读取固定旧注册中的 sparse frame IDs。']
    p=DOC/(NAME+'.md');p.write_text('\n'.join(lines)+'\n');files.append(str(p.relative_to(ROOT)))
    p=DEST/'byte_exact_artifact_index.json';write(p,dict(copies=copies));files.append(str(p.relative_to(ROOT)))
    paths=[ROOT/n for n in files];assert max(p.stat().st_size for p in paths)<512*1024
    manifest=WORK/'frozen_tokens_public_export_20261127.json'
    write(manifest,dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in paths},
        total_bytes=sum(p.stat().st_size for p in paths),max_file_bytes=max(p.stat().st_size for p in paths)))
    print(json.dumps(dict(status='EXPORTED_NEGATIVE_DINO_PILOT',files=len(files),total_bytes=sum(p.stat().st_size for p in paths))))


if __name__=='__main__':main()
