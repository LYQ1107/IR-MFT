#!/usr/bin/env python3
"""Export only lightweight, audited quality-study scientific evidence."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
WORK=Path('/var/tmp/liuyeqiang_ir_mft_perceptual_20261122')
DOC=ROOT/'docs/fusion_research'
DEST=DOC/'perceptual_20261127_proposal_quality'
NAME='PERCEPTUAL_MFT_20261127_PROPOSAL_QUALITY'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    p=Path(p);assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    train=WORK/'proposal_quality_20261127';base=WORK/'proposal_quality_evaluation_20261127';online=WORK/'proposal_quality_online_probe_20261127'
    audit=read(base/'independent_results_audit.json');assert audit['all_required_jobs_and_controller_actual_exit0']
    assert read(base/'audit_r1.execution.json')['actual_child_exit_code']==0
    assert read(online/'run_r1.execution.json')['actual_child_exit_code']==0
    assert not DEST.exists()and not(DOC/(NAME+'.json')).exists()and not(DOC/(NAME+'.md')).exists()
    copies={};files=[]
    def copy(source,relative):
        source=Path(source);target=DEST/relative;assert source.is_file()and not target.exists()
        assert source.stat().st_size<512*1024,(str(source),source.stat().st_size)
        if source.suffix=='.json':
            def inspect(v):
                if isinstance(v,dict):
                    assert not set(v)&{'coefficients','direction','center'},str(source)
                    for x in v.values():inspect(x)
                elif isinstance(v,list):
                    for x in v:inspect(x)
            inspect(read(source))
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);assert sha(target)==sha(source)
        n=str(target.relative_to(ROOT));copies[n]=dict(original_path=str(source),sha256=sha(source),bytes=source.stat().st_size,byte_exact=True);files.append(n)
    for label,folder,names in [
        ('Train',train,['CPU_regressions_r2.json','real_Train_engineering.json','engineering_r1.execution.json',
            'fit_registered.json','Train_coverage.json','fit_complete.json','fit_history.json','fit_r1.execution.json',
            'independent_fit_audit_registered.json','independent_fit_audit.json','audit_r1.execution.json',
            'engineering_r1.stdout.txt','fit_r1.stdout.txt','audit_r1.stdout.txt']),
        ('evaluation',base,['registered.json','real_Train_engineering.json','Test_selection_frozen.json','state.json',
            'run_r1.execution.json','audit_r1.execution.json','independent_results_audit.json','audit_r1.stdout.txt']),
        ('actual_image_engineering',online,['registered.json','state.json','GPU_4.json','GPU_5.json','run_r1.execution.json'])]:
        for n in names:copy(folder/n,Path(label)/n)
    for folder,label in [(train,'Train'),(base,'evaluation'),(online,'actual_image_engineering')]:
        for p in sorted(folder.glob('*.supervisor.py')):copy(p,Path(label)/p.name)
    groups=[('dev','complete_Dev.json'),('dev_intervention','complete_Dev_interventions.json'),('test','full_Test_results.json')]
    if (base/'full_Test_interventions.json').exists():groups.append(('test_intervention','full_Test_interventions.json'))
    all_rows={}
    for label,name in groups:
        result=read(base/name);all_rows[label]=[]
        for row in result['rows']:
            case=row['case']['id'];folder=base/'evaluation'/label/case
            for n in ['metrics.json','trackeval_output/pedestrian_detailed.csv','trackeval_output/pedestrian_summary.txt']:
                copy(folder/n,Path('evaluation')/label/case/n)
            all_rows[label].append(dict(case=row['case'],frames=row['frames'],combined47=row['combined47'],
                per_sequence47=row['per_sequence47'],official_CSV_sha256=row['official_CSV_sha256'],
                original_metrics_sha256=sha(folder/'metrics.json')))
        for p in sorted((base/'streams'/label/'sequence_manifests').glob('*.json')):
            copy(p,Path('sequence_manifests')/label/p.name)
    for phase,jobs in read(base/'state.json')['phases'].items():
        for name,job in jobs.items():
            assert job['status']=='COMPLETE'and job['exit_code']==0
            copy(job['log'],Path('worker_logs')/Path(job['log']).name)
    canonical=WORK/'native_pyramid_evaluation/test/NP_source_canonical_RGB/trackeval_output/pedestrian_detailed.csv'
    assert sha(canonical)=='c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
    copy(canonical,Path('baseline')/'pedestrian_detailed.csv')
    fit=read(train/'fit_complete.json');dev=read(base/'complete_Dev.json');test=read(base/'full_Test_results.json')
    selected=next(r for r in all_rows['test']if r['case']==test['selected_case'])
    parent=next(r for r in all_rows['test']if r['case']['id']=='PQ_archive_control')
    comparison={r['case']['kind']:r for r in all_rows['test']if r['case']['kind']in ['RGB','paired','duplicatedRGB']}
    baseline=test['original_RGB_baseline']['combined47']['HOTA'];delta=selected['combined47']['HOTA']-baseline
    assert delta==test['HOTA_gain_over_original_RGB_pp']and delta<5
    summary=dict(status='AUDITED_COMPLETE_QUALITY_TRAIN_DEV_TEST_NEGATIVE_FOR_PLUS5_AND_CAUSAL_IR',UTC=utc(),
        scope='Fixed November DiffV2IR and ProtocolV2; original direct RGB baseline only; strongest RGB architecture sweeps deferred.',
        fixed_original_RGB_baseline_HOTA=baseline,minimum_target_HOTA=baseline+5,stretch_target_HOTA=baseline+10,
        selected_case=test['selected_case'],selected_Test_HOTA=selected['combined47']['HOTA'],gain_over_original_RGB_pp=delta,
        selected_minus_previous_formal_parent_Test_pp=selected['combined47']['HOTA']-parent['combined47']['HOTA'],
        previous_formal_parent_Test_HOTA=parent['combined47']['HOTA'],
        paired_minus_matched_RGB_Test_pp=comparison['paired']['combined47']['HOTA']-comparison['RGB']['combined47']['HOTA'],
        paired_minus_duplicatedRGB_Test_pp=comparison['paired']['combined47']['HOTA']-comparison['duplicatedRGB']['combined47']['HOTA'],
        Dev_gain_over_parent_pp=dev['gain_over_parent_best_Dev_pp'],all_official_rows=all_rows,
        original_RGB_baseline_all47=test['original_RGB_baseline'],official_fields_independently_reparsed=audit['official_fields_reparsed'],
        Train_frames=fit['Train_frames'],Train_proposals=fit['proposals'],early_known_fit_rows=fit['early_known_rows'],
        frozen_model_sha256=fit['model_sha256'],fixed_Newton_updates=12,head_diagnostics=fit['final_diagnostics'],
        full_Train_independent_fit_audit_actual_exit0=True,all_required_evaluation_jobs_and_independent_results_audit_actual_exit0=True,
        actual_image_engineering=dict(physical_GPU_ids=[4,5],two_jobs_actual_exit0=True,visits_per_GPU=42,
            whole_IR_missing_visual_head_exact_same_source_fresh_RGB=True,all16_online_replay_MOT_exact=True,
            normal_reset_prefix6_exact=True,no_accuracy_or_full_sequence_runtime_claim=True),
        causal_assessment=dict(Dev_IR_descriptor_roll_HOTA_exact_selected=True,
            qualityIR_missing_MOT_and_all47_exact_RGB=True,
            quality_ROI_IR_controls_do_not_remove_visual_detector_IR=True,
            promotion_changes_main_association_births_and_template_confidence=True,
            original_detector_packets_boxes_and_order_preserved=True,
            tracker_input_scores_changed_and_separately_hashed=True,
            all_calibrated_input_packet_hashes_and_promotion_counts_rebuilt=True,
            supported_multimodal_innovations=0),
        limitations=['Historical Test previously exposed; this is not a fresh blind test.',
            'Historical RGB detector training included Dev videos; new quality fit uses only Train.',
            'ReID encoders have prior Train identity exposure.',
            'Train banks store FP16 descriptors; Dev/Test banks FP32. Fit/inference restore roundoff max2.98e-8 disclosed.',
            'No held-out-domain or physical thermal-sensor claim; DiffV2IR is RGB-derived.',
            'No new complete Test ablations: the pre-registered plus5 trigger was not met.',
            'Online GPU prefix checks do not establish full-sequence speed or whole-IR causal tracking gain.'],
        model_weights_images_features_crops_or_raw_MOT_uploaded=False,new_training_seeds=0,
        goal_status='active',goal_met=False,supported_innovations=0)
    report=DOC/(NAME+'.json');write(report,summary);files.append(str(report.relative_to(ROOT)))
    lines=['# Proposal quality 与因果确认：完整 Train、Dev 和 Test7 结果','',
        f'冻结方法 `{test["selected_case"]["id"]}` 的完整 Test HOTA 为 **{selected["combined47"]["HOTA"]:.6f}**，相对原始 RGB YOLOX-X + ByteTrack 的 **{baseline:.6f}** 提高 **{delta:.6f}** 个绝对点。未达到 +5；此前正式 Dev 选出的父配置为 {parent["combined47"]["HOTA"]:.6f}，本轮下降 {parent["combined47"]["HOTA"]-selected["combined47"]["HOTA"]:.6f}。目标继续保持 active，多模态创新点计 0。','',
        '主要比较按用户最新范围固定为原始 RGB baseline。最强 RGB／跨结构搜索已延期。本轮所有对照均如实保留，不依据 Test 改选。','',
        '## 完整 Test7：15,985 帧','',
        '| 配置 | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |','|---|---:|---:|---:|---:|---:|---:|']
    b=test['original_RGB_baseline']['combined47'];lines.append('| 原始 RGB baseline | '+' | '.join(f'{b[k]:.6f}'for k in ['HOTA','DetA','AssA','IDF1','MOTA','IDSW'])+' |')
    for row in all_rows['test']:
        lines.append('| '+row['case']['id']+' | '+' | '.join(f'{row["combined47"][k]:.6f}'for k in ['HOTA','DetA','AssA','IDF1','MOTA','IDSW'])+' |')
    lines+=['',f'配对质量模型比同源 RGB 质量控制低 {-summary["paired_minus_matched_RGB_Test_pp"]:.6f} HOTA，比重复 RGB 容量控制低 {-summary["paired_minus_duplicatedRGB_Test_pp"]:.6f}。这些 Test 控制不被提升为正式方法。','',
        '## 完整 Dev：4,834 帧','',
        '| 配置 | HOTA | DetA | AssA | IDF1 |','|---|---:|---:|---:|---:|']
    for label in ['dev','dev_intervention']:
        for row in all_rows[label]:lines.append('| '+row['case']['id']+' | '+' | '.join(f'{row["combined47"][k]:.6f}'for k in ['HOTA','DetA','AssA','IDF1'])+' |')
    lines+=['','Dev 冻结选择只提高 0.014687 HOTA。候选之间置换红外描述子后成绩完全相同；关闭质量分支红外时，完整 MOT 与全部 47 指标严格等于同源 RGB 质量模型。确认消融影响很小。当前宽松提分的主要负面变化是 AssA 和 IDSW，不能把前景分类 loss 下降解释为跟踪创新。','',
        '## 训练与复核','',
        '六段 Train 共 27,251 帧、249,639 个原始 proposals 全部参与收集。每段前 75% 的已知标签用于拟合，后 25% 仅作阈值校准；未知保持未知。先拟合 512D 二类前景线性判别方向，再对几何和前景证据拟合 RGB／配对／重复 RGB 质量头。每个头固定 12 次确定性 Newton 更新，最终梯度范数约 1e-17，未额外重复随机种子。这里没有重新训练 CNN。','',
        '独立审计重放全部 Train，重建确认、当前标签及几何，独立重建判别方向和排序分位数，未修改模型。结果审计重新解析 '+str(audit['official_fields_reparsed'])+' 个官方字段，并逐帧独立重建全部校准 tracker 输入哈希及提分计数。全部必需作业、控制器和审计实际退出 0。原始 CSV 的 CRLF／填充保持字节不变。','',
        'ByteTrack 的建轨门槛是 track_thresh+0.1=0.7。本轮把合格的原始 (0.3,0.7] 分数改为 0.75，因此同时影响匹配阶段、代价、建轨和记忆置信。原始 detector 框／顺序／分数包另有不变收据；不能宣称 tracker 输入分数不变。','',
        'GPU4／5 的真实图像工程检查均完成：每卡 42 次 Train 前缀帧访问，16 配置在线与新鲜 FP32 证据重放 MOT 相同，原有 Train bank 检测误差 0、描述子 FP16 误差小于 1e-3。整个红外缺失时，视觉 head 严格退回同源新鲜 RGB，质量头退回 RGB；错位图像链路与序列重置检查通过。这些是小前缀工程检查，没有新增 HOTA 或完整序列 FPS 结论。','',
        '## 归因边界与后续','',
        'Dev 的五项消融涵盖质量描述子缺失／置换、确认关闭、无 archive 和无 shape rendering。描述子控制只作用于 ROI 质量证据，固定视觉检测器仍读取红外，不能代替整个模态的因果消融。Test 未达到预注册的 +5 触发条件，因此没有追加该轮五项完整 Test 消融。','',
        '下一方案在读取本轮完整 Test 之前已记录：只给当前合格且未匹配的 (0.6,0.7) 高分候选增加建轨资格，原分数和匹配公式保持原样。新增轨迹会改变以后候选集合与竞争，不能宣称以后所有匹配结果都不变。它仍需完整 Train 机会诊断、Dev 选择和冻结 Test，尚无成功结论。','',
        '历史 Test 已反复暴露，原始 detector 见过 Dev，ReID encoder 见过 Train 身份；这些结果不是新盲测。新版伪红外从 RGB 派生，不能宣称独立热传感器信息。先验组件为线性判别、logistic、ByteTrack 和时间确认，组合本身不能自动获得创新性。','',
        f'全部 47 个聚合／逐序列字段见 [{NAME}.json]({NAME}.json)，原始官方全阈值 CSV、配置对应指标、完整作业记录和来源收据位于 [{DEST.name}/]({DEST.name}/)。模型参数、归一化向量、图像、NPZ、crop、MOT 预测均不上传。','',
        '复现顺序：`fit_mft_proposal_quality.py engineer/register/fit` → `audit_mft_proposal_quality_fit.py` → `run_mft_proposal_quality.py register/run` → `audit_mft_proposal_quality_results.py`。需按对应配置提供现有私有权重与已核验数据，不能仅凭本仓库下载后无资产运行。']
    md=DOC/(NAME+'.md');md.write_text('\n'.join(lines)+'\n');files.append(str(md.relative_to(ROOT)))
    index=DEST/'byte_exact_artifact_index.json';write(index,dict(UTC=utc(),copies=copies,raw_csv_newlines_preserved=True));files.append(str(index.relative_to(ROOT)))
    paths=[ROOT/n for n in files];assert max(p.stat().st_size for p in paths)<512*1024
    manifest=WORK/'proposal_quality_public_export_20261127.json'
    write(manifest,dict(UTC=utc(),files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in paths},
        total_bytes=sum(p.stat().st_size for p in paths),max_bytes=max(p.stat().st_size for p in paths),source='tools/export_mft_proposal_quality.py'))
    print(json.dumps(dict(status='EXPORTED_AUDITED_LIGHTWEIGHT_QUALITY_EVIDENCE',files=len(paths),total_bytes=sum(p.stat().st_size for p in paths),manifest=str(manifest))))


if __name__=='__main__':main()
