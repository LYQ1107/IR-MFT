#!/usr/bin/env python3
"""Publish audited birth-only science without weights, features or MOT files."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
WORK=Path('/var/tmp/liuyeqiang_ir_mft_perceptual_20261122')
DOC=ROOT/'docs/fusion_research'
DEST=DOC/'perceptual_20261127_birth_only'
NAME='PERCEPTUAL_MFT_20261127_BIRTH_ONLY'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
    p=Path(p);assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def main():
    train=WORK/'birth_only_20261127';base=WORK/'birth_only_evaluation_20261127'
    for folder,name in [(train,'engineering_r1'),(train,'diagnose_r1'),(base,'run_r1'),(base,'audit_r1')]:
        e=read(folder/(name+'.execution.json'));assert e['actual_child_exit_code']==0 and e['source_unchanged']
    audit=read(base/'independent_results_audit.json');assert audit['all_birth_qualification_masks_hashes_and_counts_rebuilt']
    assert not DEST.exists()and not(DOC/(NAME+'.json')).exists()and not(DOC/(NAME+'.md')).exists()
    copies={};files=[]
    def copy(source,relative):
        source=Path(source);target=DEST/relative;assert source.is_file()and source.stat().st_size<512*1024
        assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        assert sha(target)==sha(source)
        n=str(target.relative_to(ROOT));copies[n]=dict(original_path=str(source),sha256=sha(source),bytes=source.stat().st_size,byte_exact=True);files.append(n)
    for label,folder,names in [
        ('Train',train,['CPU_regressions.json','real_Train_engineering.json','engineering_r1.execution.json',
            'engineering_r1.stdout.txt','registered.json','complete_Train_diagnosis.json','diagnose_r1.execution.json','diagnose_r1.stdout.txt']),
        ('evaluation',base,['registered.json','state.json','real_Train_engineering.json','run_r1.execution.json',
            'audit_r1.execution.json','independent_results_audit.json','audit_r1.stdout.txt'])]:
        for n in names:copy(folder/n,Path(label)/n)
        for p in sorted(folder.glob('*.supervisor.py')):copy(p,Path(label)/p.name)
    heading=WORK/'body_heading_probe_20261127'
    he=read(heading/'run_r1.execution.json');assert he['actual_child_exit_code']==0 and he['source_unchanged']
    hd=read(heading/'complete.json')
    for n in ['registered.json','complete.json','run_r1.execution.json','run_r1.stdout.txt','run_r1.supervisor.py']:
        copy(heading/n,Path('heading_feature_diagnostic')/n)
    groups=[('dev','complete_Dev.json')]
    for label,n in [('dev_intervention','complete_Dev_interventions.json'),('test','full_Test_results.json'),('test_intervention','full_Test_interventions.json')]:
        if (base/n).exists():groups.append((label,n))
    all_rows={}
    for label,n in groups:
        result=read(base/n);all_rows[label]=[];manifests={}
        for p in sorted((base/'streams'/label/'sequence_manifests').glob('*.json')):
            manifests[p.stem]=read(p);copy(p,Path('sequence_manifests')/label/p.name)
        for row in result['rows']:
            name=row['case']['id'];folder=base/'evaluation'/label/name
            for filename in ['metrics.json','trackeval_output/pedestrian_detailed.csv','trackeval_output/pedestrian_summary.txt']:
                copy(folder/filename,Path('evaluation')/label/name/filename)
            extra=sum(m['statistics'][name]['extra_qualified_births']for m in manifests.values())
            qualified=sum(m['statistics'][name]['qualified_current_proposals']for m in manifests.values())
            all_rows[label].append(dict(case=row['case'],frames=row['frames'],combined47=row['combined47'],per_sequence47=row['per_sequence47'],
                extra_births_in_tracker_receipts=extra,qualified_current_proposals=qualified,
                original_metrics_sha256=sha(folder/'metrics.json'),official_CSV_sha256=row['official_CSV_sha256']))
    for phase,jobs in read(base/'state.json')['phases'].items():
        for name,job in jobs.items():
            assert job['status']=='COMPLETE'and job['exit_code']==0
            copy(job['log'],Path('worker_logs')/Path(job['log']).name)
    for n in ['Test_selection_frozen.json']:
        if (base/n).exists():copy(base/n,Path('evaluation')/n)
    diagnosis=read(train/'complete_Train_diagnosis.json');cases=read(train/'registered.json')['cases'];counts={}
    for c in cases:
        if c['kind'].startswith('reference_'):continue
        total={};by_phase={}
        for phase in ['early','late']:
            part={}
            for row in diagnosis['coverage'].values():
                for k,v in row['by_case'][phase][c['id']].items():part[k]=part.get(k,0)+v;total[k]=total.get(k,0)+v
            by_phase[phase]=part
        counts[c['id']]=dict(total=total,by_phase=by_phase)
    dev=read(base/'complete_Dev.json');test=read(base/'full_Test_results.json')if(base/'full_Test_results.json').exists()else None
    previous=read(WORK/'proposal_quality_evaluation_20261127/full_Test_results.json')
    baseline=previous['original_RGB_baseline']['combined47']['HOTA'];delta=None
    if test:
        selected=next(r for r in all_rows['test']if r['case']==test['selected_case'])
        delta=selected['combined47']['HOTA']-baseline;assert delta==test['HOTA_gain_over_original_RGB_pp']
    matched_quality={r['case']['kind']:r for r in all_rows.get('test',[])if r['case']['kind']in ['RGB','paired','duplicatedRGB']}
    summary=dict(status='AUDITED_BIRTH_ONLY_COMPLETE_DEV_WITH_CONDITIONAL_FROZEN_TEST',UTC=utc(),
        original_RGB_baseline_HOTA=baseline,minimum_Test_HOTA=baseline+5,stretch_Test_HOTA=baseline+10,
        selected_from_Dev=dev['chosen_case'],gain_over_fixed_parent_Dev_pp=dev['gain_over_parent_best_Dev_pp'],
        new_Test_performed=test is not None,new_selected_Test_HOTA=None if not test else selected['combined47']['HOTA'],
        new_Test_HOTA_gain_pp=delta,all_official_rows=all_rows,Train_opportunity_counts=counts,
        paired_minus_matched_RGB_quality_Test_pp=None if not test else matched_quality['paired']['combined47']['HOTA']-matched_quality['RGB']['combined47']['HOTA'],
        Test_controls_not_promoted_to_formal_method=True,
        Train_frames=27251,Train_proposals=249639,zero_gate_all_Train_MOT_exact_original=True,
        zero_gate_full_state_every100_frames_and_end_exact_original=True,
        original_scores_boxes_and_order_unchanged=True,source_method_port_only_birth_check_changed=True,
        new_tracks_can_change_future_candidate_graph_and_associations=True,
        qualification_masks_and_official_fields_independently_rebuilt=True,
        official_fields_reparsed=audit['official_fields_reparsed'],actual_extra_birth_events_not_independently_replayed=True,
        Train_opportunities_are_repeated_current_frame_events_not_unique_fish_or_counterfactual_HOTA=True,
        unknown_labels_not_background=True,new_optimizer_updates=0,new_NN_training=0,seed=0,
        no_new_real_image_or_FPS_claim_for_this_tracker=True,quality_controls_ROI_only_visual_IR_stays=True,
        strongest_RGB_sweeps_deferred=True,historical_Test_previously_exposed=True,
        cached_30Train_body_heading_diagnostic=dict(primary=hd['preregistered_primary'],pilot_gate_passed=hd['pilot_gate_passed'],
            no_association_or_accuracy_claim=True,oracle_GT_motion_targets=True,new_image_features=False),
        original_RGB_detector_saw_Dev=True,ReID_encoder_saw_Train_identities=True,
        public_model_weights_vectors_images_features_or_MOT=False,goal_met=False,supported_innovations=0,goal_status='active')
    report=DOC/(NAME+'.json');write(report,summary);files.append(str(report.relative_to(ROOT)))
    lines=['# Birth-only qualification：保留原始分数的建轨修正','',
        '本轮将“是否允许建轨”从检测分数中分离。原始 ByteTrack 文件保持不变；另一个可读方法副本只改 Step4 的资格判断。只允许原始分数在 (0.6,0.7) 且合格、在原有匹配步骤后仍未使用的高分候选建轨。候选分数、框、顺序、匹配代价公式、模板置信条件保持原样；新增轨迹仍会改变以后的候选集合与竞争。','',
        f'完整 Dev 共 4,834 帧、16 个预先固定配置，按 HOTA／AssA／IDF1／ID 排序，选择 `{dev["chosen_case"]["id"]}`，相对固定父配置的 Dev 变化为 **{dev["gain_over_parent_best_Dev_pp"]:+.6f}** HOTA。','']
    if test:
        lines.append(f'该冻结配置完整 Test7／15,985 帧 HOTA 为 **{selected["combined47"]["HOTA"]:.6f}**，相对原始 RGB YOLOX-X + ByteTrack **{baseline:.6f}** 的增益为 **{delta:+.6f}** 个绝对点。Test 控制不参与改选；科学归因及 +5／+10 目标是否完成仍需依据全部证据，当前 goal 保持 active。')
        lines.append('表中的 RGB 质量控制仍使用相同的新版伪红外视觉检测器，只去掉建轨质量头的 IR 描述子，不是另一个原始 RGB baseline。无条件建轨控制即使在 Test 更高，也因完整 Dev 退化而不被提升为正式选择。')
    else:
        lines.append('完整 Dev 未通过预注册的严格正向门槛，因此本轮没有启动新 Test，也没有本轮 Test 增益。原始 RGB Test 基线为 29.528971，+5 目标为 34.528971。上一轮提分方案的 Test30.474359／+0.945388 已单独记录，不能作为本轮成绩。')
    for label,title in [('dev','完整 Dev'),('dev_intervention','完整 Dev 干预'),('test','冻结完整 Test7'),('test_intervention','冻结完整 Test7 消融')]:
        if label not in all_rows:continue
        lines+=['','## '+title,'','| 配置 | HOTA | DetA | AssA | IDF1 | 实际额外建轨数 | 合格候选帧数 |','|---|---:|---:|---:|---:|---:|---:|']
        for row in all_rows[label]:
            lines.append('| '+row['case']['id']+' | '+' | '.join(f'{row["combined47"][k]:.6f}'for k in ['HOTA','DetA','AssA','IDF1'])+f' | {row["extra_births_in_tracker_receipts"]} | {row["qualified_current_proposals"]} |')
    lines+=['','## 完整 Train 机会诊断','',
        '27,251 帧、249,639 proposals 全部交付两个独立实例。关闭资格修正时，每帧 MOT 与原始教师相同，每 100 帧及每段结束时的 tracked／lost／removed、身份、状态、Kalman mean／covariance、模板和置信状态均逐项相同。当前 GT 标签只在两个推理路径完成后用于统计，且与此前已核验 Train 标签包哈希相同。','',
        '| 资格规则 | 已知正例机会 | 已知背景机会 | 未知机会 | 当前原版输出未局部覆盖的已知正例机会 |','|---|---:|---:|---:|---:|']
    for name in ['BO_unconditional_c0','BO_unconditional_c3','BO_RGB_q99_c3','BO_paired_q99_c3','BO_duplicatedRGB_q99_c3','BO_RGB_q999_c3','BO_paired_q999_c3']:
        c=counts[name]['total'];lines.append(f'| {name} | {c["known_positive"]} | {c["known_background"]} | {c["unknown"]} | {c["known_positive_uncovered_by_zero_output_local_IoU05"]} |')
    lines+=['','这些机会是重复的当前帧事件，不是独立鱼数，不是反事实跟踪召回，也不是 HOTA 的收益上界。零资格路径没有实际创建这些额外轨迹；后来的资格修正会改变未来竞争，因此不能把机会总数直接算成召回提升。未知始终不作背景。','',
        '质量模型和阈值复用上一轮已收敛的全 Train 拟合，没有新增 optimizer 更新、CNN 训练或随机种子。旧阈值在 (0.3,0.7] 人群上校准，本轮只复用到 (0.6,0.7)，这一人群变化明确保留，不能宣称重新校准了跟踪风险。','',
        f'独立审计重新解析 {audit["official_fields_reparsed"]} 个官方字段，逐帧从原始 FP32 Dev／Test bank 重建确认历史、资格布尔掩码、掩码哈希和合格数量，检查原始输入包与 source bank 相同。实际额外建轨事件数来自原跟踪器收据，没有另行重跑事件级反事实审计。原始官方 CSV 的换行和数值字节不改写。','',
        '另复核了此前已计算的 30 个固定 Train 帧的身体主轴，测试不同于旧框校正任务的“当前轴是否与真实运动方向一致”。主指标只用预设 lag3、双模态严格共同有效且 GT 位移至少 1 像素的样本。共有 '+str(hd['preregistered_primary']['rows'])+' 个样本、'+str(hd['preregistered_primary']['videos'])+' 段视频；等视频 IR-minus-gray 倍角方向点积为 '+f'{hd["preregistered_primary"]["equal_video_IR_minus_gray"]:+.6f}'+'，正向视频 '+str(hd['preregistered_primary']['positive_videos'])+'。IR 相对同帧轴置换控制虽为正，但不优于 RGB 灰度，因此预注册探索门槛未通过，不升级为跟踪模块。该分析使用 GT 运动作为 oracle 标签，仅为特征诊断，不证明身份区分、关联或 HOTA；没有新增图像特征或 Test。','',
        '本轮 ROI 红外置换／缺失控制只检验质量证据，视觉检测器仍使用固定新版伪红外。关闭模块的完整 Train 与参考输出核验为 CPU 缓存重放，本轮不新增真实图像准确率或 FPS 结论。历史 Test 暴露、原始 detector 见过 Dev、ReID 见过 Train 身份和 Train FP16／Dev-Test FP32 缓存限制均继续适用。','',
        '目前不为这个普通建轨门控、线性判别／logistic 和时间确认组合赋予多模态创新点。只有独立消融和匹配模态控制支持后才能计入，+5／+10 目标没有预设成功。最强 RGB／跨结构搜索继续按用户最新指令延期。','',
        f'全部 47 个聚合和逐序列字段见 [{NAME}.json]({NAME}.json)，原始官方 CSV、轻量指标、过程退出码及来源收据见 [{DEST.name}/]({DEST.name}/)。不上传图像、特征、权重、归一化向量或原始 MOT 预测。','',
        '复现：`diagnose_mft_birth_only.py engineer/register/diagnose` → `run_mft_birth_only.py register/run` → `audit_mft_birth_only_results.py`。使用固定已核验权重和缓存，按配置保证 GPU4／5 最多两卡、seed0、工作目录64GiB和40GiB空闲预留。']
    md=DOC/(NAME+'.md');md.write_text('\n'.join(lines)+'\n');files.append(str(md.relative_to(ROOT)))
    index=DEST/'byte_exact_artifact_index.json';write(index,dict(UTC=utc(),copies=copies));files.append(str(index.relative_to(ROOT)))
    paths=[ROOT/n for n in files];assert max(p.stat().st_size for p in paths)<512*1024
    manifest=WORK/'birth_only_public_export_20261127.json'
    write(manifest,dict(UTC=utc(),files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in paths},
        total_bytes=sum(p.stat().st_size for p in paths),max_file_bytes=max(p.stat().st_size for p in paths)))
    print(json.dumps(dict(status='EXPORTED_AUDITED_BIRTH_ONLY_EVIDENCE',files=len(files),total_bytes=sum(p.stat().st_size for p in paths),manifest=str(manifest))))


if __name__=='__main__':main()
