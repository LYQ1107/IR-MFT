#!/usr/bin/env python3
"""Export completed conditional Train fit/audit without learned parameters."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
WORK=Path('/var/tmp/liuyeqiang_ir_mft_perceptual_20261122')
DOC=ROOT/'docs/fusion_research'
DEST=DOC/'perceptual_20261127_birth_conditional_Train'
NAME='PERCEPTUAL_MFT_20261127_BIRTH_CONDITIONAL_TRAIN'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def main():
    base=WORK/'birth_conditioned_quality_20261127';evaluation=WORK/'birth_conditioned_evaluation_20261127'
    for name in ['fit_r1','audit_r1']:
        e=read(base/(name+'.execution.json'));assert e['actual_child_exit_code']==0 and e['source_unchanged']
        for path,h in e['source_hashes'].items():assert sha(ROOT/path)==h
    f=read(base/'fit_complete.json');audit=read(base/'independent_fit_audit.json')
    assert f['gradient_converged']and audit['model_sha256']==f['model_sha256']
    assert not DEST.exists()and not(DOC/(NAME+'.json')).exists()and not(DOC/(NAME+'.md')).exists()
    copies={};files=[]
    def copy(source,relative):
        source=Path(source);target=DEST/relative;assert source.is_file()and source.stat().st_size<512*1024
        assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);assert sha(target)==sha(source)
        n=str(target.relative_to(ROOT));copies[n]=dict(original_path=str(source),bytes=source.stat().st_size,sha256=sha(source),byte_exact=True);files.append(n)
    for n in ['registered.json','Train_coverage.json','fit_complete.json','fit_history.json','fit_r1.execution.json','fit_r1.stdout.txt',
        'independent_fit_audit_registered.json','independent_fit_audit.json','audit_r1.execution.json','audit_r1.stdout.txt']:
        copy(base/n,n)
    for p in sorted(base.glob('*.supervisor.py')):copy(p,p.name)
    copy(evaluation/'registered.json','prospective_Dev_registration.json')
    summary=dict(status='COMPLETE_AUDITED_CONDITIONAL_BIRTH_TRAIN_FIT_NO_NEW_ACCURACY_CLAIM',
        Train_frames=27251,all_proposals=249639,conditional_rows=18873,known_positive=13158,known_background=885,unknown=4830,
        early_known_fit_rows=f['early_known_fit_rows'],Newton_updates=f['Newton_updates'],final_diagnostics=f['final_diagnostics'],
        late_calibration=f['late_calibration'],model_sha256=f['model_sha256'],parent_foreground_views_exact=True,
        independent_all_Train_fit_audit_actual_exit0=True,all_feature_streak_label_and_zero_gate_MOT_packets_exact=True,
        original_RGB_baseline_Test_HOTA=29.528970687393972,minimum_Test_HOTA=34.52897068739397,
        latest_completed_formal_Test_HOTA=30.717081066539237,latest_completed_formal_gain_pp=1.1881103791452645,
        planned_Dev_cases=12,strict_Dev_gate=52.11311409153856,conditional_method_new_HOTA=None,
        unknown_not_background=True,new_NN_training_or_seeds=0,private_parameters_and_features_uploaded=False,
        goal_status='active',goal_met=False,supported_innovations=0)
    p=DOC/(NAME+'.json');write(p,summary);files.append(str(p.relative_to(ROOT)))
    lines=['# 条件建轨质量头：全 Train 拟合与独立复核','',
        '本阶段只改变质量头拟合／校准的人群：从所有 proposals 改为原始教师在现有匹配后仍未使用、原分数在 (0.6,0.7) 的建轨机会。检测器、原始分数、512D 前景方向、11 个特征、确认规则、原始跟踪流程和输出渲染保持原样。新增轨迹依然可能改变后续候选竞争，离线阈值不构成闭环安全保证。','',
        '六段 Train 全部 27,251 帧、249,639 proposals 参与原始教师和独立确认流。18,873 个候选事件包括 13,158 个已知正例、885 个已知背景和 4,830 个未知。当前标签在无 GT 特征与教师处理后才加入；未知只从监督 loss／校准中排除，不变成背景，也不从推理图中删掉。','',
        f'每段前 75% 中共有 {f["early_known_fit_rows"]:,} 个已知拟合行，采用等视频／等可用类别权重。三个头都在 {f["Newton_updates"]} 次确定性 Newton 更新后收敛，梯度范数约 1e-16。预先允许的优化扩展未触发，没有新增 CNN 训练或随机种子。这个 loss 的数据人群变了，不能与上一轮全 proposals 的 loss 直接比较成性能改善。','',
        '独立审计再次重放全部 Train，重建条件特征、确认、标签和零资格 MOT 哈希；单独计算标准化统计、加权 logistic 目标、梯度和 Hessian，并用排序插值核对所有校准分位数。模型及归一化参数没有被审计修改。拟合与审计实际退出均为 0。','',
        '## 后 25% 校准，确认至少三帧','',
        '| 头 | 分位数 | 已知正例通过数 | 已知背景通过数 |','|---|---|---:|---:|']
    for kind,values in f['late_calibration'].items():
        for q,x in values['confirmed']['operating_points'].items():
            lines.append(f'| {kind} | {q} | {x["known_positive_passing"]} | {x["known_background_passing"]} |')
    lines+=['','校准共有 3,203 个已知正例和 209 个已知背景。以上是当前帧候选事件，可能反复来自同一条鱼，不能当作独立鱼数、召回或 HOTA。分位数是这组已知背景上的经验分数，不是跟踪错误概率界。','',
        '12 个新 Dev 配置在评估前冻结：原始 native 与 C3 archive 两个参考、无条件 c3、RGB／配对／重复 RGB 三种头各三个分位数。必须完整 4,834 帧严格超过上一轮 Dev52.113114，才能冻结新 Test；完整 Test 前还要求全 Train 审计实际退出 0。Test 对照不能被直接提升为正式选择。','',
        '本文件只交付已经完成的 Train 拟合／审计和不可变 Dev 注册，没有该条件方法的新 HOTA。最新已经完成的正式 Test 为30.717081／+1.188110，距离原始 RGB baseline29.528971之上的 +5 目标34.528971仍未达标；goal 保持 active，多模态创新点暂计0。','',
        f'全部诊断与来源见 [{NAME}.json]({NAME}.json) 及 [{DEST.name}/]({DEST.name}/)。学习到的方向、中心、系数、归一化向量、NPZ、权重和原始 MOT 不上传。']
    p=DOC/(NAME+'.md');p.write_text('\n'.join(lines)+'\n');files.append(str(p.relative_to(ROOT)))
    p=DEST/'byte_exact_artifact_index.json';write(p,dict(copies=copies));files.append(str(p.relative_to(ROOT)))
    paths=[ROOT/n for n in files];assert max(p.stat().st_size for p in paths)<512*1024
    write(WORK/'birth_conditional_Train_public_export_20261127.json',dict(files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in paths},
        total_bytes=sum(p.stat().st_size for p in paths),max_file_bytes=max(p.stat().st_size for p in paths)))
    print(json.dumps(dict(status='EXPORTED_CONDITIONAL_TRAIN_FIT_AND_AUDIT',files=len(files),total_bytes=sum(p.stat().st_size for p in paths))))


if __name__=='__main__':main()
