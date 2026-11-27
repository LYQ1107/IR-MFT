#!/usr/bin/env python3
"""Publish the audited negative12case conditional-birth full-Dev study."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
WORK=Path('/var/tmp/liuyeqiang_ir_mft_perceptual_20261122')
DOC=ROOT/'docs/fusion_research'
DEST=DOC/'perceptual_20261127_birth_conditional_Dev'
NAME='PERCEPTUAL_MFT_20261127_BIRTH_CONDITIONAL_DEV'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def main():
    base=WORK/'birth_conditioned_evaluation_20261127';dev=read(base/'complete_Dev.json');audit=read(base/'independent_results_audit.json')
    for n in ['run_r1','audit_r1']:
        e=read(base/(n+'.execution.json'));assert e['actual_child_exit_code']==0 and e['source_unchanged']
        for path,h in e['source_hashes'].items():assert sha(ROOT/path)==h
    assert read(base/'state.json')['status']=='COMPLETE_BIRTH_CONDITIONED_NO_DEV_GAIN_NO_NEW_TEST'
    assert dev['gain_over_parent_best_Dev_pp']<=0 and not(base/'Test_selection_frozen.json').exists()and not(base/'full_Test_results.json').exists()
    assert audit['official_fields_reparsed']==1692
    assert not DEST.exists()and not(DOC/(NAME+'.json')).exists()and not(DOC/(NAME+'.md')).exists()
    copies={};files=[]
    def copy(source,relative):
        source=Path(source);target=DEST/relative;assert source.is_file()and source.stat().st_size<512*1024
        assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);assert sha(target)==sha(source)
        n=str(target.relative_to(ROOT));copies[n]=dict(original_path=str(source),sha256=sha(source),bytes=source.stat().st_size,byte_exact=True);files.append(n)
    for n in ['registered.json','real_Train_engineering.json','state.json','run_r1.execution.json','audit_r1.execution.json','audit_r1.stdout.txt','independent_results_audit.json']:
        copy(base/n,n)
    for p in sorted(base.glob('*.supervisor.py')):copy(p,p.name)
    manifests={}
    for p in sorted((base/'streams/dev/sequence_manifests').glob('*.json')):
        manifests[p.stem]=read(p);copy(p,Path('sequence_manifests')/p.name)
    rows=[]
    for r in dev['rows']:
        name=r['case']['id'];folder=base/'evaluation/dev'/name
        for n in ['metrics.json','trackeval_output/pedestrian_detailed.csv','trackeval_output/pedestrian_summary.txt']:copy(folder/n,Path('cases')/name/n)
        rows.append(dict(case=r['case'],combined47=r['combined47'],per_sequence47=r['per_sequence47'],frames=r['frames'],
            actual_extra_births=sum(m['statistics'][name]['extra_qualified_births']for m in manifests.values()),
            qualified_current_proposals=sum(m['statistics'][name]['qualified_current_proposals']for m in manifests.values()),official_CSV_sha256=r['official_CSV_sha256']))
    for phase,jobs in read(base/'state.json')['phases'].items():
        for n,j in jobs.items():assert j['status']=='COMPLETE'and j['exit_code']==0;copy(j['log'],Path('worker_logs')/Path(j['log']).name)
    fitted=[r for r in rows if r['case']['kind']in ['RGB','paired','duplicatedRGB']]
    best=min(fitted,key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),r['case']['id']))
    summary=dict(status='AUDITED_ALL12_FULL_DEV_CONDITIONAL_BIRTH_NEGATIVE_NO_NEW_TEST',rows=rows,
        best_new_fitted_case=best['case'],best_new_fitted_Dev_HOTA=best['combined47']['HOTA'],
        previous_best_Dev_HOTA=dev['parent_best_Dev_HOTA'],best_new_minus_previous_Dev_pp=best['combined47']['HOTA']-dev['parent_best_Dev_HOTA'],
        all12_best_case=dev['chosen_case'],all12_best_minus_previous_Dev_pp=dev['gain_over_parent_best_Dev_pp'],
        frames=4834,sequences=2,official_fields_independently_reparsed=1692,all_qualification_mask_hashes_and_counts_rebuilt=True,
        original_scores_boxes_order_unchanged=True,actual_extra_birth_events_not_independently_replayed=True,
        all_required_jobs_controller_and_audit_actual_exit0=True,new_Test=False,previous_formal_Test_method_not_changed=True,
        latest_formal_Test_HOTA=30.717081066539237,original_RGB_Test_HOTA=29.528970687393972,formal_gain_pp=1.1881103791452645,
        minimum_Test_HOTA=34.52897068739397,goal_status='active',goal_met=False,supported_innovations=0)
    p=DOC/(NAME+'.json');write(p,summary);files.append(str(p.relative_to(ROOT)))
    lines=['# 条件建轨质量头：完整12项 Dev 负结果','',
        f'新拟合条件模型中，最好的 `{best["case"]["id"]}` 在全部 4,834 个 Dev 帧上为 **{best["combined47"]["HOTA"]:.6f} HOTA**，低于上一轮固定 Dev **{dev["parent_best_Dev_HOTA"]:.6f}**，变化 **{best["combined47"]["HOTA"]-dev["parent_best_Dev_HOTA"]:+.6f}**。包含参考的全部12项里，最好的是已有 archive 参考52.098427，也未超过原52.113114。未通过预注册门槛，因此没有新 Test 或冻结 Test 选择。','',
        '上一轮正式方法继续保留，完整 Test7 为30.717081，相对原始 RGB YOLOX-X+ByteTrack29.528971提高1.188110，仍未达到+5目标34.528971。本轮 Train 校准中 IR模型82对RGB30的优势，没有转化成完整 Dev 净增益。不能据此算作成功创新。','',
        '| 配置 | HOTA | DetA | AssA | IDF1 | 实际额外建轨数 |','|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('| '+r['case']['id']+' | '+' | '.join(f'{r["combined47"][k]:.6f}'for k in ['HOTA','DetA','AssA','IDF1'])+f' | {r["actual_extra_births"]} |')
    lines+=['','宽松规则仍会产生有害建轨。严格配对 q999 虽优于相同 q999 的 RGB 质量头，却也低于已有安全配置，不能把相对退化控制的差值当作整体提升。后续应先从 Train 检查“当前是否需要额外建轨”，再考虑新策略；当前研究的严格选择规则不作事后放宽。','',
        '全 Train 27,251帧拟合与独立拟合审计已在上一阶段交付。当前15个作业为一次六序列工程检查、两段完整 Dev 重放、12次官方评估，控制器及独立结果审计均实际退出0。独立审计重解析1,692个官方字段，逐帧重建资格掩码哈希与计数，检查原始输入包和两个参考 MOT。没有重跑实际额外建轨事件级反事实审计。','',
        '原始 CSV 保留全阈值字段与 CRLF／填充，不重新格式化。47个聚合及逐序列指标均保留。未知不作背景，推理只读当前和过去；Train FP16／Dev FP32描述子、历史Test暴露、旧detector见过Dev、ReID见过Train身份的边界不变。','',
        f'完整数值见 [{NAME}.json]({NAME}.json)，原始官方 CSV 与退出／来源收据见 [{DEST.name}/]({DEST.name}/)。没有上传模型、归一化向量、图像、NPZ、crop、patch tokens 或原始MOT。goal保持active，创新点暂计0。']
    p=DOC/(NAME+'.md');p.write_text('\n'.join(lines)+'\n');files.append(str(p.relative_to(ROOT)))
    p=DEST/'byte_exact_artifact_index.json';write(p,dict(copies=copies));files.append(str(p.relative_to(ROOT)))
    paths=[ROOT/n for n in files];assert max(p.stat().st_size for p in paths)<512*1024
    write(WORK/'birth_conditional_Dev_public_export_20261127.json',dict(files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in paths},
        total_bytes=sum(p.stat().st_size for p in paths),max_file_bytes=max(p.stat().st_size for p in paths)))
    print(json.dumps(dict(status='EXPORTED_AUDITED_CONDITIONAL_DEV_NEGATIVE',files=len(files),total_bytes=sum(p.stat().st_size for p in paths))))


if __name__=='__main__':main()
