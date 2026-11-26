#!/usr/bin/env python3
"""Independent all-Train/official-CSV audit after the original risk run closes."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.fit_mft_candidate_risk import locations,require_planning,threshold_values
from tools.run_mft_candidate_risk import require_fitted,policies,identity_score_hash
from tools.mft_candidate_risk_data import qualify_Train
from tools.mft_candidate_risk_fit_audit import audit_full_Train
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.run_mft_candidate_train_bank import atomic_json
from tools.run_mft_native_memory_priority_v3 import identity
from tools.run_pseudo_ir_comparison import sha256
from tools.export_perceptual_metrics47 import official_schema
from tools.fusion_memory_evaluation import official_csv47,validate_mot
from tools.fusion_memory_observation import LENGTHS


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def stamp(value):return datetime.datetime.fromisoformat(value)


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=require_planning();plan,protocol,work,base=locations()
    proof=read(base/'independent_audit_CPU_regressions_r1.json')
    assert proof['actual_tool_exit_code']==0 and proof['tests_passed']==2
    names=['tools/audit_mft_candidate_risk.py','tools/mft_candidate_risk_fit_audit.py',
        'tests/test_mft_candidate_risk_fit_audit.py']
    for name,digest in proof['source_hashes'].items():assert sha256(ROOT/name)==digest,name
    assert set(names)==set(proof['source_hashes'])
    execution=read(base/'controller.execution.json')
    target=base/'independent_audit_registered.json';assert not target.exists()
    frozen_json(target,dict(status='REGISTERED_READ_ONLY_INDEPENDENT_TRAIN_AND_FULL_METRIC_AUDIT',UTC=utc(),
        source_hashes={n:sha256(ROOT/n)for n in names},
        original_pipeline_registration_sha256=sha256(base/'pipeline_registered.json'),
        original_controller_identity=execution['identity'],
        CPU_proof_sha256=sha256(base/'independent_audit_CPU_regressions_r1.json'),
        wait_for_original_actual_exit_zero=True,rebuild_all6_Train_once=True,
        perform_new_optimizer_steps=False,change_model_or_selection=False,
        read_new_Test_for_Dev_selection=False,coefficient_publication=False,new_GPU_work=False))
    print(json.dumps(dict(status='REGISTERED_READ_ONLY_INDEPENDENT_TRAIN_AND_FULL_METRIC_AUDIT',sources=3)),flush=True)


def registered():
    require_planning();plan,protocol,work,base=locations()
    value=read(base/'independent_audit_registered.json')
    for name,digest in value['source_hashes'].items():assert sha256(ROOT/name)==digest,name
    assert value['original_pipeline_registration_sha256']==sha256(base/'pipeline_registered.json')
    assert value['CPU_proof_sha256']==sha256(base/'independent_audit_CPU_regressions_r1.json')
    return value


def require_closed_jobs(state,names,expected_count):
    assert set(names)<=set(state['phases'])
    jobs=[j for name in names for j in state['phases'][name].values()]
    assert len(jobs)==expected_count
    for j in jobs:
        assert j['status']=='COMPLETE'and type(j.get('exit_code'))is int and j['exit_code']==0
        assert j['argv_identity_observed']and identity(j['pid'])is None
    return jobs


def wait_closed():
    reg=registered();plan,protocol,work,base=locations();expected=reg['original_controller_identity']
    while True:
        registered();guard_storage(work)
        execution=read(base/'controller.execution.json');assert execution['identity']==expected
        if execution['actual_child_exit_code']is not None:
            assert type(execution['actual_child_exit_code'])is int and execution['actual_child_exit_code']==0
            assert execution['source_unchanged']and identity(expected['PID'])is None
            return
        live=identity(expected['PID'])
        if live is not None:assert all(live[k]==expected[k]for k in ['PID','start_ticks','raw_argv_sha256'])
        atomic_json(base/'independent_audit_progress.json',dict(status='WAITING_ORIGINAL_RISK_CONTROLLER',
            UTC=utc(),original_controller_live=live,new_GPU_work=False))
        time.sleep(15)


def audit_split(split,source,reg,fit,train_audit,state,base,work,plan):
    assert split in ['dev','test'];cases=[r['case']for r in source['rows']]
    expected=policies()if split=='dev'else read(base/'Test_selection_frozen.json')['cases']
    assert cases==expected
    jobs=require_closed_jobs(state,['CANDIDATE_RISK_'+split.upper()+'_REPLAY',
        'CANDIDATE_RISK_'+split.upper()+'_OFFICIAL'],len(LENGTHS[split])+len(cases))
    schema,_=official_schema(plan['evaluation']['TrackEval_root']);references=[];fields=0
    for name in ['NP_source_canonical_RGB','NP_source_fresh_RGB','NP_temporal_ir_e06']:
        folder=work/'native_pyramid_evaluation'/split/name;csv=folder/'trackeval_output/pedestrian_detailed.csv'
        values=official_csv47(csv,schema,split,read(folder/'metrics.json')['trackeval'])
        if split=='test'and name=='NP_source_canonical_RGB':
            assert sha256(csv)=='c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
            assert abs(values['combined47']['HOTA']-29.528970687393972)<1e-10
        fields+=47*len(values['per_sequence47']);references.append(dict(id=name,**values))
    pose=read(work/'pose_continuity_20261127'/('complete_Dev.json'if split=='dev'else'full_Test_results.json'))
    prior=next(r for r in pose['rows']if r['case']['id']=='PC_shape_anchor_late')
    pose_values=official_csv47(prior['official_CSV'],schema,split,prior['metrics39'])
    assert pose_values['combined47']==prior['combined47']and pose_values['per_sequence47']==prior['per_sequence47']
    references.append(dict(id='PC_shape_anchor_late',**pose_values));fields+=47*len(pose_values['per_sequence47'])
    records={}
    for sequence,frames in LENGTHS[split].items():
        m=read(base/'streams'/split/'sequence_manifests'/(sequence+'.json'))
        assert m['cases']==cases and m['frames']==frames and m['source_versions']==reg['source_hashes']
        assert m['model_sha256']==fit['model_sha256']and m['only_current_frame_inputs']
        assert not m['GT_read']and not m['previous_outputs_rewritten']and m['compact_feature_cache_read']
        assert m['native_and_frozen_pose_control_MOT_exact']and m['future_suffix_prefix_MOT_exact']
        assert m['missing_IR_prefix_MOT_exact_RGB_model_fallback']
        old=read(work/'pose_continuity_20261127/streams'/split/'sequence_manifests'/(sequence+'.json'))
        for own,name in [('CR_native_zero','PC_native_zero'),('CR_pose_late_control','PC_shape_anchor_late')]:
            assert m['streams'][own]['track_sha256']==old['streams'][name]['track_sha256']
        if split=='dev':
            a=identity_score_hash(m['streams']['CR_paired_risk']['track_path'])
            b=identity_score_hash(m['streams']['CR_paired_risk_shape']['track_path'])
            assert a==b==m['shape_identity_score_order_proof']['filtered_hash']==m['shape_identity_score_order_proof']['current_shape_hash']
        records[sequence]=m
    rows=[]
    for r in source['rows']:
        csv=Path(r['official_CSV']);assert sha256(csv)==r['official_CSV_sha256']
        values=official_csv47(csv,schema,split,r['metrics39']);fields+=47*len(values['per_sequence47'])
        assert values['combined47']==r['combined47']and values['per_sequence47']==r['per_sequence47']
        assert r['frames']==sum(LENGTHS[split].values())and r['source_versions']==reg['source_hashes']
        assert r['model_sha256']==fit['model_sha256']
        coverage={}
        for sequence,m in records.items():
            s=m['streams'][r['case']['id']]
            assert sha256(s['track_path'])==s['track_sha256']and s['detection_packet_sha256']==m['bank']['packet_sha256']
            validate_mot(s['track_path'],LENGTHS[split][sequence],s['track_rows'])
            coverage[sequence]=dict(track_sha256=s['track_sha256'],track_rows=s['track_rows'],
                statistics=m['statistics'][r['case']['id']])
        rows.append(dict(case=r['case'],coverage=coverage,**values))
    assert rows[0]['combined47']==references[2]['combined47']and rows[0]['per_sequence47']==references[2]['per_sequence47']
    assert rows[1]['combined47']==references[3]['combined47']and rows[1]['per_sequence47']==references[3]['per_sequence47']
    assert fields==(1410 if split=='dev'else 2632)
    if split=='dev':
        assert source['Test_read_for_selection']is False
        chosen=min(rows,key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),r['case']['id']))
        assert chosen['case']==source['chosen_case']
        gain=chosen['combined47']['HOTA']-rows[1]['combined47']['HOTA']
        assert gain==source['gain_over_frozen_pose_control_pp']
        if gain>1e-8:
            freeze=read(base/'Test_selection_frozen.json')
            assert freeze['chosen_case']==chosen['case']and freeze['Test_read_for_selection']is False
            assert freeze['cases']==[policies()[0],policies()[1],chosen['case']]
            assert freeze['complete_Dev_sha256']==sha256(base/'complete_Dev.json')
            assert stamp(source['UTC'])<=stamp(freeze['UTC'])
            for j in state['phases']['CANDIDATE_RISK_TEST_REPLAY'].values():assert stamp(j['started_UTC'])>=stamp(freeze['UTC'])
        else:
            assert not(base/'Test_selection_frozen.json').exists()and not(base/'full_Test_results.json').exists()
        identity_delta=rows[3]['combined47']['HOTA']-rows[2]['combined47']['HOTA']
        details=dict(gain_over_frozen_pose_control_pp=gain,identity_IR_addition_same_policy_Dev_HOTA_delta_pp=identity_delta,
            whole_IR_contribution_established=False,Test_used_for_selection=False)
    else:
        freeze=read(base/'Test_selection_frozen.json');assert source['Test_used_for_selection']is False
        assert source['selected_case']==freeze['chosen_case']
        chosen=next(r for r in rows if r['case']==freeze['chosen_case'])
        delta=chosen['combined47']['HOTA']-references[0]['combined47']['HOTA']
        assert delta==source['HOTA_gain_over_original_RGB_pp']
        assert source['target_plus5_met']==(delta>=5)and source['target_plus10_met']==(delta>=10)
        details=dict(HOTA_gain_over_original_RGB_pp=delta,target_plus5_met=delta>=5,target_plus10_met=delta>=10,
            Test_used_for_selection=False,independent_Test_IR_addition_control_included=False)
    out=ROOT/'docs/fusion_research'/('perceptual_20261127_candidate_risk_full_'+split)
    assert not out.exists();out.mkdir();files=[]
    def copy(source,name):
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        assert dest.stat().st_size<1024**2
        files.append(dict(file=str(dest.relative_to(ROOT)),sha256=sha256(dest),bytes=dest.stat().st_size))
        return str(dest.relative_to(ROOT))
    copy(base/('complete_Dev.json'if split=='dev'else'full_Test_results.json'),'complete_result.json')
    copy(base/'pipeline_registered.json','pipeline_registered.json');copy(base/'fit_registered.json','fit_registered.json')
    copy(base/'fit_complete.json','fit_complete.json');copy(base/'fit_history.json','fit_history.json')
    copy(base/'CPU_real_bank_engineering.json','real_Train_engineering.json')
    copy(base/'controller.execution.json','controller_execution.json');copy(base/'state.json','state.json')
    copy(base/'independent_Train_fit_audit.json','independent_Train_fit_audit.json')
    if(base/'Test_selection_frozen.json').exists():copy(base/'Test_selection_frozen.json','Test_selection_frozen.json')
    for sequence in LENGTHS[split]:copy(base/'streams'/split/'sequence_manifests'/(sequence+'.json'),'manifests/'+sequence+'.json')
    for row in [*rows,*references]:
        name=row.get('id',row.get('case',{}).get('id'));original=row['official_CSV']
        row['official_CSV_source']=original;row['official_CSV']=copy(original,'official/'+name+'.csv')
    result=dict(status='INDEPENDENTLY_VERIFIED_CANDIDATE_RISK_FULL_'+split.upper(),UTC=utc(),
        split=split,frames=sum(LENGTHS[split].values()),rows=rows,references=references,chosen_case=chosen['case'],
        official_numeric_fields_reparsed=fields,actual_split_worker_exits_zero=len(jobs),
        Train_fit_audit_sha256=sha256(base/'independent_Train_fit_audit.json'),
        Train_convergence=train_audit['convergence'],auditor_source_sha256=sha256(Path(__file__)),
        historical_Test_previously_exposed=True,new_blind_Test_claim=False,
        cached_CPU_replay_is_not_end_to_end_FPS=True,coefficient_publication=False,
        fit_history_raw_status_scope='Last optimizer-loop snapshot retains RUNNING; closed controller exit0 and fit_complete.json determine actual completion',
        goal_met=False,supported_innovations=0,**details,lightweight_files=files)
    report=ROOT/'docs/fusion_research'/('PERCEPTUAL_MFT_20261127_CANDIDATE_RISK_FULL_'+split.upper()+'.json')
    assert not report.exists();frozen_json(report,result)
    text=['# 候选关联风险学习：完整'+split.upper()+'与独立审计','',
        '固定原始 RGB YOLOX-X+ByteTrack 基线、新版十一月伪红外和 ProtocolV2。完整 '+str(result['frames'])+' 帧；按官方合并47指标与逐序列原始CSV复核，不能用序列均值代替合并HOTA。','',
        '| 配置 | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |','|---|---:|---:|---:|---:|---:|---:|']
    for row in [*references,*rows]:
        m=row['combined47'];name=row.get('id',row.get('case',{}).get('id'))
        text.append('| '+name+' | '+' | '.join(f'{m[k]:.6f}'for k in ['HOTA','DetA','AssA','IDF1','MOTA'])+f' | {m["IDSW"]} |')
    text+=['',f'完整 Dev 冻结选择：`{chosen["case"]["id"]}`。原始官方CSV重新核对数值字段 {fields} 项；本分区全部 {len(jobs)} 个执行任务真实退出0。',
        '', '独立重新遍历全部6个Train视频27251帧，检查早75%监督样本、归一化统计和特征/目标哈希，以及晚25%错误候选的99百分位阈值；没有增加优化步骤、随机种子或模型拟合。拟合后梯度和下一步估计记录在JSON中，用于判断实际收敛情况。',
        '', '原始 fit_history.json 是最后一次优化循环快照，其 RUNNING 字段按原样保留；实际完成由控制器真实退出0、fit_complete.json及完整校准结果证明。',
        '', 'RGB对照仅移除身份证据中的IR，视觉检测器保持相同。形状输出可通过改变GT几何匹配而改变AssA/IDF1，不应将这些变化等同于内部身份分配改进。历史Test曾暴露，Train校准使用编码器已见过的身份；缓存CPU回放不代表端到端速度。独立IR贡献和创新性仍需对应证据，工程审计本身不构成创新点。']
    if split=='dev':text+=['',f'相对冻结形状/迟链接对照的 Dev HOTA 差值：{gain:+.9f}。同策略配对IR身份特征相对RGB身份特征差值：{identity_delta:+.9f}；不据此宣称完整Test的IR独立贡献。']
    else:text+=['',f'相对原始RGB完整Test HOTA提升：{delta:+.9f}；至少+5目标：{delta>=5}，至少+10目标：{delta>=10}。全部Test视频使用同一Dev冻结方案。']
    text+=['','轻量证据目录：['+out.name+']('+out.name+'/)。不含模型系数、权重、图像、NPZ、FPN或MOT文件。','']
    report.with_suffix('.md').write_text('\n'.join(text))
    frozen_json(base/('independent_'+split+'_audit_complete.json'),dict(status=result['status'],UTC=utc(),
        report=str(report),report_sha256=sha256(report),official_numeric_fields_reparsed=fields,
        source_sha256=sha256(Path(__file__)),goal_met=False,supported_innovations=0))
    return result


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    wait_closed();audit_reg=registered();reg,fit,models,calibration=require_fitted()
    plan,protocol,work,base=locations();state=read(base/'state.json')
    fit_jobs=require_closed_jobs(state,['CANDIDATE_RISK_CPU_ENGINEERING','CANDIDATE_RISK_CPU_FIT_REGISTER',
        'CANDIDATE_RISK_CPU_FIT'],3)
    qualified=qualify_Train(work,protocol);private=read(fit['model_file']);model_sha=fit['model_sha256']
    statistics=read(base/'Train_feature_statistics.json')
    assert sha256(base/'Train_feature_statistics.json')==private['train_statistics_sha256']
    assert len(private['fit_history'])==6 and [r['pass_index']for r in private['fit_history']]==list(range(1,7))
    assert read(base/'fit_history.json')['rows']==private['fit_history']
    for r in private['fit_history']:
        for m in r['models'].values():
            assert m['rows']==statistics['known_early_edges']and abs(m['weight']-1)<1e-7 and m['actual_step_norm']<=2+1e-12
    bank_execution=read(work/'candidate_train_bank_20261127/controller.execution.json')
    first_engineering=state['phases']['CANDIDATE_RISK_CPU_ENGINEERING']['all6']
    assert stamp(first_engineering['started_UTC'])>=stamp(bank_execution['UTC_finished'])
    fit_reg=read(base/'fit_registered.json');assert stamp(fit_reg['UTC'])>=stamp(first_engineering['completion_observed_UTC'])
    assert stamp(state['phases']['CANDIDATE_RISK_CPU_FIT']['fit']['started_UTC'])>=stamp(fit_reg['UTC'])
    for j in state['phases']['CANDIDATE_RISK_DEV_REPLAY'].values():assert stamp(j['started_UTC'])>=stamp(fit['UTC'])
    started=time.monotonic()
    def progress(sequence,frame):
        guard_storage(work);atomic_json(base/'independent_audit_progress.json',dict(
            status='REBUILDING_FULL_TRAIN_AUDIT',UTC=utc(),sequence=sequence,frame=frame,
            frames=LENGTHS['train'][sequence],seconds=time.monotonic()-started,new_optimizer_steps=0))
    train=audit_full_Train(qualified,statistics,private,models,threshold_values(work)['Train_unknown_speed_upper'],progress)
    assert sha256(fit['model_file'])==model_sha
    train.update(UTC=utc(),seconds=time.monotonic()-started,private_model_file_sha256=model_sha,
        private_Train_statistics_sha256=sha256(base/'Train_feature_statistics.json'),
        auditor_registration_sha256=sha256(base/'independent_audit_registered.json'))
    frozen_json(base/'independent_Train_fit_audit.json',train)
    dev=audit_split('dev',read(base/'complete_Dev.json'),reg,fit,train,state,base,work,plan)
    test=None
    if(base/'full_Test_results.json').exists():
        test=audit_split('test',read(base/'full_Test_results.json'),reg,fit,train,state,base,work,plan)
    registered();assert sha256(fit['model_file'])==model_sha
    frozen_json(base/'independent_audit_complete.json',dict(status='COMPLETE_INDEPENDENT_TRAIN_AND_FORMAL_METRIC_AUDIT',
        UTC=utc(),Train_frames=27251,new_optimizer_steps=0,new_model_fits=0,
        Dev_numeric_fields=dev['official_numeric_fields_reparsed'],
        Test_numeric_fields=None if test is None else test['official_numeric_fields_reparsed'],
        actual_fit_stage_exits_zero=len(fit_jobs),model_bytes_unchanged=True,
        source_registration_sha256=sha256(base/'independent_audit_registered.json'),
        goal_met=False,supported_innovations=0))
    atomic_json(base/'independent_audit_progress.json',dict(status='COMPLETE_INDEPENDENT_TRAIN_AND_FORMAL_METRIC_AUDIT',UTC=utc()))
    print(json.dumps(dict(status='COMPLETE_INDEPENDENT_TRAIN_AND_FORMAL_METRIC_AUDIT',
        Train_frames=27251,Dev_fields=dev['official_numeric_fields_reparsed'],
        Test_fields=None if test is None else test['official_numeric_fields_reparsed'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','run']);args=p.parse_args()
    register()if args.mode=='register'else run()


if __name__=='__main__':main()
