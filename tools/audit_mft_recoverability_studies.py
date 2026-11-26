#!/usr/bin/env python3
"""Read-only Train calibration and full official-metric audits for C2/C3."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.fit_mft_candidate_risk import locations as parent_locations,threshold_values
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.run_mft_candidate_train_bank import atomic_json
from tools.run_mft_native_memory_priority_v3 import identity
from tools.run_pseudo_ir_comparison import sha256
from tools.fusion_memory_observation import LENGTHS
from tools.fusion_memory_evaluation import official_csv47,validate_mot
from tools.export_perceptual_metrics47 import official_schema
from tools.audit_mft_candidate_risk import require_closed_jobs,stamp
from tools.run_mft_candidate_risk import identity_score_hash


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def context(kind):
    assert kind in ['selective','archive']
    plan,protocol,work,risk=parent_locations()
    if kind=='selective':
        from tools import run_mft_selective_protection as driver
        base=work/'selective_protection_20261127';prefix='SELECTIVE_PROTECTION'
    else:
        from tools import run_mft_archive_reasoning as driver
        base=work/'archive_reasoning_evaluation_20261127';prefix='ARCHIVE_REASONING'
    return plan,protocol,work,risk,base,prefix,driver


def linear_quantile(values,q):
    """Independent sorted interpolation, rather than the producer's quantile call."""
    a=np.sort(np.asarray(values,np.float64));assert a.ndim==1 and len(a)>0 and 0<=q<=1
    assert np.isfinite(a).all()and(a>=0).all()and(a<=1).all()
    point=(len(a)-1)*q;lo=int(np.floor(point));hi=int(np.ceil(point));fraction=point-lo
    return float(a[lo]+(a[hi]-a[lo])*fraction)


def check_frozen_choice(rows,chosen_case,gate,freeze=None):
    assert len({r['case']['id']for r in rows})==len(rows)>0
    for row in rows:
        assert all(np.isfinite(row['combined47'][k])for k in ['HOTA','AssA','IDF1'])
    chosen=min(rows,key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),r['case']['id']))
    assert chosen['case']==chosen_case,'Choice differs from complete official Dev ranking'
    gain=chosen['combined47']['HOTA']-gate
    if freeze is not None:
        assert gain>1e-8 and freeze['chosen_case']==chosen_case and freeze['Test_read_for_selection']is False
    return chosen,gain


def register(kind):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan,protocol,work,risk,base,prefix,driver=context(kind);main=driver.registered()
    proof=read(base/'independent_study_CPU_regressions_r1.json')
    assert proof['actual_tool_exit_code']==0 and proof['tests_passed']==2
    for n,h in proof['source_hashes'].items():assert sha256(ROOT/n)==h,n
    dependencies=[]
    def dependency(name,path,complete):
        e=read(path);dependencies.append(dict(name=name,execution=str(path),identity=e['identity'],complete=str(complete)))
    dependency('original_study',base/'controller.execution.json',base/'complete_Dev.json')
    if kind=='selective':
        dependency('independent_original_risk_audit',risk/'independent_audit_r1.execution.json',risk/'independent_audit_complete.json')
    else:
        dependency('independent_archive_Train_audit',work/'archive_reasoning_20261127/independent_audit_r1.execution.json',
            work/'archive_reasoning_20261127/independent_Train_fit_audit.json')
        s=work/'selective_protection_20261127'
        dependency('independent_selective_study_audit',s/'independent_study_audit_r1.execution.json',s/'independent_study_audit_complete.json')
    v=dict(status='REGISTERED_READ_ONLY_FULL_'+kind.upper()+'_RESEARCH_AUDIT',UTC=utc(),kind=kind,
        source_hashes=proof['source_hashes'],pipeline_registration_sha256=sha256(base/'pipeline_registered.json'),
        CPU_proof_sha256=sha256(base/'independent_study_CPU_regressions_r1.json'),dependencies=dependencies,
        new_optimizer_steps=0,new_model_fits=0,new_GPU_work=False,
        full_Train_strict_quantile_rebuild=kind=='selective',full_official47_and_all_sequences=True,
        Test_selection=False,coefficient_or_normalization_publication=False)
    assert not(base/'independent_study_audit_registered.json').exists()
    frozen_json(base/'independent_study_audit_registered.json',v)
    print(json.dumps(dict(status=v['status'],sources=len(v['source_hashes']),dependencies=len(dependencies))),flush=True)


def registered(kind):
    plan,protocol,work,risk,base,prefix,driver=context(kind);driver.registered()
    r=read(base/'independent_study_audit_registered.json')
    assert r['kind']==kind and r['pipeline_registration_sha256']==sha256(base/'pipeline_registered.json')
    assert r['CPU_proof_sha256']==sha256(base/'independent_study_CPU_regressions_r1.json')
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    return r


def wait_closed(kind):
    reg=registered(kind);plan,protocol,work,risk,base,prefix,driver=context(kind)
    while True:
        registered(kind);guard_storage(work);waiting=[]
        for d in reg['dependencies']:
            e=read(d['execution']);assert e['identity']==d['identity']
            if e['actual_child_exit_code']is not None:
                assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(e['identity']['PID'])is None
                assert Path(d['complete']).exists()
            else:
                live=identity(e['identity']['PID'])
                if live is not None:assert all(live[k]==d['identity'][k]for k in ['PID','start_ticks','raw_argv_sha256'])
                waiting.append(dict(name=d['name'],live=live))
        if not waiting:return reg
        atomic_json(base/'independent_study_audit_progress.json',dict(status='WAITING_ACTUAL_CLOSED_STUDY_AND_TRAIN_AUDITS',
            UTC=utc(),waiting=waiting,new_optimizer_steps=0,new_GPU_work=False));time.sleep(15)


def audit_selective_Train(driver):
    from tools.mft_candidate_risk_data import qualify_Train,stream_video
    plan,protocol,work,risk,base,prefix,driver=context('selective')
    main,fit,models,thresholds,strict,ev=driver.require_evaluation()
    q=qualify_Train(work,protocol);statistics=read(risk/'Train_feature_statistics.json')
    private=read(fit['model_file']);cal=read(base/'strict_Train_thresholds.json')
    assert cal['parent_model_sha256']==fit['model_sha256']and cal['new_optimizer_steps']==cal['new_model_fits']==0
    wrong={k:[]for k in models};sizes={k:0 for k in models};coverage={};started=time.monotonic()
    for sequence,row in q.items():
        counts=dict(known=0,positive=0,negative=0)
        def sink(batch):
            if batch.frame<=int(.75*row['frames']):return
            counts['known']+=len(batch.y);counts['positive']+=int(batch.y.sum());counts['negative']+=int((batch.y==0).sum())
            for k,m in models.items():
                values=m.probabilities(batch.x)[batch.y==0].copy();sizes[k]+=values.nbytes
                assert sizes[k]<=67108864,'Never drop strict-calibration edges';wrong[k].append(values)
        def progress(frame):
            guard_storage(work);atomic_json(base/'independent_study_audit_progress.json',dict(
                status='REBUILDING_ALL6_TRAIN_STRICT_PROTECTION_CALIBRATION',UTC=utc(),sequence=sequence,
                frame=frame,frames=row['frames'],seconds=time.monotonic()-started,new_optimizer_steps=0))
        teacher=stream_video(sequence,row,threshold_values(work)['Train_unknown_speed_upper'],sink,progress=progress)
        assert teacher==statistics['full_teacher_references'][sequence]
        assert counts==private['calibration'][sequence]['counts']==cal['coverage'][sequence]['late_counts']
        coverage[sequence]=dict(frames=row['frames'],proposals=teacher['proposal_rows'],late_counts=counts,
            original_teacher_MOT_sha256=teacher['original_teacher_MOT_sha256'],teacher_MOT_exact=True)
    values={}
    for k in models:
        a=np.concatenate(wrong[k]);q99=linear_quantile(a,.99);q999=linear_quantile(a,.999)
        assert abs(q99-thresholds[k])<=2e-15 and abs(q999-strict[k])<=2e-15
        assert strict[k]>=thresholds[k]and len(a)==sum(c['late_counts']['negative']for c in coverage.values())
        values[k]=dict(wrong_edges=len(a),wrong_score_RAM_bytes=a.nbytes,recomputed_q99=q99,
            recomputed_q999=q999,producer_q99=thresholds[k],producer_q999=strict[k],absolute_tolerance=2e-15)
    assert sha256(fit['model_file'])==fit['model_sha256'];registered('selective')
    result=dict(status='INDEPENDENTLY_REBUILT_ALL6_TRAIN_STRICT_PROTECTION_QUANTILES',UTC=utc(),frames=27251,
        sequences=6,coverage=coverage,models=values,parent_model_sha256=fit['model_sha256'],model_bytes_unchanged=True,
        new_optimizer_steps=0,new_model_fits=0,new_GPU_work=False,Dev_rows=0,Test_rows=0,
        parent_independent_Train_fit_audit_sha256=sha256(risk/'independent_Train_fit_audit.json'),
        seconds=time.monotonic()-started,score_quantile_is_not_episode_error_bound=True)
    frozen_json(base/'independent_strict_Train_audit.json',result);return result


def parse_refs(work,plan,split,kind):
    schema,_=official_schema(plan['evaluation']['TrackEval_root']);refs=[]
    for name in ['NP_source_canonical_RGB','NP_source_fresh_RGB','NP_temporal_ir_e06']:
        folder=work/'native_pyramid_evaluation'/split/name;p=folder/'trackeval_output/pedestrian_detailed.csv'
        values=official_csv47(p,schema,split,read(folder/'metrics.json')['trackeval'])
        if split=='test'and name=='NP_source_canonical_RGB':
            assert sha256(p)=='c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
            assert abs(values['combined47']['HOTA']-29.528970687393972)<1e-10
        refs.append(dict(id=name,**values))
    pose=read(work/'pose_continuity_20261127'/('complete_Dev.json'if split=='dev'else'full_Test_results.json'))
    for name in ['PC_shape_anchor_late']+(['PC_late_only']if kind=='archive'else[]):
        r=next(r for r in pose['rows']if r['case']['id']==name)
        v=official_csv47(r['official_CSV'],schema,split,r['metrics39'])
        assert v['combined47']==r['combined47']and v['per_sequence47']==r['per_sequence47'];refs.append(dict(id=name,**v))
    return schema,refs


def audit_split(kind,split,source,main,fit,ev,state,stage='normal'):
    plan,protocol,work,risk,base,prefix,driver=context(kind);cases=[r['case']for r in source['rows']]
    freeze=read(base/'Test_selection_frozen.json')if(base/'Test_selection_frozen.json').exists()else None
    expected=driver.policies()if split=='dev'and stage=='normal'else(freeze['cases']if split=='test'else freeze['Dev_interventions'])
    assert cases==expected
    label=split if stage=='normal'else split+'_intervention'
    jobs=require_closed_jobs(state,[prefix+'_'+label.upper()+'_REPLAY',prefix+'_'+label.upper()+'_OFFICIAL'],len(LENGTHS[split])+len(cases))
    schema,refs=parse_refs(work,plan,split,kind);fields=47*len(LENGTHS[split].keys()|{'COMBINED'})*len(refs)
    records={};model_field='parent_model_sha256'if kind=='selective'else'model_sha256'
    for sequence,frames in LENGTHS[split].items():
        m=read(base/'streams'/label/'sequence_manifests'/(sequence+'.json'))
        assert m['cases']==cases and m['frames']==frames and m['source_versions']==main['source_hashes']
        assert m[model_field]==fit['model_sha256']and not m['GT_read']and not m['previous_outputs_rewritten']and m['compact_feature_cache_read']
        if kind=='selective':
            assert m['static_controls_or_Test_references_MOT_exact']and m['future_suffix_prefix_MOT_exact']
            assert m['missing_IR_filtered_prefix_MOT_exact_RGB_fallback']and m['only_current_frame_inputs']
            assert m['strict_thresholds_sha256']==sha256(base/'strict_Train_thresholds.json')
            if split=='dev':
                old=read(risk/'streams/dev/sequence_manifests'/(sequence+'.json'))
                for c in driver.policies()[:4]:assert m['streams'][c['id']]['track_sha256']==old['streams'][c['source_case']]['track_sha256']
                for a,b in [('SP_paired_static','SP_paired_shape_static'),('SP_paired_selective','SP_paired_shape_selective')]:
                    assert identity_score_hash(m['streams'][a]['track_path'])==identity_score_hash(m['streams'][b]['track_path'])
            else:
                old=read(work/'pose_continuity_20261127/streams/test/sequence_manifests'/(sequence+'.json'))
                for a,b in [('CR_native_zero','PC_native_zero'),('CR_pose_late_control','PC_shape_anchor_late')]:
                    assert m['streams'][a]['track_sha256']==old['streams'][b]['track_sha256']
        else:
            assert m['current_frame_inputs_only']and m['model_bytes_unchanged']
            assert m['original_references_or_RGB_missing_IR_MOT_exact']and m['future_suffix_prefix_exact']
            if stage=='intervention':
                rgb=driver.matched_case(freeze['chosen_case'],'RGB')['id']
                old=read(base/'streams/dev/sequence_manifests'/(sequence+'.json'))
                assert m['streams']['AR_intervention_missing']['track_sha256']==old['streams'][rgb]['track_sha256']
            else:
                old=read(work/'pose_continuity_20261127/streams'/split/'sequence_manifests'/(sequence+'.json'))
                for a,b in [('AR_native_zero','PC_native_zero'),('AR_fixed_RGB_late','PC_late_only'),('AR_fixed_RGB_late_shape','PC_shape_anchor_late')]:
                    if a in m['streams']:assert m['streams'][a]['track_sha256']==old['streams'][b]['track_sha256']
                for p in m['shape_identity_score_order_checks']:
                    assert identity_score_hash(m['streams'][p['current_shape']]['track_path'])==p['identity_score_order_hash']==identity_score_hash(m['streams'][p['filtered']]['track_path'])
        records[sequence]=m
    rows=[]
    for row in source['rows']:
        assert sha256(row['official_CSV'])==row['official_CSV_sha256']
        values=official_csv47(row['official_CSV'],schema,split,row['metrics39']);fields+=47*(len(LENGTHS[split])+1)
        assert values['combined47']==row['combined47']and values['per_sequence47']==row['per_sequence47']
        assert row['source_versions']==main['source_hashes']and row[model_field]==fit['model_sha256']
        coverage={}
        for sequence,m in records.items():
            s=m['streams'][row['case']['id']];expected_path=base/'streams'/label/row['case']['id']/'track_results'/(sequence+'.txt')
            assert Path(s['track_path']).resolve()==expected_path.resolve()
            assert sha256(s['track_path'])==s['track_sha256']and s['detection_packet_sha256']==m['bank']['packet_sha256']
            validate_mot(s['track_path'],LENGTHS[split][sequence],s['track_rows'])
            coverage[sequence]=dict(track_sha256=s['track_sha256'],track_rows=s['track_rows'],statistics=m['statistics'][row['case']['id']])
        rows.append(dict(case=row['case'],coverage=coverage,**values))
    details={}
    if split=='dev'and stage=='normal':
        if kind=='selective':
            gate=ev['parent_best_Dev_metrics']['HOTA'];parent=read(risk/'complete_Dev.json')
            parent_best=min(parent['rows'],key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),r['case']['id']))
            assert parent_best['case']==ev['parent_best_Dev_case']and parent_best['combined47']['HOTA']==gate
            for r in rows[:4]:
                old=next(v for v in parent['rows']if v['case']['id']==r['case']['source_case'])
                assert r['combined47']==old['combined47']and r['per_sequence47']==old['per_sequence47']
        else:
            gate=ev['parent_best_Dev_HOTA'];prior=[]
            for ref in ev['parent_complete_Dev_references']:
                original=read(work/ref['study']/'complete_Dev.json')
                best=min(original['rows'],key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),r['case']['id']))
                assert best['case']==ref['case']and best['combined47']==ref['combined47'];prior.append(best['combined47']['HOTA'])
            assert gate==max(prior)
            for index,ref_index in [(0,2),(1,4),(2,3)]:
                assert rows[index]['combined47']==refs[ref_index]['combined47']and rows[index]['per_sequence47']==refs[ref_index]['per_sequence47']
        chosen,gain=check_frozen_choice(rows,source['chosen_case'],gate,freeze)
        assert source['Test_read_for_selection']is False and gain==source['gain_over_parent_best_Dev_pp']
        if gain>1e-8:
            assert freeze is not None and freeze['complete_Dev_sha256']==sha256(base/'complete_Dev.json')
            assert stamp(source['UTC'])<=stamp(freeze['UTC'])
            for j in state['phases'][prefix+'_TEST_REPLAY'].values():assert stamp(j['started_UTC'])>=stamp(freeze['UTC'])
            if kind=='selective':
                from tools.mft_candidate_risk_association import policies as originals
                assert freeze['cases']==[*originals()[:2],chosen['case']]
            else:
                other=driver.matched_case(chosen['case'],'paired'if chosen['case']['kind']=='RGB'else'RGB')
                assert freeze['matched_identity_control']==other and freeze['cases']==[driver.policies()[0],driver.policies()[2],chosen['case'],other]
                assert freeze['Dev_interventions']==driver.interventions(chosen['case'])
        else:assert freeze is None and not(base/'full_Test_results.json').exists()
        details=dict(chosen_case=chosen['case'],gain_over_parent_best_Dev_pp=gain,Test_used_for_selection=False)
    elif stage=='intervention':
        assert kind=='archive'and source['no_reselection_after_controls']and source['missing_IR_all47_and_MOT_exact_RGB']
        normal=read(base/'complete_Dev.json');rgb=driver.matched_case(freeze['chosen_case'],'RGB')
        old=next(r for r in normal['rows']if r['case']==rgb)
        assert rows[0]['combined47']==old['combined47']and rows[0]['per_sequence47']==old['per_sequence47']
        paired=driver.matched_case(freeze['chosen_case'],'paired');old=next(r for r in normal['rows']if r['case']==paired)
        details=dict(chosen_case=freeze['chosen_case'],missing_IR_HOTA_minus_RGB_pp=0.,
            rolled_IR_HOTA_minus_normal_paired_pp=rows[1]['combined47']['HOTA']-old['combined47']['HOTA'],no_reselection_after_controls=True)
    else:
        assert source['Test_used_for_selection']is False and source['selected_case']==freeze['chosen_case']
        chosen=next(r for r in rows if r['case']==freeze['chosen_case']);delta=chosen['combined47']['HOTA']-refs[0]['combined47']['HOTA']
        assert delta==source['HOTA_gain_over_original_RGB_pp']and source['target_plus5_met']==(delta>=5)and source['target_plus10_met']==(delta>=10)
        details=dict(chosen_case=chosen['case'],HOTA_gain_over_original_RGB_pp=delta,target_plus5_met=delta>=5,
            target_plus10_met=delta>=10,Test_used_for_selection=False)
        if kind=='archive':
            rgb=next(r for r in rows if r['case']['kind']=='RGB');paired=next(r for r in rows if r['case']['kind']=='paired')
            details['matched_identity_IR_HOTA_addition_pp']=paired['combined47']['HOTA']-rgb['combined47']['HOTA']
    return dict(split=split,stage=stage,rows=rows,references=refs,records=records,
        official_numeric_fields_reparsed=fields,actual_split_worker_exits_zero=len(jobs),**details)


def publish(kind,parts,main,fit,train_audit,state):
    plan,protocol,work,risk,base,prefix,driver=context(kind)
    out=ROOT/'docs/fusion_research'/('perceptual_20261127_'+kind+'_full_audit');assert not out.exists();out.mkdir()
    copied=[]
    def copy(source,name):
        p=Path(source);dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True)
        assert p.stat().st_size<1024**2;shutil.copyfile(p,dest)
        copied.append(dict(file=str(dest.relative_to(ROOT)),sha256=sha256(dest),source_sha256=sha256(p),bytes=dest.stat().st_size))
        return str(dest.relative_to(ROOT))
    for name in ['pipeline_registered.json','evaluation_registered.json','controller.execution.json','state.json',
            'independent_study_audit_registered.json','independent_study_CPU_regressions_r1.json']:
        copy(base/name,name)
    if kind=='selective':
        for name in ['strict_Train_thresholds.json','calibration_registered.json','real_Train_engineering.json','independent_strict_Train_audit.json']:copy(base/name,name)
    else:
        a=work/'archive_reasoning_20261127'
        for name in ['fit_complete.json','independent_Train_fit_audit.json','fit_r1.execution.json','independent_audit_r1.execution.json']:copy(a/name,'Train/'+name)
        copy(base/'real_Train_engineering.json','real_Train_engineering.json')
    if(base/'Test_selection_frozen.json').exists():copy(base/'Test_selection_frozen.json','Test_selection_frozen.json')
    for part in parts:
        label=part['split']if part['stage']=='normal'else'dev_intervention'
        for sequence in part.pop('records'):copy(base/'streams'/label/'sequence_manifests'/(sequence+'.json'),'manifests/'+label+'/'+sequence+'.json')
        for row in part['rows']+part['references']:
            name=row.get('id',row.get('case',{}).get('id'));source=row['official_CSV'];row['official_CSV_source']=source
            row['official_CSV']=copy(source,'official/'+label+'/'+name+'.csv')
    total_fields=sum(p['official_numeric_fields_reparsed']for p in parts)
    result=dict(status='INDEPENDENTLY_VERIFIED_COMPLETE_'+kind.upper()+'_STUDY',UTC=utc(),parts=parts,
        official_numeric_fields_reparsed=total_fields,source_registration_sha256=sha256(base/'pipeline_registered.json'),
        auditor_registration_sha256=sha256(base/'independent_study_audit_registered.json'),
        model_sha256=fit['model_sha256'],Train_audit=train_audit,original_controller_actual_exit=0,
        all_actual_jobs_exit_zero=True,Test_selection=False,model_bytes_unchanged=True,
        source_detector_historical_Dev_exposure=True,historical_Test_previously_exposed=True,
        RGB_identity_control_retains_common_pseudoIR_visual_detector=True,cached_CPU_replay_not_end_to_end_FPS=True,
        coefficient_or_normalization_publication=False,goal_met=False,supported_innovations=0,lightweight_files=copied)
    report=ROOT/'docs/fusion_research'/('PERCEPTUAL_MFT_20261127_'+kind.upper()+'_FULL_AUDIT.json');assert not report.exists()
    frozen_json(report,result)
    lines=['# '+('条件保护'if kind=='selective'else'身份重连')+'完整实验与独立复核','',
        '固定十一月DiffV2IR、ProtocolV2和原始RGB YOLOX-X+ByteTrack。Test基线29.5289706874，至少+5目标34.5289706874；仅完整Dev选择，历史Test暴露继续披露。','']
    for part in parts:
        lines+=['## '+part['split'].upper()+' '+part['stage'],'','| 配置 | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |','|---|---:|---:|---:|---:|---:|---:|']
        for r in part['references']+part['rows']:
            m=r['combined47'];name=r.get('id',r.get('case',{}).get('id'))
            lines.append('| '+name+' | '+' | '.join(f'{m[k]:.6f}'for k in ['HOTA','DetA','AssA','IDF1','MOTA'])+f' | {m["IDSW"]} |')
        if 'gain_over_parent_best_Dev_pp'in part:lines+=['',f'相对已有最佳完整Dev HOTA：{part["gain_over_parent_best_Dev_pp"]:+.9f}。冻结选择 `{part["chosen_case"]["id"]}`。']
        if 'HOTA_gain_over_original_RGB_pp'in part:lines+=['',f'冻结方法相对原始RGB完整Test：{part["HOTA_gain_over_original_RGB_pp"]:+.9f}；+5达标：{part["target_plus5_met"]}；+10达标：{part["target_plus10_met"]}。']
    lines+=['',f'独立重读全部官方47指标及逐序列CSV，合计复核{total_fields}项数值，检查原控制器和全部任务实际退出0、源码/模型哈希、完整MOT、当前帧输入、历史输出不改写、匹配参考复现及Dev/Test冻结时间顺序。',
        '', '条件保护另行遍历全部6个Train序列复算99/99.9错误边分位数；身份重连复用已完成的独立全Train拟合审计。没有新增优化、种子或模型；CSV保留原始字节，模型系数、归一化统计、权重、图像、NPZ和MOT不进入Git。',
        '', '解锁保护、改变匹配或接受链接的次数不等同纠错数。形状输出改变GT几何匹配，不能将AssA/IDF1变化直接当作内部ID改善。错误边分位数不保证整段链接错误率。编码器已见Train身份、源检测器历史见过Dev、伪IR是RGB派生且当前身份对照共用IR视觉检测器；独立模态贡献与创新性仍需对应证据，审计本身不构成创新。',
        '', '['+out.name+']('+out.name+'/)。','']
    report.with_suffix('.md').write_text('\n'.join(lines))
    frozen_json(base/'independent_study_audit_complete.json',dict(status=result['status'],UTC=utc(),report=str(report),
        report_sha256=sha256(report),official_numeric_fields_reparsed=total_fields,model_bytes_unchanged=True,
        new_optimizer_steps=0,new_model_fits=0,goal_met=False,supported_innovations=0))


def run(kind):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=wait_closed(kind);plan,protocol,work,risk,base,prefix,driver=context(kind);state=read(base/'state.json')
    for name in state['phases']:require_closed_jobs(state,[name],len(state['phases'][name]))
    if kind=='selective':
        main,fit,models,thresholds,strict,ev=driver.require_evaluation();train=audit_selective_Train(driver)
    else:
        main,fit,models,thresholds,ev=driver.ready();train=read(work/'archive_reasoning_20261127/independent_Train_fit_audit.json')
        assert train['model_sha256']==fit['model_sha256']and train['model_bytes_unchanged']
    parts=[audit_split(kind,'dev',read(base/'complete_Dev.json'),main,fit,ev,state)]
    if(base/'full_Test_results.json').exists():
        if kind=='archive':parts.append(audit_split(kind,'dev',read(base/'complete_Dev_interventions.json'),main,fit,ev,state,'intervention'))
        parts.append(audit_split(kind,'test',read(base/'full_Test_results.json'),main,fit,ev,state))
    assert sha256(fit['model_file'])==fit['model_sha256'];registered(kind);publish(kind,parts,main,fit,train,state)
    atomic_json(base/'independent_study_audit_progress.json',dict(status='COMPLETE_READ_ONLY_FULL_STUDY_AUDIT',UTC=utc()))
    print(json.dumps(dict(status='COMPLETE_READ_ONLY_FULL_'+kind.upper()+'_AUDIT',
        official_numeric_fields_reparsed=sum(p['official_numeric_fields_reparsed']for p in parts),new_optimizer_steps=0)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','run']);p.add_argument('--kind',required=True,choices=['selective','archive'])
    a=p.parse_args();register(a.kind)if a.mode=='register'else run(a.kind)


if __name__=='__main__':main()
