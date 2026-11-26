#!/usr/bin/env python3
"""Audit only the Dev-frozen pose Test, never choose a model from Test."""
import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_mft_tracklet_r2 import bank_audit
from tools.export_perceptual_metrics47 import official_schema
from tools.fusion_memory_evaluation import official_csv47,validate_mot
from tools.fusion_memory_observation import LENGTHS
from tools.native_pyramid_experiment import read
from tools.run_mft_pose_continuity import locations,registered,identity_score_hash,policies
from tools.run_mft_tracklet_permanence_r2 import registered as bank_registration
from tools.run_mft_native_memory_priority_v3 import identity
from tools.run_pseudo_ir_comparison import sha256


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan,study,protocol,calibration,work,checkpoint,base,peer=locations()
    receipt=base/'independent_Test_audit_complete.json';assert not receipt.exists()
    expected=read(base/'controller.execution.json')['identity']
    while True:
        execution=read(base/'controller.execution.json');assert execution['identity']==expected
        if execution.get('actual_child_exit_code')is not None:
            assert execution['actual_child_exit_code']==0,'Preserve original controller failure'
            assert identity(expected['PID'])is None
            break
        time.sleep(10)
    reg=registered();dev=read(base/'complete_Dev.json')
    chosen_dev=min(dev['rows'],key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),policies().index(r['case'])))
    assert chosen_dev['case']==dev['chosen_case']and dev['Test_read_for_selection']is False
    if dev['gain_over_late_only_pp']<=0:
        assert not(base/'Test_selection_frozen.json').exists()and not(base/'full_Test_results.json').exists()
        value=dict(status='VERIFIED_POSE_NONPOSITIVE_DEV_NO_TEST',actual_main_exit_code=0,Test_accuracy_parsed=False,goal_met=False)
        receipt.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True);return
    freeze=read(base/'Test_selection_frozen.json');full=read(base/'full_Test_results.json')
    assert freeze['chosen_case']==chosen_dev['case']==full['selected_case']
    assert freeze['complete_Dev_sha256']==sha256(base/'complete_Dev.json')and freeze['Test_read_for_selection']is False
    assert full['Test_used_for_selection']is False
    assert [r['case']for r in full['rows']]==freeze['cases']
    jobs=[j for rows in read(base/'state.json')['phases'].values()for j in rows.values()]
    assert len(jobs)==20
    for job in jobs:
        assert job['status']=='COMPLETE'and type(job.get('exit_code'))is int and job['exit_code']==0
        assert job['argv_identity_observed']and identity(job['pid'])is None
    banks={s:bank_audit(peer,'test',s,bank_registration())for s in LENGTHS['test']}
    schema,_=official_schema(plan['evaluation']['TrackEval_root'])
    refs=[];fields=0
    for name in ['NP_source_canonical_RGB','NP_source_fresh_RGB','NP_temporal_ir_e06']:
        folder=work/'native_pyramid_evaluation/test'/name;csv=folder/'trackeval_output/pedestrian_detailed.csv'
        values=official_csv47(csv,schema,'test',read(folder/'metrics.json')['trackeval']);fields+=len(values['per_sequence47'])*47
        refs.append(dict(id=name,**values))
    assert refs[0]['official_CSV_sha256']=='c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
    out=ROOT/'docs/fusion_research/perceptual_20261127_pose_continuity_full_test';assert not out.exists();out.mkdir()
    files=[];raw=[]

    def copy(src,name):
        src=Path(src);dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();shutil.copyfile(src,dest)
        n=str(dest.relative_to(ROOT));files.append(n);raw.append(dict(file=n,sha256=sha256(dest),source=str(src),source_sha256=sha256(src)));return n

    for name in ['full_Test_results.json','complete_Dev.json','Test_selection_frozen.json','pipeline_registered.json','controller.execution.json','state.json']:
        copy(base/name,name)
    manifests={}
    for seq,count in LENGTHS['test'].items():
        path=base/'streams/test/sequence_manifests'/(seq+'.json');record=read(path)
        assert record['frames']==count and record['source_versions']==reg['source_hashes']and record['cases']==freeze['cases']
        assert record['GT_read']is False and record['future_frame_input']is False and record['previous_outputs_rewritten']is False
        assert record['only_current_frame_values_supplied']and record['compact_feature_cache_read']
        assert record['future_suffix_corruption_prefix_invariant']and record['scoped_prediction_filter_restored']
        assert record['bank']['bank_sha256']==banks[seq]['bank_sha256']
        prior=read(peer/'streams/test/sequence_manifests'/(seq+'.json'))
        for own,old in [('PC_native_zero','TP_native_zero'),('PC_late_only','TP_late_RGB')]:
            assert record['streams'][own]['track_sha256']==prior['streams'][old]['track_sha256']
        copy(path,'replay_manifests/'+seq+'.json');manifests[seq]=record
    rows=[]
    for row in full['rows']:
        case=row['case'];csv=Path(row['official_CSV']);assert sha256(csv)==row['official_CSV_sha256']
        values=official_csv47(csv,schema,'test',row['metrics39']);fields+=len(values['per_sequence47'])*47
        assert values['combined47']==row['combined47']and values['per_sequence47']==row['per_sequence47']
        assert row['source_versions']==reg['source_hashes']and row['frames']==15985
        coverage={}
        for seq,count in LENGTHS['test'].items():
            record=manifests[seq];stream=record['streams'][case['id']]
            assert sha256(stream['track_path'])==stream['track_sha256']and stream['detection_packet_sha256']==banks[seq]['packet_sha256']
            validate_mot(stream['track_path'],count,stream['track_rows']);detail={}
            if case['render']!='filtered':
                p=record['anchor_frame_ID_score_order_exact_proofs'][case['id']]
                old=read(peer/'streams/test/sequence_manifests'/(seq+'.json'))['streams']['TP_late_RGB'if case['late']else'TP_native_zero']
                actual=identity_score_hash(stream['track_path']);expected_meta=identity_score_hash(old['track_path'])
                assert actual==expected_meta==p['actual_frame_ID_score_order_hash']==p['reference_hash']
                detail=dict(actual_frame_ID_score_order_hash=actual,reference_hash=expected_meta)
            coverage[seq]=dict(track_sha256=stream['track_sha256'],track_rows=stream['track_rows'],statistics=record['statistics'][case['id']],**detail)
        exported=copy(csv,'official/'+case['id']+'.csv');copy(base/'evaluation/test'/case['id']/'metrics.json','official/'+case['id']+'.json')
        rows.append({**values,'case':case,'coverage':coverage,'official_CSV_source':str(csv),'official_CSV':exported})
    for r in refs:
        r['official_CSV_source']=r['official_CSV'];r['official_CSV']=copy(r['official_CSV_source'],'official/'+r['id']+'.csv')
    assert fields==2256
    selected=next(r for r in rows if r['case']==freeze['chosen_case'])
    assert rows[0]['combined47']==refs[2]['combined47']and rows[0]['per_sequence47']==refs[2]['per_sequence47']
    prior_selected=read(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261127_TRACKLET_R2_FULL_TEST.json')
    prior_selected=next(r for r in prior_selected['rows']if r['id']=='TP_late_RGB')
    assert rows[1]['combined47']==prior_selected['combined47']and rows[1]['per_sequence47']==prior_selected['per_sequence47']
    delta=selected['combined47']['HOTA']-refs[0]['combined47']['HOTA'];assert delta==full['HOTA_gain_over_original_RGB_pp']
    result=dict(status='INDEPENDENTLY_VERIFIED_DEV_FROZEN_POSE_COMPLETE_TEST',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rows=rows,references=refs,banks=banks,selected_case=freeze['chosen_case'],selected_Test_HOTA=selected['combined47']['HOTA'],
        original_RGB_Test_HOTA=refs[0]['combined47']['HOTA'],HOTA_gain_over_original_RGB_pp=delta,
        HOTA_gain_over_same_IR_late_parent_pp=selected['combined47']['HOTA']-rows[1]['combined47']['HOTA'],
        minimum_target_Test_HOTA=refs[0]['combined47']['HOTA']+5,stretch_target_Test_HOTA=refs[0]['combined47']['HOTA']+10,
        target_plus5_met=delta>=5,target_plus10_met=delta>=10,complete_Test_sequences=7,complete_Test_frames=15985,
        actual_main_exit_code=0,all20_related_workers_actual_exit_zero=True,new_Test_stage_workers_actual_exit_zero=10,
        official_numeric_fields_checked=fields,source_versions_unchanged=len(reg['source_hashes']),auditor_source_sha256=sha256(Path(__file__)),
        Test_used_for_selection=False,historical_Test_previously_exposed=True,earlier_R2_Test_seen_by_agent_before_pose_Dev=True,
        hypothesis_written_before_agent_R2_Test_review=True,implementation_after_agent_R2_Test_review=True,
        RGB_pretraining_included_current_Dev=True,IR_generated_from_RGB=True,
        zero_and_late_all47_and_actual_MOT_exact=True,output_anchor_frame_ID_score_order_row_counts_exact=True,
        runtime_scope='CPU replay excludes all image/NN and offline IR generation costs; not end-to-end FPS.',
        new_NN_fits=0,new_GPU_extraction=False,new_seeds=0,fixed_seed=0,goal_met=False,supported_innovations=0,raw_files=raw)
    report=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261127_POSE_CONTINUITY_FULL_TEST.json';assert not report.exists();report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');files.append(str(report.relative_to(ROOT)))
    md='# 当前形状与身份连续性：完整七段Test\n\n'
    md+=f"完整15985帧，全部20个相关worker实际0，独立核对2256个官方字段。方案仅按完整Dev预选`{freeze['chosen_case']['id']}`。原始RGB HOTA{refs[0]['combined47']['HOTA']:.6f}，本方案{selected['combined47']['HOTA']:.6f}，**{delta:+.6f}个绝对HOTA点**；+5目标{'达到'if delta>=5 else '未达到'}。\n\n"
    md+='| 条件 | HOTA | 对原始RGB | DetA | AssA | IDF1 | MOTA | IDSW |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in [*refs,*rows]:
        m=r['combined47'];name=r.get('id',r.get('case',{}).get('id'));md+=f"| {name} | {m['HOTA']:.6f} | {m['HOTA']-refs[0]['combined47']['HOTA']:+.6f} | {m['DetA']:.6f} | {m['AssA']:.6f} | {m['IDF1']:.6f} | {m['MOTA']:.6f} | {m['IDSW']} |\n"
    md+='\n零与late-only的全部47字段、逐视频和真实MOT精确回到各自旧参考。输出形状/框锚定在七段视频中保留frame/ID/score/order/行数的真实文件哈希，只改当前输出几何，不把渲染反馈给内部状态；不能把HOTA/AssA改善都解释为身份分配改变。\n\n'
    md+='伪IR视觉检测器、源权重与11月图像不变；复用此前真实两卡提取的FP32银行，无新增网络训练、GPU提取或种子。全部47字段、逐视频和COMBINED及原276列CSV公开，银行、MOT、图像和权重不上传。回放速度不包含编码、检测、图像IO和离线IR生成，不是端到端FPS。\n\n'
    md+='此前R2 Test已在本研究实现前被agent阅读；八项假设在该阅读前写定，之后实现并按完整Dev选型。历史Test暴露、RGB预训练见当前Dev、IR由RGB派生等限制保留，不声称新盲测。整体差值不等于红外独立贡献，经典组合不自动证明新颖性。数值是否达到+5单独报告，Goal是否完成还需最终复核和交付。\n'
    report.with_suffix('.md').write_text(md);files.extend([str(report.with_suffix('.md').relative_to(ROOT)),str(Path(__file__).relative_to(ROOT))])
    for n in files:
        p=ROOT/n;p.read_text();assert p.stat().st_size<2*2**20 and b'\0'not in p.read_bytes()
    allow=dict(files=files,hashes={n:sha256(ROOT/n)for n in files},total_bytes=sum((ROOT/n).stat().st_size for n in files))
    (work/'pose_continuity_full_Test_publication_allowlist_20261127_r1.json').write_text(json.dumps(allow,indent=2)+'\n')
    value=dict(status=result['status'],UTC=result['UTC'],selected_case=freeze['chosen_case'],HOTA=selected['combined47']['HOTA'],
        gain_over_original_RGB_pp=delta,target_plus5_met=delta>=5,target_plus10_met=delta>=10,original_workers_actual_exit_zero=20,
        official_numeric_fields_checked=fields,report_sha256=sha256(report),new_Test_used_for_selection=False,goal_met=False)
    receipt.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True)


if __name__=='__main__':main()
