#!/usr/bin/env python3
"""Independent complete-split audit for the original live tracklet R2 attempt.

No model forward, optimizer, new official evaluation or selection. Raw helper
metadata is preserved; cache-replay costs and metadata have explicit scopes.
"""
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
from tools.export_perceptual_metrics47 import official_schema
from tools.fusion_memory_evaluation import official_csv47,validate_mot
from tools.fusion_memory_observation import LENGTHS
from tools.run_mft_tracklet_permanence_r2 import PREFIX,identity,locations,registered
from tools.run_pseudo_ir_comparison import sha256


def read(path):
    """A live final-receipt write can briefly be incomplete; never restart it."""
    for attempt in range(10):
        try:return json.loads(Path(path).read_text())
        except json.JSONDecodeError:
            if attempt==9:raise
            time.sleep(.2)


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def wait_original(base,split):
    expected=read(base/'controller.execution.json')['identity']
    report=base/('complete_Dev.json'if split=='dev'else'full_Test_results.json')
    while True:
        execution=read(base/'controller.execution.json');assert execution['identity']==expected
        if report.exists()and(split=='dev'or execution.get('actual_child_exit_code')is not None):
            if split=='test':assert execution['actual_child_exit_code']==0 and not identity(expected['PID'])
            return read(report)
        state=read(base/'state.json')
        if execution.get('actual_child_exit_code')is not None:
            assert execution['actual_child_exit_code']==0,'Preserve original controller failure'
            if(split=='test'and state['status']=='COMPLETE_TRACKLET_NO_POSITIVE_DEV_GAIN_NO_NEW_TEST'):
                assert not(base/'Test_selection_frozen.json').exists()
                return None
            raise RuntimeError('Original controller exited0 without its required complete report')
        live=identity(expected['PID'])
        if live is not None:
            assert all(live[k]==expected[k]for k in ['PID','start_ticks','raw_argv_sha256'])
        time.sleep(15 if live is not None else 3)


def bank_audit(base,split,sequence,registration):
    path=base/'cache'/split/(sequence+'.json');meta=read(path)
    assert meta['status']=='COMPLETE_CAUSAL_NATIVE_IR_PROPOSAL_DUAL_IDENTITY_BANK'
    assert meta['frames']==LENGTHS[split][sequence] and meta['split']==split and meta['sequence']==sequence
    assert meta['GT_read']is False and meta['future_frame_input']is False
    assert meta['source_versions']==registration['source_hashes'] and meta['physical_gpu']in['4','5']
    assert meta['pipeline_registration_sha256']==sha256(base/'pipeline_registered.json')
    assert sha256(meta['path'])==meta['sha256']
    anchor=registration['image_anchors'][split][sequence]
    assert sha256(anchor['path'])==anchor['sha256']==meta['source_anchor_sha256']
    original=read(anchor['path'])
    assert meta['image_delivery_sha256']==original['original_image_bytes_delivery_sha256']
    count=LENGTHS[split][sequence]
    with np.load(meta['path'])as data:
        assert set(data.files)=={'detections','rgb','ir','offsets','frame_ids'}
        np.testing.assert_array_equal(data['frame_ids'],np.arange(1,count+1))
        offsets=data['offsets'];det=data['detections']
        assert offsets.dtype==np.int64 and offsets.shape==(count+1,)and offsets[0]==0
        assert np.all(np.diff(offsets)>=0)and offsets[-1]==len(det)
        assert det.dtype==np.float32 and det.shape[1:]==(5,)and np.isfinite(det).all()
        assert((det[:,4]>=0)&(det[:,4]<=1)).all()and(det[:,2:4]>det[:,:2]).all()
        norms={}
        for cue in ['rgb','ir']:
            values=data[cue];assert values.dtype==np.float32 and values.shape==(len(det),512)
            assert np.isfinite(values).all();error=float(np.max(np.abs(np.linalg.norm(values,axis=1)-1)))if len(det)else 0.
            assert error<2e-5;norms[cue]=error
        digest=hashlib.sha256()
        for i in range(count):
            lo,hi=map(int,offsets[i:i+2]);digest.update(np.asarray([i+1,hi-lo],dtype='<i8').tobytes());digest.update(np.asarray(det[lo:hi],dtype='<f4').tobytes())
    assert digest.hexdigest()==meta['detection_packet_sha256']==original['streams']['NP_temporal_ir_e06']['detection_packet_sha256']
    return dict(sequence=sequence,frames=count,proposal_count=len(det),bank_sha256=meta['sha256'],bank_bytes=meta['bytes'],
        bank_metadata_sha256=sha256(path),packet_sha256=digest.hexdigest(),identity_norm_max_error=norms,
        source_image_delivery_sha256=meta['image_delivery_sha256'],physical_gpu=meta['physical_gpu'])


def literal_geometry_hash(path):
    result=hashlib.sha256()
    with Path(path).open()as stream:
        for line in stream:
            fields=line.split(',');result.update(','.join([fields[0],*fields[2:]]).encode())
    return result.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split',choices=['dev','test'],required=True);args=p.parse_args();split=args.split
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    started=time.monotonic();reg=registered();plan,study,protocol,calibration,work,checkpoint,base=locations()
    own_receipt=base/('independent_'+split+'_audit_complete.json');assert not own_receipt.exists()
    full=wait_original(base,split)
    if full is None:
        value=dict(status='VERIFIED_TRACKLET_R2_NONPOSITIVE_DEV_NO_NEW_TEST',UTC=utc(),
            original_controller_actual_exit_code=0,new_Test=False,Test_accuracy_parsed=False,
            complete_Dev_sha256=sha256(base/'complete_Dev.json'),goal_met=False,supported_innovations=0)
        own_receipt.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True);return
    state=read(base/'state.json');dev=read(base/'complete_Dev.json')
    chosen=min(dev['rows'],key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),reg['normal_cases'].index(r['case'])))
    assert chosen['case']==dev['chosen_case']
    gain=chosen['combined47']['HOTA']-dev['rows'][0]['combined47']['HOTA'];assert gain==dev['Dev_gain_pp']
    phases=[PREFIX+'_GPU_ENGINEERING',PREFIX+'_DEV_NN',PREFIX+'_DEV_REPLAY',PREFIX+'_DEV_OFFICIAL']
    if split=='test':
        phases += [PREFIX+'_TEST_NN',PREFIX+'_TEST_REPLAY',PREFIX+'_TEST_OFFICIAL']
        freeze=read(base/'Test_selection_frozen.json')
        assert freeze['chosen_case']==chosen['case']and freeze['Test_read_for_selection']is False
        assert freeze['complete_Dev_sha256']==sha256(base/'complete_Dev.json')and gain>0
        expected_cases=freeze['cases']
    else:expected_cases=reg['normal_cases']
    jobs=[j for phase in phases for j in state['phases'][phase].values()]
    assert len(jobs)==(11 if split=='dev'else 27)
    for j in jobs:assert j['status']=='COMPLETE'and type(j.get('exit_code'))is int and j['exit_code']==0 and not identity(j['pid'])
    assert [r['case']for r in full['rows']]==expected_cases
    schema,_=official_schema(plan['evaluation']['TrackEval_root']);nfields=0
    references=[]
    for name in ['NP_source_canonical_RGB','NP_source_fresh_RGB','NP_temporal_ir_e06']:
        folder=work/'native_pyramid_evaluation'/split/name;csv=folder/'trackeval_output/pedestrian_detailed.csv'
        values=official_csv47(csv,schema,split,read(folder/'metrics.json')['trackeval']);nfields+=len(values['per_sequence47'])*47
        references.append(dict(id=name,combined47=values['combined47'],per_sequence47=values['per_sequence47'],CSV_source=str(csv),CSV_sha256=sha256(csv)))
    canonical,fresh,parent=references
    if split=='test':
        original=Path(plan['frozen_detection_sources']['rgb_test'])/'trackeval_output/pedestrian_detailed.csv'
        assert canonical['CSV_sha256']==sha256(original)=='c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
    banks={seq:bank_audit(base,split,seq,reg)for seq in LENGTHS[split]}
    checked=[]
    for row in full['rows']:
        case=row['case'];csv=Path(row['official_CSV']);assert sha256(csv)==row['official_CSV_sha256']
        values=official_csv47(csv,schema,split,row['metrics39']);nfields+=len(values['per_sequence47'])*47
        assert values['combined47']==row['combined47']and values['per_sequence47']==row['per_sequence47']
        assert row['source_versions']==reg['source_hashes']and row['frames']==sum(LENGTHS[split].values())
        coverage={}
        for seq,count in LENGTHS[split].items():
            source=read(base/'streams'/split/'sequence_manifests'/(seq+'.json'))
            assert source['status']=='COMPLETE_TRACKLET_CAUSAL_BANK_REPLAY'and source['frames']==count
            assert source['cases']==expected_cases and source['source_versions']==reg['source_hashes']
            assert source['compact_feature_cache_read']is True and source['only_current_frame_values_supplied_to_tracker']is True
            assert source['GT_read']is False and source['future_frame_input']is False and source['previous_outputs_rewritten']is False
            assert source['bank_sha256']==banks[seq]['bank_sha256']and source['source_image_delivery_sha256']==banks[seq]['source_image_delivery_sha256']
            stream=source['streams'][case['id']];track=Path(stream['track_path']);assert sha256(track)==stream['track_sha256']
            validate_mot(track,count,stream['track_rows']);assert stream['detection_packet_sha256']==banks[seq]['packet_sha256']
            geometry=literal_geometry_hash(track);assert geometry==source['nonidentity_output_rows_sha256'][case['id']]
            zero=source['streams']['TP_native_zero']
            if case['buffer']==25:assert geometry==source['nonidentity_output_rows_sha256']['TP_native_zero']
            if case['id']=='TP_native_zero':
                old=read(reg['image_anchors'][split][seq]['path'])['streams']['NP_temporal_ir_e06'];assert stream['track_sha256']==old['track_sha256']
            coverage[seq]=dict(track_sha256=stream['track_sha256'],track_rows=stream['track_rows'],frames=count,
                nonidentity_rows_sha256=geometry,identity_links=source['memory_statistics'][case['id']]['identity_links'],
                all_tracker_statistics=source['memory_statistics'][case['id']],workflow_reads_compact_feature_cache=True,
                raw_writer_local_no_cache_default=stream['feature_or_detection_disk_cache'])
        checked.append(dict(id=case['id'],case=case,combined47=values['combined47'],per_sequence47=values['per_sequence47'],
            official_CSV_source=str(csv),official_CSV_sha256=sha256(csv),coverage=coverage,
            HOTA_gain_over_same_visual_parent_pp=values['combined47']['HOTA']-parent['combined47']['HOTA']))
    zero=next(r for r in checked if r['id']=='TP_native_zero')
    assert zero['combined47']==parent['combined47']and zero['per_sequence47']==parent['per_sequence47']
    reg=registered();selected=next(r for r in checked if r['case']==dev['chosen_case'])
    output=ROOT/'docs/fusion_research'/('perceptual_20261127_tracklet_r2_full_'+split)
    output.mkdir(exist_ok=True);files=[];raw_files=[]
    def copy_file(src,name):
        src=Path(src);dest=output/name;assert not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        rel=str(dest.relative_to(ROOT));files.append(rel);raw_files.append(dict(file=rel,sha256=sha256(dest),source=str(src),source_sha256=sha256(src)));return rel
    report_source=base/('complete_Dev.json'if split=='dev'else'full_Test_results.json')
    copy_file(report_source,report_source.name);copy_file(base/'pipeline_registered.json','pipeline_registered.json')
    for seq in LENGTHS[split]:
        copy_file(base/'cache'/split/(seq+'.json'),'bank_metadata/'+seq+'.json')
        copy_file(base/'streams'/split/'sequence_manifests'/(seq+'.json'),'replay_manifests/'+seq+'.json')
    for row in checked:
        row['official_CSV']=copy_file(row['official_CSV_source'],'official/'+row['id']+'.csv')
        copy_file(base/'evaluation'/split/row['id']/'metrics.json','official/'+row['id']+'.json')
    for ref in references:ref['official_CSV']=copy_file(ref['CSV_source'],'official/'+ref['id']+'.csv')
    copy_file(base/'controller.execution.json','controller_execution_snapshot.json')
    copy_file(base/'state.json','state_snapshot.json')
    delta=selected['combined47']['HOTA']-canonical['combined47']['HOTA']
    result=dict(status='INDEPENDENTLY_VERIFIED_TRACKLET_R2_COMPLETE_'+split.upper(),UTC=utc(),split=split,
        rows=checked,references=references,banks=banks,chosen_case=dev['chosen_case'],Dev_gain_pp=gain,
        selected_HOTA_gain_over_original_RGB_pp=delta,selected_HOTA_gain_over_same_visual_parent_pp=selected['HOTA_gain_over_same_visual_parent_pp'],
        original_workers_exit_zero=len(jobs),official_numeric_fields_checked=nfields,complete_frames=sum(LENGTHS[split].values()),
        zero_all47_and_actual_MOT_exact=True,same_buffer25_all_output_boxes_scores_order_actual_bytes_equal=True,
        current_controller_execution_snapshot=read(base/'controller.execution.json'),source_versions_unchanged=len(reg['source_hashes']),
        actual_compact_feature_cache_read=True,writer_metadata_scope_note='Nested feature_or_detection_disk_cache=False is the unchanged F4 final-MOT writer-local default: that helper writes MOT and creates no proposal/feature bank. The actual enclosing workflow explicitly reads its compact NPZ bank; this audit verifies that bank and reports true cache usage. Raw helper metadata is retained.',
        runtime_scope='GPU bank-production time excludes model loading, NPZ serialization, cache replay, official evaluation and offline DiffV2IR generation. CPU replay excludes image IO/encoding; neither is selected-method end-to-end FPS.',
        old_reference_Test_HOTA=29.528970687393972,minimum_target_Test_HOTA=34.52897068739397,stretch_target_Test_HOTA=39.52897068739397,
        Test_used_for_selection=False,new_Test_accuracy_parsed_for_Dev_audit=False,new_Test_accuracy_parsed=(split=='test'),historical_Test_previously_exposed=True,
        RGB_pretraining_included_current_Dev=True,IR_generated_from_RGB=True,physical_gpus=['4','5'],fixed_seed=0,
        new_NN_training=False,extra_seeds=False,supported_innovations=0,goal_met=False,raw_files=raw_files,seconds=time.monotonic()-started)
    if split=='test':result.update(target_plus5_met=delta>=5,target_plus10_met=delta>=10)
    path=ROOT/'docs/fusion_research'/('PERCEPTUAL_MFT_20261127_TRACKLET_R2_FULL_'+split.upper()+'.json');assert not path.exists()
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');files.append(str(path.relative_to(ROOT)))
    md=f'# 身份续接R2：完整{split}独立核验\n\n'
    md+=f"完整{result['complete_frames']}帧；{len(jobs)}个相关子进程实际退出0，核对{nfields}项官方数值。零路径全部47项及实际MOT精确复现旧IR视觉父模型。表中的百分比为官方COMBINED，不能相加或简单平均逐视频HOTA。\n\n"
    md+='| 条件 | HOTA | 对同视觉父模型 | DetA | AssA | IDF1 | IDSW | 续接次数 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for row in checked:
        m=row['combined47'];md+=f"| {row['id']} | {m['HOTA']:.6f} | {row['HOTA_gain_over_same_visual_parent_pp']:+.6f} | {m['DetA']:.6f} | {m['AssA']:.6f} | {m['IDF1']:.6f} | {m['IDSW']} | {sum(x['identity_links']for x in row['coverage'].values())} |\n"
    md+=f"\n完整Dev预选 `{dev['chosen_case']['id']}`，Dev对同视觉父模型{gain:+.6f} HOTA点。"
    md+=('本表为Dev，不能据此声称Test达到+5；是否进入Test按原先登记的正增益规则决定。'if split=='dev'else f"完整Test对原始RGB-only baseline为{delta:+.6f}点，+5目标{'达到'if delta>=5 else '未达到'}；独立贡献与剩余必要评估尚未因此完成。")
    md+='\n\n25帧时长的身份续接条件逐行保留原输出框、分数、顺序，只在首次输出时选定身份；250帧时长会影响原关联，单列报告。计数减少或发生身份续接本身不证明续接正确，需结合HOTA、AssA、IDF1及完整逐视频表现判断。\n\n'
    md+='该流程确实读取紧凑特征缓存。原F4最终MOT写入器自带的每流`feature_or_detection_disk_cache=False`只描述该写入器不创建特征银行，不能当成整个流程未用缓存；上层清单明确为True，本独立核验逐帧重算检测包SHA并校验两个512维FP32身份银行。源清单原样保留。缓存回放速度排除编码和图像IO，非端到端FPS；完整成本仍需单独核验。\n\n'
    md+='全部47项、逐视频及COMBINED、原276列CSV、特征银行元数据、来源与当前执行回执见JSON及附属目录，不上传银行、MOT、图片或权重。检测器仍为同一YOLOX-X/固定新版IR父模型，使用显式已有RGB/IR ResNet50身份编码器；无新网络拟合和随机种子。Fisher等先前负结果保留。历史Test暴露、RGB原预训练见过当前Dev、伪IR由RGB派生等限制持续适用，经典身份续接不自动计为新颖贡献。Goal仍active，独立贡献计数0。\n'
    markdown=path.with_suffix('.md');markdown.write_text(md);files.append(str(markdown.relative_to(ROOT)));files.append(str(Path(__file__).relative_to(ROOT)))
    for name in files:
        q=ROOT/name;q.read_text();assert q.stat().st_size<2*2**20 and b'\0'not in q.read_bytes()
    allow=work/('tracklet_R2_full_'+split+'_publication_allowlist_20261127_r1.json')
    allow.write_text(json.dumps(dict(files=files,hashes={n:sha256(ROOT/n)for n in files},total_bytes=sum((ROOT/n).stat().st_size for n in files)),indent=2)+'\n')
    receipt=dict(status=result['status'],UTC=utc(),official_numeric_fields_checked=nfields,original_workers_exit_zero=len(jobs),selected_case=selected['id'],selected_HOTA=selected['combined47']['HOTA'],gain_over_original_RGB_pp=delta,report_sha256=sha256(path),cache_usage_scope_explicit=True,goal_met=False,supported_innovations=0)
    own_receipt.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
