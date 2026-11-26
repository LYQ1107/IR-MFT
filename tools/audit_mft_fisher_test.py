#!/usr/bin/env python3
"""Conditional, independent complete-Test export after the original driver exits.

No inference, fitting, model selection, GT evaluation or GPU allocation. If
complete Dev is nonpositive, record the intentional absence of a new Test.
"""
import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.export_perceptual_metrics47 import official_schema
from tools.fusion_memory_evaluation import official_csv47, validate_mot
from tools.fusion_memory_observation import LENGTHS
from tools.run_mft_fisher_memory import PREFIX, identity, locations, registered
from tools.run_pseudo_ir_comparison import sha256


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    started = time.monotonic()
    plan, study, protocol, calibration, work, checkpoint, base = locations()
    terminal = base/'independent_Test_audit_complete.json'
    assert not terminal.exists(), 'Preserve any prior independent audit attempt'
    registration = registered()
    expected = read(base/'controller.execution.json')['identity']
    while True:
        execution = read(base/'controller.execution.json')
        assert execution['identity'] == expected
        if execution.get('actual_child_exit_code') is not None:
            break
        current = identity(expected['PID'])
        if current is not None:
            assert all(current[k] == expected[k] for k in ['PID','start_ticks','raw_argv_sha256'])
        # PID disappearance alone may precede the wrapper's final receipt write.
        time.sleep(5 if current is None else 15)
    assert execution['actual_child_exit_code'] == 0 and execution['source_unchanged'], 'Preserve actual original driver failure'
    assert not identity(expected['PID'])
    state, dev = read(base/'state.json'), read(base/'complete_Dev.json')
    if state['status'] == 'COMPLETE_FISHER_NO_POSITIVE_DEV_GAIN_NO_NEW_TEST':
        assert dev['Dev_gain_over_identical_visual_parent_pp'] <= 0
        assert not (base/'Test_selection_frozen.json').exists() and not (base/'full_Test_results.json').exists()
        result = dict(status='VERIFIED_NONPOSITIVE_FISHER_DEV_NO_NEW_TEST',UTC=utc(),
            original_controller_actual_exit_code=0,complete_Dev_sha256=sha256(base/'complete_Dev.json'),
            Test_accuracy_read=False,new_Test=False,goal_met=False,supported_innovations=0)
        terminal.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        return
    assert state['status'] == 'COMPLETE_FISHER_FULL_TEST_REQUIRES_SCIENTIFIC_ASSESSMENT'
    report_path, freeze_path = base/'full_Test_results.json', base/'Test_selection_frozen.json'
    original, freeze = read(report_path), read(freeze_path)
    assert original['status'] == 'COMPLETE_DEV_FROZEN_PAIRED_FISHER_FULL7_TEST'
    assert freeze['Test_read_for_selection'] is False and freeze['chosen_case'] == dev['chosen_case']
    assert freeze['complete_Dev_sha256'] == sha256(base/'complete_Dev.json')
    assert original['primary_case'] == dev['chosen_case']
    chosen = next(r for r in dev['rows'] if r['case'] == dev['chosen_case'])
    best = min(dev['rows'],key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),r['case']['lost_buffer'],r['case']['appearance_weight']))
    assert chosen == best and chosen['combined47']['HOTA'] > dev['rows'][0]['combined47']['HOTA']
    expected_phases = [PREFIX+'_GPU_ENGINEERING',PREFIX+'_DEV_NN',PREFIX+'_DEV_OFFICIAL',PREFIX+'_TEST_NN',PREFIX+'_TEST_OFFICIAL']
    assert set(state['phases']) == set(expected_phases)
    jobs = [j for phase in expected_phases for j in state['phases'][phase].values()]
    assert len(jobs) == 18
    assert all(j['status']=='COMPLETE' and type(j.get('exit_code')) is int and j['exit_code']==0
               and not identity(j['pid']) for j in jobs)
    rows = original['rows']
    assert [r['case']for r in rows] == freeze['cases'] and len(rows) == 2
    schema,_ = official_schema(plan['evaluation']['TrackEval_root'])
    output = ROOT/'docs/fusion_research/perceptual_20261127_fisher_full_Test'
    output.mkdir(exist_ok=True)
    files, raw = [], []
    def copy_file(source,relative):
        source = Path(source);destination = output/relative
        assert not destination.exists(), 'Preserve earlier publication files'
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
        name = str(destination.relative_to(ROOT));files.append(name)
        raw.append(dict(file=name,sha256=sha256(destination),source=str(source),source_sha256=sha256(source)))
        return name
    for name in ['full_Test_results.json','Test_selection_frozen.json','controller.execution.json',
                 'state.json','specifications/test.json']:
        copy_file(base/name,name)
    def reference(name):
        folder = work/'native_pyramid_evaluation/test'/name
        csv = folder/'trackeval_output/pedestrian_detailed.csv'
        values = official_csv47(csv,schema,'test',read(folder/'metrics.json')['trackeval'])
        public = copy_file(csv,'official/'+name+'.csv')
        return dict(id=name,combined47=values['combined47'],per_sequence47=values['per_sequence47'],
                    official_CSV=public,official_CSV_sha256=sha256(csv))
    canonical, fresh, old_parent = [reference(n)for n in ['NP_source_canonical_RGB','NP_source_fresh_RGB','NP_temporal_ir_e06']]
    canonical_source = Path(plan['frozen_detection_sources']['rgb_test'])/'trackeval_output/pedestrian_detailed.csv'
    assert canonical['official_CSV_sha256'] == sha256(canonical_source) == 'c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
    assert original['original_RGB_baseline']['combined47'] == canonical['combined47']
    checked = []
    for row in rows:
        case = row['case']['id']
        csv = Path(row['official_CSV'])
        assert sha256(csv) == row['official_CSV_sha256']
        values = official_csv47(csv,schema,'test',row['metrics39'])
        assert values['combined47'] == row['combined47'] and values['per_sequence47'] == row['per_sequence47']
        public = copy_file(csv,'official/'+case+'.csv')
        copy_file(base/'evaluation/test'/case/'metrics.json','official/'+case+'.json')
        coverage = {}
        for seq,count in LENGTHS['test'].items():
            source = read(base/'streams/test/sequence_manifests'/(seq+'.json'))
            anchor = read(work/'native_pyramid_streams/test_original_baseline_priority_20261126/sequence_manifests'/(seq+'.json'))
            assert source['frames'] == count and source['split'] == 'test' and source['GT_read'] is False
            assert source['future_frame_input'] is False and source['previous_outputs_rewritten'] is False
            assert source['original_image_bytes_delivery_sha256'] == anchor['original_image_bytes_delivery_sha256']
            assert source['provenance']['source_versions'] == registration['source_hashes']
            stream = source['streams'][case]
            assert stream['detection_packet_sha256'] == anchor['streams']['NP_temporal_ir_e06']['detection_packet_sha256']
            path = base/'streams/test'/case/'track_results'/(seq+'.txt')
            assert sha256(path) == stream['track_sha256']
            validate_mot(path,count,stream['track_rows'])
            if row['case']['appearance_weight'] == 0:
                assert stream['track_sha256'] == anchor['streams']['NP_temporal_ir_e06']['track_sha256']
            coverage[seq] = dict(frames=count,track_sha256=sha256(path),track_rows=stream['track_rows'],
                detector_packet_sha256=stream['detection_packet_sha256'],tracker_statistics=source['memory_statistics'][case])
        checked.append(dict(id=case,case=row['case'],combined47=values['combined47'],per_sequence47=values['per_sequence47'],
            official_CSV=public,official_CSV_sha256=sha256(csv),coverage=coverage))
    zero = next(r for r in checked if r['case']['appearance_weight']==0)
    selected = next(r for r in checked if r['case']==freeze['chosen_case'])
    assert zero['combined47'] == old_parent['combined47'] and zero['per_sequence47'] == old_parent['per_sequence47']
    for seq in LENGTHS['test']:
        copy_file(base/'streams/test/sequence_manifests'/(seq+'.json'),'producer_manifests/'+seq+'.json')
        copy_file(base/'producer_validation/test'/(seq+'.json'),'producer_validation/'+seq+'.json')
    delta = selected['combined47']['HOTA']-canonical['combined47']['HOTA']
    assert delta == original['HOTA_gain_over_original_RGB_pp']
    per_video=[]
    for seq in LENGTHS['test']:
        a=next(r for r in selected['per_sequence47']if r['seq']==seq)
        b=next(r for r in canonical['per_sequence47']if r['seq']==seq)
        p=next(r for r in old_parent['per_sequence47']if r['seq']==seq)
        per_video.append(dict(sequence=seq,selected_HOTA=a['HOTA'],original_RGB_HOTA=b['HOTA'],
            visual_parent_HOTA=p['HOTA'],gain_over_original_RGB_pp=a['HOTA']-b['HOTA'],gain_over_visual_parent_pp=a['HOTA']-p['HOTA']))
    registered()
    result = dict(status='INDEPENDENTLY_VERIFIED_DEV_FROZEN_FISHER_COMPLETE7_TEST',UTC=utc(),
        references=[canonical,fresh,old_parent],rows=checked,selected_case=selected['case'],
        HOTA_gain_over_original_RGB_pp=delta,HOTA_gain_over_visual_parent_pp=selected['combined47']['HOTA']-old_parent['combined47']['HOTA'],
        target_plus5_met=delta>=5,target_plus10_met=delta>=10,minimum_target_HOTA=canonical['combined47']['HOTA']+5,
        stretch_target_HOTA=canonical['combined47']['HOTA']+10,per_video_HOTA=per_video,
        complete_Test_frames=15985,complete_Test_sequences=7,original_controller_actual_exit_code=0,
        original_workers_actual_exit_zero=18,independent_numeric_fields_verified=5*8*47,
        zero_all47_per_sequence_and_MOT_exact_to_prior_native_parent=True,
        actual_IR_detector_packets_identical_all_cases=True,source_versions_unchanged=len(registration['source_hashes']),
        Test_used_for_selection=False,historical_Test_previously_exposed=True,
        RGB_source_pretraining_included_current_Dev=True,IR_is_generated_from_RGB=True,
        no_additional_seeds=True,no_additional_RGB_architecture=True,physical_gpus=['4','5'],maximum_cards=2,
        Test_freeze_sha256=sha256(freeze_path),complete_Dev_sha256=sha256(base/'complete_Dev.json'),
        original_full_Test_result_sha256=sha256(report_path),raw_files=raw,
        standalone_selected_model_runtime_not_measured_here=True,supported_innovations=0,goal_met=False,
        seconds=time.monotonic()-started)
    json_path=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261127_FISHER_FULL_TEST.json'
    assert not json_path.exists()
    json_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');files.append(str(json_path.relative_to(ROOT)))
    md='# Dev冻结的配对Fisher方案：完整七段Test\n\n'
    md+=f"七个原始视频15985帧全部完成；原控制器及18个工程、Dev、Test子进程实际退出0。独立读取原276列官方CSV并核对全部47项及每视频/COMBINED，共{result['independent_numeric_fields_verified']}项数值。原始RGB CSV保持原字节，零权重逐视频MOT与旧IR父模型完全一致。\n\n"
    md+='| 条件 | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for row in [canonical,fresh,old_parent,selected]:
        m=row['combined47'];md+='| '+row['id']+' | '+' | '.join(str(m[k])if k=='IDSW'else f'{m[k]:.6f}'for k in ['HOTA','DetA','AssA','IDF1','MOTA','IDSW'])+' |\n'
    md+=f"\n原完整Dev预选方案 `{selected['id']}` 对原始RGB-only YOLOX-X+ByteTrack的完整Test HOTA为{delta:+.6f}个绝对点，对相同视觉IR父模型为{result['HOTA_gain_over_visual_parent_pp']:+.6f}点。+5目标{'达到'if delta>=5 else '未达到'}，+10目标{'达到'if delta>=10 else '未达到'}。数值阈值与独立模块贡献分开核验；此处独立创新计数仍0，研究Goal未宣布完成。\n\n"
    md+='| 视频 | 所选方案 | 原始RGB | 同视觉IR父模型 | 对原RGB增量 | 对视觉父模型增量 |\n|---|---:|---:|---:|---:|---:|\n'
    for x in per_video:md+='| '+x['sequence']+' | '+' | '.join(f'{x[k]:.6f}'for k in ['selected_HOTA','original_RGB_HOTA','visual_parent_HOTA','gain_over_original_RGB_pp','gain_over_visual_parent_pp'])+' |\n'
    md+='\n全部条件、负结果、47项联合和逐视频值与原件见JSON。不能累加不同视频或方法的提升；COMBINED来自官方聚合。方案只按完整Dev预选，未按本次Test改选权重或时长。原始历史Test暴露、RGB预训练见过当前Dev、伪红外由RGB派生等局限保持。\n\n'
    md+='Fisher/LDA、余弦/EMA与延长身份保留是经典工具，组合提升不等于红外或任一单模块的独立贡献。当前只验证相同固定伪红外视觉父模型下的组合，独立去除、错位/缺失等归因证据仍需另行完成。共享完整序列生产耗时不是所选单模型端到端FPS，本工具未测所选模型的独立运行成本。不上传权重、图像、MOT或特征数组。\n'
    md_path=json_path.with_suffix('.md');md_path.write_text(md);files.append(str(md_path.relative_to(ROOT)))
    files.append(str(Path(__file__).relative_to(ROOT)))
    for name in files:
        p=ROOT/name;p.read_text();assert p.stat().st_size<2*2**20 and b'\0'not in p.read_bytes()
    allow=work/'fisher_full_Test_publication_allowlist_20261127_r1.json'
    allow.write_text(json.dumps(dict(files=files,hashes={n:sha256(ROOT/n)for n in files},total_bytes=sum((ROOT/n).stat().st_size for n in files)),indent=2)+'\n')
    receipt=dict(status=result['status'],UTC=utc(),HOTA_gain_over_original_RGB_pp=delta,
        report_sha256=sha256(json_path),original_controller_actual_exit_code=0,original_workers_exit_zero=18,
        independent_numeric_fields_verified=result['independent_numeric_fields_verified'],goal_met=False,supported_innovations=0)
    terminal.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
