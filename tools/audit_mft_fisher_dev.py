#!/usr/bin/env python3
"""Independent CPU audit/export of complete five-case paired Fisher Dev.

Waits on the specific original controller; never starts models or changes the
selection. The primary controller may continue its Dev-frozen Test afterwards.
"""
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.fusion_memory_evaluation import official_csv47, validate_mot
from tools.fusion_memory_observation import LENGTHS
from tools.run_mft_fisher_memory import PREFIX, SPEC, identity, locations, registered
from tools.run_pseudo_ir_comparison import sha256


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def wait_for_complete(base):
    expected = read(base/'controller.execution.json')['identity']
    while not (base/'complete_Dev.json').exists():
        execution = read(base/'controller.execution.json')
        assert execution['identity'] == expected, 'Different controller attempt'
        state = read(base/'state.json')
        current = identity(expected['PID'])
        matches = current is not None and all(current[k] == expected[k] for k in ['PID','start_ticks','raw_argv_sha256'])
        if execution.get('actual_child_exit_code') is not None or not matches or state['status'].startswith('FAILED'):
            raise RuntimeError('Original Fisher attempt ended before complete Dev; preserve actual failure and outputs.')
        time.sleep(15)
    return read(base/'complete_Dev.json')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    started = time.monotonic()
    plan, study, protocol, calibration, work, checkpoint, base = locations()
    assert not (base/'independent_Dev_audit_complete.json').exists(), 'Do not overwrite an actual audit'
    registration = registered()
    complete = wait_for_complete(base)
    assert complete['status'] == 'COMPLETE_ALL5_PAIRED_FISHER_DEV'
    assert complete['Test_read_for_selection'] is False
    state = read(base/'state.json')
    phases = [PREFIX+'_GPU_ENGINEERING',PREFIX+'_DEV_NN',PREFIX+'_DEV_OFFICIAL']
    jobs = [j for phase in phases for j in state['phases'][phase].values()]
    assert len(jobs) == 9
    for job in jobs:
        assert job['status'] == 'COMPLETE' and type(job.get('exit_code')) is int and job['exit_code'] == 0
        assert not identity(job['pid']), 'A formal Dev worker is still live'
    rows = complete['rows']
    assert [r['case'] for r in rows] == registration['normal_cases']
    zero = rows[0]
    assert zero['case']['appearance_weight'] == 0
    expected = read(work/'native_memory_priority_20261126_v3/complete_Dev.json')['rows'][0]
    assert zero['combined47'] == expected['combined47']
    assert zero['per_sequence47'] == expected['per_sequence47']
    output = ROOT/'docs/fusion_research/perceptual_20261127_fisher_full_Dev'
    output.mkdir(exist_ok=True)
    files, raw_files = [], []
    def copy_file(source, relative):
        source = Path(source)
        destination = output/relative
        assert not destination.exists(), 'Preserve an earlier export: '+str(destination)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
        name = str(destination.relative_to(ROOT))
        files.append(name)
        raw_files.append(dict(file=name,sha256=sha256(destination),source=str(source),source_sha256=sha256(source)))
        return name
    for name in ['complete_Dev.json','pipeline_registered.json','specifications/dev.json',
                 'fit_complete.json','gpu_engineering_4.json','gpu_engineering_5.json']:
        copy_file(base/name,name)
    for seq in LENGTHS['dev']:
        copy_file(base/'streams/dev/sequence_manifests'/(seq+'.json'),'producer_manifests/'+seq+'.json')
        copy_file(base/'producer_validation/dev'/(seq+'.json'),'producer_validation/'+seq+'.json')
    checked, numerical_fields = [], 0
    for row in rows:
        case = row['case']['id']
        original_csv = Path(row['official_CSV'])
        assert sha256(original_csv) == row['official_CSV_sha256']
        parsed = official_csv47(original_csv,row['official_schema'],'dev',row['metrics39'])
        assert parsed['combined47'] == row['combined47'] and parsed['per_sequence47'] == row['per_sequence47']
        numerical_fields += 3*47
        csv_file = copy_file(original_csv,'official/'+case+'.csv')
        copy_file(base/'evaluation/dev'/case/'metrics.json','official/'+case+'.json')
        coverage = {}
        for seq,count in LENGTHS['dev'].items():
            source = read(base/'streams/dev/sequence_manifests'/(seq+'.json'))
            stream = source['streams'][case]
            original_source = read(work/'native_pyramid_streams/dev_normal_06/sequence_manifests'/(seq+'.json'))
            original = original_source['streams']['NP_temporal_ir_e06']
            assert source['frames'] == count and source['GT_read'] is False and source['future_frame_input'] is False
            assert source['cases'] == registration['normal_cases']
            assert source['provenance']['source_versions'] == registration['source_hashes']
            assert source['original_image_bytes_delivery_sha256'] == original_source['original_image_bytes_delivery_sha256']
            assert stream['detection_packet_sha256'] == original['detection_packet_sha256']
            track_path = base/'streams/dev'/case/'track_results'/(seq+'.txt')
            assert sha256(track_path) == stream['track_sha256']
            validate_mot(track_path,count,stream['track_rows'])
            literal_exact = track_path.read_bytes() == (base/'streams/dev'/zero['case']['id']/'track_results'/(seq+'.txt')).read_bytes()
            if row['case']['appearance_weight'] == 0:
                assert stream['track_sha256'] == original['track_sha256']
            coverage[seq] = dict(frames=count,track_rows=stream['track_rows'],track_sha256=stream['track_sha256'],
                literal_MOT_exact_to_zero=literal_exact,detector_packet_sha256=stream['detection_packet_sha256'],
                tracker_statistics=source['memory_statistics'][case])
        checked.append(dict(id=case,case=row['case'],combined47=parsed['combined47'],per_sequence47=parsed['per_sequence47'],
            HOTA_gain_over_identical_parent_pp=parsed['combined47']['HOTA']-zero['combined47']['HOTA'],
            official_CSV=csv_file,official_CSV_sha256=sha256(original_csv),coverage=coverage))
    chosen = min(rows,key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),r['case']['lost_buffer'],r['case']['appearance_weight']))
    assert chosen['case'] == complete['chosen_case']
    gain = chosen['combined47']['HOTA']-zero['combined47']['HOTA']
    assert gain == complete['Dev_gain_over_identical_visual_parent_pp']
    original_Dev = read(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261126_NATIVE_STAGE9_FULL_DEV.json')
    canonical = next(r for r in original_Dev['rows'] if r['id'] == 'NP_source_canonical_RGB')
    canonical_csv = ROOT/canonical['official_CSV']['file']
    assert sha256(canonical_csv) == canonical['official_CSV']['sha256']
    original_values = official_csv47(canonical_csv,original_Dev['official_fields'],'dev',canonical['original_official_metrics39'])
    assert original_values['combined47'] == canonical['combined47']
    current = identity(read(base/'controller.execution.json')['identity']['PID'])
    fit = read(base/'fit_complete.json')
    report = dict(status='INDEPENDENTLY_VERIFIED_COMPLETE_ALL5_FISHER_DEV',UTC=now(),
        rows=checked,chosen_case=chosen['case'],Dev_gain_over_identical_parent_pp=gain,
        original_direct_RGB_Dev_HOTA=original_values['combined47']['HOTA'],Dev_HOTA_parent=zero['combined47']['HOTA'],
        original_direct_RGB_Dev_CSV_sha256=sha256(canonical_csv),
        actual_required_Dev_related_workers_exit_zero=9,official_numeric_fields_independently_verified=numerical_fields,
        all_cases_same_native_detector_packets=True,zero_MOT_exact_to_original_parent=True,
        all_Train_fit_provenance_unchanged=True,Train_fit_details=fit['fit_details'],
        fit_source_sha256=sha256(base/'fit_complete.json'),complete_Dev_sha256=sha256(base/'complete_Dev.json'),
        pipeline_registration_sha256=sha256(base/'pipeline_registered.json'),source_files_unchanged=len(registration['source_hashes']),
        original_Controller_still_live=current is not None,original_controller_execution_snapshot=read(base/'controller.execution.json'),
        current_phase_snapshot=state['status'],Dev_phases_actual_jobs={p:state['phases'][p]for p in phases},
        Test_results_read=False,Test_GT_or_accuracy_used_for_selection=False,historical_Test_previously_exposed=True,
        current_prior_primary_Test_HOTA=29.799525425926028,original_RGB_Test_HOTA=29.528970687393972,
        current_prior_primary_Test_gain_pp=.2705547385320557,minimum_target_Test_HOTA=34.52897068739397,
        stretch_target_Test_HOTA=39.52897068739397,extra_seed_work=False,new_NN_training=False,
        physical_gpus=['4','5'],maximum_cards=2,supported_innovations=0,goal_met=False,
        raw_files=raw_files,seconds=time.monotonic()-started)
    if gain > 0 and chosen['case']['appearance_weight']:
        # Publication follows the original decision; it never makes a new choice.
        freeze_path = base/'Test_selection_frozen.json'
        while not freeze_path.exists():
            execution = read(base/'controller.execution.json')
            if execution.get('actual_child_exit_code') is not None or not identity(execution['identity']['PID']):
                raise RuntimeError('Positive Dev but original Test freeze missing; preserve attempt')
            time.sleep(2)
        freeze = read(freeze_path)
        assert freeze['chosen_case'] == chosen['case'] and freeze['Test_read_for_selection'] is False
        assert freeze['complete_Dev_sha256'] == sha256(base/'complete_Dev.json')
        report['Test_selection_frozen_sha256'] = sha256(freeze_path)
        report['positive_Dev_promoted_by_original_driver'] = True
        copy_file(freeze_path,'Test_selection_frozen.json')
    else:
        assert not (base/'Test_selection_frozen.json').exists()
        report['positive_Dev_promoted_by_original_driver'] = False
    registered()
    json_path = ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261127_FISHER_FULL_DEV.json'
    assert not json_path.exists()
    json_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    files.append(str(json_path.relative_to(ROOT)))
    md = '# 配对 Fisher 外观与身份保留：完整 Dev\n\n'
    md += f'五组条件均完成原始两视频4834帧；九个工程、推理及官方评估子进程实际退出0。独立核验全部47项、每视频及COMBINED共{numerical_fields}项字段，零权重MOT精确复现视觉父模型。所有条件检测包相同。\n\n'
    md += '| 条件 | 外观权重 | 保留帧数 | HOTA | Δ同视觉父模型 | AssA | IDF1 | IDSW |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for row in checked:
        m,c=row['combined47'],row['case']
        md+=f"| {row['id']} | {c['appearance_weight']} | {c['lost_buffer']} | {m['HOTA']:.6f} | {row['HOTA_gain_over_identical_parent_pp']:+.6f} | {m['AssA']:.6f} | {m['IDF1']:.6f} | {m['IDSW']} |\n"
    md += f"\n按预先固定的HOTA、AssA、IDF1、较短保留时长及较小权重顺序，选择 `{chosen['case']['id']}`，相对相同新版伪红外视觉父模型的 Dev HOTA 为{gain:+.6f}个绝对点。"
    md += ('原控制器已冻结该方案进入完整Test；本报告未读取新的Test准确率，Dev改善不能视为Test已达到+5。' if gain>0 else '没有正向Dev改善，未启动新Test，全部负结果保留。')
    md += '\n\n外观投影仅拟合六个Train视频采样1152帧中的94组身份、17266条有效ROI；三组稀疏采样身份的4条观测排除。Fisher/LDA、EMA和长记忆属于经典工具，不能仅因实现而计为原创贡献。当前试验同时改变外观代价和部分条件的保留时长，不能把组合提升全部归因于其中一个模块。\n\n'
    md += '统计 `actual_float_cost_entries_changed` 才计实际浮点代价变化，`frames_with_changed_primary_assignment` 计原始首轮匹配集合变化，均不等同于正确身份改善；逐视频MOT是否变化及所有47项见JSON。保持检测包相同能定位到关联行为，不能自动证明红外本身或某个单模块的独立贡献。\n\n'
    md += '固定比较基线仍为原始RGB-only YOLOX-X+ByteTrack完整Test29.528971，最低目标34.528971，进一步39.528971。本轮之前的native IR完整Test29.799525，仅+0.270555点；本报告不改选历史Test模型。历史Test暴露、原始RGB训练见过当前Dev以及伪红外由RGB派生的局限继续适用。无新随机种子、额外RGB结构、图像/FPN副本，所有权重和MOT只保留本机。Goal仍active，独立创新计数0。\n'
    md_path=json_path.with_suffix('.md');md_path.write_text(md);files.append(str(md_path.relative_to(ROOT)))
    files.append(str(Path(__file__).relative_to(ROOT)))
    for name in files:
        p=ROOT/name;p.read_text();assert p.stat().st_size < 2*2**20 and b'\0' not in p.read_bytes()
    receipt=dict(status=report['status'],UTC=now(),actual_Dev_workers=9,fields_verified=numerical_fields,
        report_sha256=sha256(json_path),source_sha256=sha256(Path(__file__)),Test_accuracy_read=False,
        selected_case=chosen['case']['id'],Dev_gain_pp=gain)
    (base/'independent_Dev_audit_complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
    allow=work/'fisher_full_Dev_publication_allowlist_20261127_r1.json'
    allow.write_text(json.dumps(dict(files=files,hashes={n:sha256(ROOT/n)for n in files},
        total_bytes=sum((ROOT/n).stat().st_size for n in files)),indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
