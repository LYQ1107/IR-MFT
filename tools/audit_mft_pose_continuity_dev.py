#!/usr/bin/env python3
"""Read-only independent full-Dev CSV/MOT audit and lightweight publication."""
import datetime
import hashlib
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
from tools.native_pyramid_experiment import read
from tools.run_mft_pose_continuity import locations, registered, policies, identity_score_hash
from tools.run_mft_native_memory_priority_v3 import identity
from tools.run_pseudo_ir_comparison import sha256


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plan, study, protocol, calibration, work, checkpoint, base, peer = locations()
    receipt = base / 'independent_Dev_audit_complete.json'
    assert not receipt.exists()
    while not (base / 'complete_Dev.json').exists():
        execution = read(base / 'controller.execution.json')
        assert execution.get('actual_child_exit_code') is None, 'Preserve failed original attempt'
        time.sleep(10)
    reg = registered()
    dev = read(base / 'complete_Dev.json')
    assert dev['Test_read_for_selection'] is False
    assert [r['case'] for r in dev['rows']] == policies()
    jobs = [job for phase, rows in read(base / 'state.json')['phases'].items()
            if phase in ['POSE_CONTINUITY_20261127_DEV_REPLAY', 'POSE_CONTINUITY_20261127_DEV_OFFICIAL']
            for job in rows.values()]
    assert len(jobs) == 10
    for job in jobs:
        assert job['status'] == 'COMPLETE' and job['exit_code'] == 0 and type(job['exit_code']) is int
        assert job['argv_identity_observed'] and identity(job['pid']) is None
    schema, _ = official_schema(plan['evaluation']['TrackEval_root'])
    references = []
    numeric_fields = 0
    for name in ['NP_source_canonical_RGB', 'NP_source_fresh_RGB', 'NP_temporal_ir_e06']:
        folder = work / 'native_pyramid_evaluation/dev' / name
        csv = folder / 'trackeval_output/pedestrian_detailed.csv'
        values = official_csv47(csv, schema, 'dev', read(folder / 'metrics.json')['trackeval'])
        numeric_fields += len(values['per_sequence47']) * 47
        references.append(dict(id=name, official_CSV_source=str(csv), **values))
    out = ROOT / 'docs/fusion_research/perceptual_20261127_pose_continuity_full_dev'
    assert not out.exists()
    out.mkdir()
    files, raw = [], []

    def copy(source, name):
        source = Path(source)
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        rel = str(dest.relative_to(ROOT))
        files.append(rel)
        raw.append(dict(file=rel, sha256=sha256(dest), source=str(source), source_sha256=sha256(source)))
        return rel

    copy(base / 'complete_Dev.json', 'complete_Dev.json')
    copy(base / 'pipeline_registered.json', 'pipeline_registered.json')
    copy(base / 'controller.execution.json', 'controller_execution_snapshot.json')
    copy(base / 'state.json', 'state_snapshot.json')
    manifests = {}
    for sequence, count in LENGTHS['dev'].items():
        path = base / 'streams/dev/sequence_manifests' / (sequence + '.json')
        record = read(path)
        assert record['frames'] == count and record['source_versions'] == reg['source_hashes']
        assert record['GT_read'] is False and record['future_frame_input'] is False
        assert record['previous_outputs_rewritten'] is False
        assert record['compact_feature_cache_read'] and record['only_current_frame_values_supplied']
        assert record['future_suffix_corruption_prefix_invariant'] and record['scoped_prediction_filter_restored']
        parent = read(peer / 'streams/dev/sequence_manifests' / (sequence + '.json'))
        for own, ref in [('PC_native_zero', 'TP_native_zero'), ('PC_late_only', 'TP_late_RGB')]:
            assert record['streams'][own]['track_sha256'] == parent['streams'][ref]['track_sha256']
        assert record['bank']['bank_sha256'] == reg['Dev_banks'][sequence]['bank_sha256']
        assert record['bank']['packet_sha256'] == reg['Dev_banks'][sequence]['packet_sha256']
        copy(path, 'replay_manifests/' + sequence + '.json')
        manifests[sequence] = record
    rows = []
    for row in dev['rows']:
        csv = Path(row['official_CSV'])
        assert sha256(csv) == row['official_CSV_sha256']
        values = official_csv47(csv, schema, 'dev', row['metrics39'])
        numeric_fields += len(values['per_sequence47']) * 47
        assert values['combined47'] == row['combined47'] and values['per_sequence47'] == row['per_sequence47']
        assert row['frames'] == 4834 and row['source_versions'] == reg['source_hashes']
        coverage = {}
        for sequence, record in manifests.items():
            stream = record['streams'][row['case']['id']]
            assert sha256(stream['track_path']) == stream['track_sha256']
            assert stream['detection_packet_sha256'] == reg['Dev_banks'][sequence]['packet_sha256']
            validate_mot(stream['track_path'], LENGTHS['dev'][sequence], stream['track_rows'])
            detail={}
            if row['case']['render']!='filtered':
                proof=record['anchor_frame_ID_score_order_exact_proofs'][row['case']['id']]
                old=read(peer/'streams/dev/sequence_manifests'/(sequence+'.json'))
                ref=old['streams']['TP_late_RGB'if row['case']['late']else'TP_native_zero']
                actual=identity_score_hash(stream['track_path']);expected=identity_score_hash(ref['track_path'])
                assert actual==expected==proof['actual_frame_ID_score_order_hash']==proof['reference_hash']
                detail=dict(actual_frame_ID_score_order_hash=actual,reference_hash=expected)
            coverage[sequence] = dict(track_sha256=stream['track_sha256'], track_rows=stream['track_rows'],
                                     statistics=record['statistics'][row['case']['id']],**detail)
        official = copy(csv, 'official/' + row['case']['id'] + '.csv')
        copy(base / 'evaluation/dev' / row['case']['id'] / 'metrics.json', 'official/' + row['case']['id'] + '.json')
        rows.append({**values, 'case': row['case'], 'coverage': coverage,
                     'official_CSV': official, 'official_CSV_source': str(csv)})
    for row in references:
        row['official_CSV'] = copy(row['official_CSV_source'], 'official/' + row['id'] + '.csv')
    assert numeric_fields == 1551
    chosen = min([r for r in rows if r['case'] in policies()], key=lambda r: (*(-r['combined47'][k] for k in ['HOTA', 'AssA', 'IDF1']),
                                       policies().index(r['case'])))
    assert chosen['case'] == dev['chosen_case']
    gain = chosen['combined47']['HOTA'] - rows[1]['combined47']['HOTA']
    assert gain == dev['gain_over_late_only_pp']
    assert rows[0]['combined47'] == references[2]['combined47']
    assert rows[0]['per_sequence47'] == references[2]['per_sequence47']
    if gain > 0:
        freeze = read(base / 'Test_selection_frozen.json')
        assert freeze['chosen_case'] == chosen['case'] and freeze['Test_read_for_selection'] is False
        assert freeze['complete_Dev_sha256'] == sha256(base / 'complete_Dev.json')
        copy(base / 'Test_selection_frozen.json', 'Test_selection_frozen.json')
    else:
        assert not (base / 'Test_selection_frozen.json').exists()
    result = dict(status='INDEPENDENTLY_VERIFIED_ALL8_POSE_CONTINUITY_COMPLETE_DEV',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows=rows, references=references,
        auditor_source_sha256=sha256(Path(__file__)),
        chosen_case=chosen['case'], Dev_gain_over_late_only_pp=gain,
        selected_Dev_HOTA=chosen['combined47']['HOTA'],
        selected_Dev_gain_over_original_RGB_Dev_pp=chosen['combined47']['HOTA']-references[0]['combined47']['HOTA'],
        complete_frames=4834, original_Dev_related_workers_actual_exit_zero=10,
        official_numeric_fields_checked=numeric_fields, source_versions_unchanged=len(reg['source_hashes']),
        zero_and_late_only_all47_and_actual_MOT_exact=True, scoped_prediction_filter_restored_in_real_replays=True,
        Test_accuracy_parsed=False, Test_used_for_selection=False,
        conditional_new_Test_frozen=(gain > 0), original_R2_GPU_experiment_changed=False,
        CPU_only=True, new_NN_fits=0, new_seeds=0, fixed_seed=0,
        historical_Test_previously_exposed=True, RGB_pretraining_included_current_Dev=True,
        Train_identity_encoder_seen_threshold_calibration_IDs=True, IR_generated_from_RGB=True,
        motion_scope='Aspect-only covariance uses the unchanged Train sigma and retains original center/height covariance. Shape/raw rendering anchors are output-only and retain original filter/association states.',
        matching_scope='Aspect Q can change shape costs and assignments. Output-only anchors retain literal frame/ID/score/order/row counts versus their original or late-only references.',
        counter_scope='Aspect update frames and numeric output-box changes count activations, not correct matches. Legacy primary-hook counters are unused and cannot measure these effects.',
        current_R2_Test_previously_reviewed_by_agent=True, hypothesis_authored_before_agent_R2_Test_review=True,
        implementation_after_agent_R2_Test_review=True, Test_accuracy_scope='Only new pose Test; historical and current earlier R2 benchmark exposure is explicitly retained.',
        runtime_scope='CPU cache replay excludes image IO, RGB/IR encoders, detector and offline IR generation; not end-to-end FPS.',
        original_RGB_Test_HOTA=29.528970687393972, required_Test_HOTA=34.52897068739397,
        goal_met=False, supported_innovations=0, raw_files=raw)
    report = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261127_POSE_CONTINUITY_FULL_DEV.json'
    assert not report.exists()
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    files.append(str(report.relative_to(ROOT)))
    md = '# 当前形状与身份连续性：完整Dev结果\n\n'
    md += '两段完整4834帧；10个相关实验子进程全部实际退出0。独立解析原276列CSV的COMBINED与逐视频，核对1551个官方字段；零路径和late-only的47字段及实际MOT字节均验证。\n\n'
    md += '| 条件 | HOTA | 对late-only | DetA | AssA | IDF1 | MOTA | IDSW | 宽高比Q更新帧数 | late续接次数 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for row in rows:
        m=row['combined47'];changed=sum(x['statistics']['aspect_calibrated_update_frames'] for x in row['coverage'].values())
        ir=sum(x['statistics']['identity_links'] for x in row['coverage'].values())
        md += f"| {row['case']['id']} | {m['HOTA']:.6f} | {m['HOTA']-rows[1]['combined47']['HOTA']:+.6f} | {m['DetA']:.6f} | {m['AssA']:.6f} | {m['IDF1']:.6f} | {m['MOTA']:.6f} | {m['IDSW']} | {changed} | {ir} |\n"
    md += f"\n完整Dev预选`{chosen['case']['id']}`，HOTA{chosen['combined47']['HOTA']:.6f}，比late-only{gain:+.6f}。"
    md += ('已按预先规则冻结一个新完整Test策略，只复用原R2完成的七段银行，不增加GPU提取。' if gain>0 else '没有超越已存在的late-only方案，不追加新Test，保留全部负向条件。')
    md += '\n\n表为Dev，不能与原始RGB Test29.528971跨划分相减或声称达到+5。所有策略保持同一固定新版IR检测器。本实验没有IR身份描述符干预；所有方案仍用同一新版IR视觉检测器。仅宽高比Q会影响后续几何匹配；输出形状和原始框锚定逐视频保持frame/ID/score/order/行数的真实文件哈希。更新帧数和续接次数是激活记录，不能当成正确匹配或独立红外贡献。\n\n'
    md += '宽高比Q只复用原Train sigma_aspect0.029661453082623204，没有新拟合；包括标注噪声和插值限制。当前形状/框来自同帧检测，不把输出锚定反馈给内部滤波器，不改写历史。八项假设在agent首次阅读当前R2 Test前写定，实现与本次Dev在该阅读之后；历史Test暴露继续披露，不能声称全新盲测。新方案选型只读取全部Dev官方指标。CPU回放排除图像、编码器与检测器成本，不是端到端速度。历史Test暴露、原RGB预训练见过当前Dev以及IR由RGB派生等限制保留。所有47字段、逐视频、原CSV和实际执行来源在JSON及附属目录；不上传MOT、银行、图像、权重。Goal继续active，独立贡献计数0。\n'
    report.with_suffix('.md').write_text(md)
    files.extend([str(report.with_suffix('.md').relative_to(ROOT)), str(Path(__file__).relative_to(ROOT))])
    for name in files:
        p=ROOT/name;p.read_text();assert p.stat().st_size<2*2**20 and b'\0'not in p.read_bytes()
    allow=dict(files=files, hashes={n:sha256(ROOT/n) for n in files}, total_bytes=sum((ROOT/n).stat().st_size for n in files))
    (work/'pose_continuity_full_Dev_publication_allowlist_20261127_r1.json').write_text(json.dumps(allow, indent=2)+'\n')
    final=dict(status=result['status'], UTC=result['UTC'], selected_case=chosen['case'],
        HOTA=chosen['combined47']['HOTA'], gain_over_late_only_pp=gain,
        original_workers_actual_exit_zero=10, official_numeric_fields_checked=numeric_fields,
        new_Test_frozen=(gain>0), Test_accuracy_parsed=False, report_sha256=sha256(report), goal_met=False)
    receipt.write_text(json.dumps(final, indent=2)+'\n')
    print(json.dumps(final), flush=True)


if __name__ == '__main__':
    main()
