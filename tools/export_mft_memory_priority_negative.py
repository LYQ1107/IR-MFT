#!/usr/bin/env python3
"""Audit/publish the completed zero-gain native-memory Dev experiment."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.fusion_memory_evaluation import official_csv47
from tools.run_mft_native_memory_priority_v3 import identity, registered
from tools.run_pseudo_ir_comparison import sha256


def read(path):
    return json.loads(Path(path).read_text())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    work = Path('/var/tmp/liuyeqiang_ir_mft_perceptual_20261122')
    base = work/'native_memory_priority_20261126_v3'
    reg = registered()
    state, execution = read(base/'state.json'), read(base/'controller.execution.json')
    assert state['status'] == 'COMPLETE_NATIVE_MEMORY_NO_POSITIVE_DEV_GAIN_NO_NEW_TEST'
    assert execution['actual_child_exit_code'] == 0 and not identity(execution['identity']['PID'])
    jobs = [j for phase in state['phases'].values() for j in phase.values()]
    assert len(jobs) == 7 and all(j['status']=='COMPLETE' and type(j['exit_code']) is int and
                                 j['exit_code']==0 and not identity(j['pid']) for j in jobs)
    complete = read(base/'complete_Dev.json')
    rows = complete['rows']
    assert len(rows) == 3 and complete['Dev_gain_over_identical_visual_parent_pp'] == 0
    assert complete['chosen_case']['appearance_weight'] == 0
    assert not (base/'Test_selection_frozen.json').exists()
    output = ROOT/'docs/fusion_research/perceptual_20261127_native_memory_no_gain'
    output.mkdir(exist_ok=True)
    copied = []
    def copy_file(source, name):
        source = Path(source)
        destination = output/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert destination.read_bytes() == source.read_bytes()
        else:
            shutil.copyfile(source, destination)
        copied.append(dict(file=str(destination.relative_to(ROOT)),sha256=sha256(destination),bytes=destination.stat().st_size,
                           source=str(source),source_sha256=sha256(source)))
        return str(destination.relative_to(ROOT))
    for name in ['complete_Dev.json','state.json','pipeline_registered.json','controller.execution.json',
                 'gpu_engineering_4.json','gpu_engineering_5.json','CPU_engineering_r1.json',
                 'CPU_engineering_r1.execution.json','real_original_Dev_CSV_layout_regression_r1.json']:
        copy_file(base/name,name)
    parent = rows[0]
    table, literal = [], {}
    for row in rows:
        case = row['case']['id']
        csv_path = Path(row['official_CSV'])
        assert sha256(csv_path) == row['official_CSV_sha256']
        parsed = official_csv47(csv_path,row['official_schema'],'dev',row['metrics39'])
        assert parsed['combined47'] == row['combined47'] == parent['combined47']
        assert parsed['per_sequence47'] == row['per_sequence47'] == parent['per_sequence47']
        csv_public = copy_file(csv_path,'official/'+case+'.csv')
        copy_file(base/'evaluation/dev'/case/'metrics.json','official/'+case+'.json')
        coverage = {}
        for seq in ['BT-005','SN-013']:
            track = base/'streams/dev'/case/'track_results'/(seq+'.txt')
            actual = sha256(track)
            expected = row['manifest']['coverage'][seq]
            assert actual == expected['track_sha256'] == parent['manifest']['coverage'][seq]['track_sha256']
            assert expected['detection_packet_sha256'] == parent['manifest']['coverage'][seq]['detection_packet_sha256']
            if seq not in literal:
                literal[seq] = track.read_bytes()
            assert literal[seq] == track.read_bytes(), 'All3 actual MOT files must be byte-identical'
            coverage[seq] = dict(frames=expected['frames'],track_sha256=actual,track_rows=expected['track_rows'],
                detector_packet_sha256=expected['detection_packet_sha256'],literal_MOT_exact_to_zero=True)
        table.append(dict(id=case,appearance_weight=row['case']['appearance_weight'],combined47=row['combined47'],
            per_sequence47=row['per_sequence47'],official_CSV=csv_public,official_CSV_sha256=sha256(csv_path),coverage=coverage))
    statistics = {}
    for seq in ['BT-005','SN-013']:
        p=base/'streams/dev/sequence_manifests'/(seq+'.json')
        statistics[seq] = read(p)['memory_statistics']
        copy_file(p,'producer_manifests/'+seq+'.json')
        copy_file(base/'producer_validation/dev'/(seq+'.json'),'producer_validation/'+seq+'.json')
    runtime_path = work/'native_pyramid_individual_runtime.json'
    runtime, runtime_execution = read(runtime_path), read(work/'native_primary_individual_runtime_r1.execution.json')
    assert runtime_execution['actual_child_exit_code'] == 0 and not identity(runtime_execution['identity']['PID'])
    assert runtime['status'] == 'COMPLETE_NATIVE_INDIVIDUAL_RUNTIME'
    copy_file(runtime_path,'native_pyramid_individual_runtime.json')
    copy_file(work/'native_primary_individual_runtime_r1.execution.json','native_primary_individual_runtime_r1.execution.json')
    primary = read(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261126_NATIVE_PRIMARY_FULL_TEST.json')
    report = dict(status='COMPLETE_NATIVE_MEMORY_ALL3_DEV_IDENTICAL_NO_NEW_TEST',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=table,
        source_registration_sha256=sha256(base/'pipeline_registered.json'),actual_original_workers_exit_zero=7,
        actual_controller_exit_zero=True,all47_combined_and_per_sequence_metrics_identical=True,
        actual_MOT_byte_identical_all3_cases_both_complete_Dev_videos=True,
        selection=complete['chosen_case'],Dev_gain_pp=0.,new_Test_launched=False,
        memory_formula_term_statistics=statistics,
        counter_interpretation='Old changed_edges counts positive formula penalties only; it does not count changed float matrix entries or changed assignments. All emitted MOT bytes are identical here.',
        runtime=runtime,runtime_scope='Original existing single-condition prefix benchmark:7videos x32inputs,8warmup/24measured per video.168 timed frames per condition; not full15985-frame FPS. Fresh B1 RGB is the timing comparator, not canonical cached B4. Offline IR generation and checkpoint loading excluded.',
        current_primary_Test_report_sha256=sha256(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261126_NATIVE_PRIMARY_FULL_TEST.json'),
        current_Test_HOTA=29.799525425926028,original_canonical_RGB_HOTA=29.528970687393972,
        current_Test_gain_pp=29.799525425926028-29.528970687393972,
        plus5_target_HOTA=34.52897068739397,plus10_target_HOTA=39.52897068739397,
        Test_read_for_current_Dev_selection=False,historical_Test_previously_exposed=True,
        extra_RGB_architecture_work=False,new_NN_training=False,fixed_seed=0,
        physical_gpus=['4','5'],source_files_unchanged=True,raw_files=copied,
        supported_innovations=0,goal_met=False)
    json_path = ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261127_NATIVE_MEMORY_NO_GAIN.json'
    json_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    md = '''# 固定新版伪红外：外观记忆无增益的完整 Dev 结果

三组权重0、0.05、0.10均完成原始 Dev 两视频4834帧。HOTA均为 **50.115210**；全部47项联合及逐视频指标相同，六份实际 MOT 文件在同视频内逐字节一致。按照预注册的并列规则选择权重0，没有启动新的 Test。全部7个工程、推理、官方评估子进程及控制器实际退出0。

| 外观权重 | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
|---:|---:|---:|---:|---:|---:|---:|
'''
    for row in table:
        m=row['combined47'];md+='| '+str(row['appearance_weight'])+' | '+' | '.join(f'{m[k]:.6f}' if k!='IDSW' else str(m[k]) for k in ['HOTA','DetA','AssA','IDF1','MOTA','IDSW'])+' |\n'
    md+='''
该实验使用冻结的 `NP_temporal_ir_e06`、固定11月 DiffV2IR PNG 和现有 F4 余弦惩罚，未更换 RGB 网络、增加种子或修改检测结果。旧实现中 `changed_edges` 统计的是公式惩罚为正的项，并不证明浮点代价实际变化，更不证明匹配发生变化；本次输出完全一致，不能宣称外观记忆有效。

单条件运行成本在物理 GPU4 上实际测量：每个模型读取7个 Test 视频各前32帧，其中每视频前8帧预热，后24帧计时，总计168个计时帧。fresh B1 RGB 为11.619 FPS，native IR为2.744 FPS；包含实际图像读取、预处理、网络、融合、光流、NMS与 ByteTrack，不含离线伪红外生成和权重加载。这是前缀成本抽样，不是完整15985帧速度，也不是原始缓存 B4 的速度测量。

当前冻结主方法完整 Test 仍为29.799525，对原始直接 RGB-only baseline 29.528971提升0.270555个绝对点，**未达到+5**。最低目标34.528971，进一步目标39.528971。支持的独立创新点仍为0。

下一轮保留同一新版红外和视觉检测器，使用 Train 身份标签学习经典正则 Fisher 外观投影，比较固定的外观权重与身份保留时长；先完整 Dev 选择，再决定是否进入完整 Test。不追加 RGB 网络结构或随机种子搜索。Train 拟合诊断不代表跟踪 HOTA 提升。

全部47项、逐视频指标、原始276列官方 CSV、实际进程记录和负结果见同名 JSON 及 `perceptual_20261127_native_memory_no_gain/`。权重、图像、FPN、MOT 文件只保留在本机。
'''
    md_path=json_path.with_suffix('.md');md_path.write_text(md)
    files=[str(json_path.relative_to(ROOT)),str(md_path.relative_to(ROOT))]+[x['file']for x in copied]+['tools/export_mft_memory_priority_negative.py']
    for name in files:
        p=ROOT/name
        assert p.stat().st_size<2*2**20 and b'\0' not in p.read_bytes()
        p.read_text()
    allowlist=work/'memory_negative_publication_allowlist_20261127_r1.json'
    allowlist.write_text(json.dumps(dict(status='AUDITED_SMALL_TEXT_ALLOWLIST',files=files,
        hashes={n:sha256(ROOT/n)for n in files},total_bytes=sum((ROOT/n).stat().st_size for n in files)),indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],files=len(files),all47_and_MOT_exact=True,new_Test=False,allowlist=str(allowlist))),flush=True)


if __name__=='__main__':
    main()
