#!/usr/bin/env python3
"""Read-only CPU delivery audit of the pre-frozen primary's complete7 Test.

Reuses the official47 CSV parser; does not recompute tracking metrics, infer,
fit/select models or modify the original run. Publication requires original
actual exit0 receipts, every video, every original condition and stable sources.
"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.fusion_memory_evaluation import official_csv47, validate_mot
from tools.fusion_memory_observation import LENGTHS
from tools.native_pyramid_experiment import context, read, registry
from tools.run_pseudo_ir_comparison import sha256

FREEZE_SHA = '65dad7840e5c8df49f15c68ce16e293581a4c7d432f395a5e9a3ccdaaa5cd8db'
SPEC_SHA = '25c180c1e4c6065bd5cefa6f59d0c1506a0f143114ee75684e49f46e7dbaec5b'
NORMAL_SHA = 'fd3b82e44e499f6e8d4488a5155a2ef55da75b4b4f15e6573307e1c2255857d9'
CONTROLLER_SHA = 'f5579fe7dca72d734d2d030a86216dd9584f672732252c77ec40ef2720fc91bb'
PHASE = 'test_original_baseline_priority_20261126'
PRIMARY, CANONICAL, FRESH = 'NP_temporal_ir_e06', 'NP_source_canonical_RGB', 'NP_source_fresh_RGB'
MAX_TEXT = 2 * 1024**2


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def text_bytes(path):
    data = Path(path).read_bytes()
    assert len(data) < MAX_TEXT and b'\0' not in data
    data.decode('utf-8')
    return data


def inputs():
    plan, study, protocol, calibration, work, checkpoint = context()
    registered = registry(work)
    assert sha256(ROOT / 'tools/run_mft_original_baseline_priority.py') == CONTROLLER_SHA
    assert sha256(work / 'native_pyramid_normal_selection_frozen.json') == NORMAL_SHA
    freeze_path = work / 'native_pyramid_Test_selection_frozen.json'
    assert sha256(freeze_path) == FREEZE_SHA
    published = ROOT / 'docs/fusion_research/perceptual_20261126_original_baseline_priority/frozen_Test_selection.json'
    assert sha256(published) == FREEZE_SHA
    freeze = read(freeze_path)
    assert freeze['status'] == 'FROZEN_FROM_COMPLETE_NATIVE_DEV_BEFORE_TEST'
    assert freeze['Test_read_for_selection'] is False and freeze['complete_Dev_before_Test'] is True
    assert freeze['roles'] == dict(primary_IR=PRIMARY, source_RGB_fresh=FRESH, source_RGB_canonical=CANONICAL)
    assert sha256(freeze['complete_Dev_file']) == freeze['complete_Dev_sha256']
    complete_Dev = read(freeze['complete_Dev_file'])
    assert complete_Dev['Test_read_for_selection'] is False
    assert complete_Dev['original_normal_selection_sha256'] == NORMAL_SHA
    assert sha256(complete_Dev['source_report']) == complete_Dev['source_report_sha256']
    spec_path = work / 'native_pyramid_specs' / (PHASE + '.json')
    assert sha256(spec_path) == freeze['specification_sha256'] == SPEC_SHA
    spec = read(spec_path)
    assert spec['split'] == 'test' and spec['models'] == freeze['models']
    assert {row['id'] for row in spec['models']} == {PRIMARY, FRESH}
    assert list(LENGTHS['test']) == protocol['historical_test'] and sum(LENGTHS['test'].values()) == 15985
    cases = {row['id']: row for row in freeze['cases']}
    assert set(cases) == {PRIMARY, CANONICAL, FRESH}
    for key, reference in cases.items():
        assert sha256(reference['config']) == reference['config_sha256']
        case = read(reference['config'])
        assert case['id'] == key and case['split'] == 'test' and case['seed'] == 0
        assert case['specification_sha256'] == SPEC_SHA and case['stream_directory'] == spec['stream_directory']
        if key == PRIMARY:
            chosen = read(work / 'native_pyramid_normal_selection_frozen.json')['selection']['primary']
            assert all(case[name] == chosen[name] for name in ['fit_id', 'epoch', 'mode', 'cue', 'checkpoint', 'checkpoint_sha256'])
            assert sha256(case['checkpoint']) == case['checkpoint_sha256'] and case['epoch'] == 6
    return plan, study, work, registered, freeze_path, freeze, spec_path, spec, cases


def readiness(work, cases):
    base = work / 'original_baseline_priority_20261126'
    state = read(base / 'state.json')
    execution = read(base / 'controller.execution.json')
    assert execution['source_sha256'] == CONTROLLER_SHA
    freeze = read(work / 'native_pyramid_Test_selection_frozen.json')
    assert datetime.datetime.fromisoformat(freeze['UTC']) < datetime.datetime.fromisoformat(execution['UTC_started'])
    if 'FAILED' in state['status'] or execution.get('actual_child_exit_code') not in (None, 0):
        raise RuntimeError('Preserve the actual original primary Test failure; never restart from this audit.')
    expected = {PHASE + '_NN': set(LENGTHS['test']), PHASE + '_CANONICAL': {'source'},
                PHASE + '_SOURCE_OFFICIAL': {FRESH, CANONICAL}, PHASE + '_OFFICIAL': {PRIMARY}}
    phases = state['phases']
    summary = {}
    complete = True
    for phase, ids in expected.items():
        entries = phases.get(phase, {})
        count = sum(job['status'] == 'COMPLETE' and type(job.get('exit_code')) is int and job['exit_code'] == 0 for job in entries.values())
        summary[phase] = dict(expected=len(ids), present=len(entries), complete=count)
        complete &= set(entries) == ids and count == len(ids)
    complete &= (type(execution.get('actual_child_exit_code')) is int and execution['actual_child_exit_code'] == 0 and
                 state['status'] == 'COMPLETE_PRIMARY_ORIGINAL_BASELINE_TEST_REQUIRES_ASSESSMENT' and
                 (base / 'full_Test_results.json').exists())
    current = None
    proc = Path('/proc') / str(execution['identity']['PID'])
    try:
        fields = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
        raw = (proc / 'cmdline').read_bytes()
        same = int(fields[19]) == execution['identity']['start_ticks']
        current = dict(PID=execution['identity']['PID'], same_start_ticks=same, kernel_state=fields[0],
                       raw_argv_sha256=hashlib.sha256(raw).hexdigest())
        if same and fields[0] != 'Z':
            assert current['raw_argv_sha256'] == execution['identity']['raw_argv_sha256']
            complete = False
    except (FileNotFoundError, ProcessLookupError):
        pass
    return dict(status='READY_ORIGINAL_PRIMARY_TEST_ACTUAL_EXIT_ZERO' if complete else 'WAITING_FOR_ORIGINAL_PRIMARY_FULL7_TEST_AND_ACTUAL_EXIT',
        UTC=utc(), original_phase=state['status'], original_controller=current,
        actual_controller_child_exit_code=execution.get('actual_child_exit_code'), phases=summary,
        publication_created=False, GPU_work=False, Test_accuracy_read=False), state, execution


def publish():
    from tools.export_perceptual_metrics47 import official_schema
    plan, study, work, registered, freeze_path, freeze, spec_path, spec, cases = inputs()
    ready, state, execution = readiness(work, cases)
    assert ready['status'] == 'READY_ORIGINAL_PRIMARY_TEST_ACTUAL_EXIT_ZERO'
    base = work / 'original_baseline_priority_20261126'
    original_report = read(base / 'full_Test_results.json')
    assert original_report['status'] == 'COMPLETE_PRIMARY_AND_ORIGINAL_BASELINE_FULL7_TEST'
    assert original_report['freeze_sha256'] == FREEZE_SHA and original_report['Test_used_for_selection'] is False
    lookup = {row['id']: row for row in original_report['rows']}
    assert set(lookup) == set(cases)
    schema, families = official_schema(plan['evaluation']['TrackEval_root'])
    assert schema == original_report['official_fields'] and len(schema) == 47
    docs = ROOT / 'docs/fusion_research'
    report_path = docs / 'PERCEPTUAL_MFT_20261126_NATIVE_PRIMARY_FULL_TEST.json'
    destination = docs / 'perceptual_20261126_native_primary_full_test'
    assert not destination.exists() and not any(report_path.with_suffix(ext).exists() for ext in ['.json', '.csv', '.md'])
    copies, rows = [], []
    def retain(source, name):
        source = Path(source)
        data = text_bytes(source)
        target = destination / name
        copies.append((target, data))
        return dict(file=str(target.relative_to(ROOT)), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    retain(base / 'full_Test_results.json', 'original_controller_full_Test_results.json')
    retain(base / 'controller.execution.json', 'original_controller_execution.json')
    retain(base / 'state.json', 'original_controller_final_state.json')
    retain(freeze_path, 'Test_selection_frozen_before_production.json')
    retain(spec_path, 'original_specification.json')
    producer_bindings = {}
    for sequence, frames in LENGTHS['test'].items():
        packet_path = Path(plan['frozen_detection_sources']['rgb_test']) / 'detections' / (sequence + '.npz')
        for canonical in [False, True]:
            path = Path(spec['stream_directory']) / 'sequence_manifests' / (sequence + ('.canonical.json' if canonical else '.json'))
            producer = read(path)
            assert producer['split'] == 'test' and producer['sequence'] == sequence and producer['frames'] == frames
            assert producer['GT_read'] is False and producer['future_frame_input'] is False and producer['previous_outputs_rewritten'] is False
            assert producer['specification_sha256'] == SPEC_SHA
            assert producer['canonical_detection_sha256'] == sha256(packet_path)
            assert set(producer['streams']) == ({CANONICAL} if canonical else {PRIMARY, FRESH})
            if not canonical:
                assert producer['source_RGB_sha256'] == study['source_RGB_sha256']
                assert producer['independent_input_batch_size'] == 1 and producer['physical_gpu'] in ['4', '5']
                assert len(producer['original_image_bytes_delivery_sha256']) == 64
            for key, stream in producer['streams'].items():
                assert stream['frames'] == frames and stream['sequence'] == sequence
                assert sha256(stream['track_path']) == stream['track_sha256']
                validate_mot(stream['track_path'], frames, stream['track_rows'])
            binding = retain(path, sequence + ('.canonical.json' if canonical else '.json'))
            producer_bindings[sequence + (':canonical' if canonical else ':current')] = binding
    for key in [CANONICAL, FRESH, PRIMARY]:
        reference = cases[key]
        output = work / 'native_pyramid_evaluation/test' / key
        metrics = read(output / 'metrics.json')
        manifest = metrics['manifest']
        assert metrics['status'] == 'COMPLETE' and manifest['config_sha256'] == reference['config_sha256']
        assert manifest['config'] == read(reference['config'])
        assert manifest['sequences'] == list(LENGTHS['test']) and manifest['frames'] == 15985
        assert manifest['GT_inference_input'] is False and manifest['future_frame_input'] is False
        assert manifest['previous_outputs_rewritten'] is False
        assert manifest['source_versions'] == registered['source_hashes']
        csv_path = output / 'trackeval_output/pedestrian_detailed.csv'
        parsed = official_csv47(csv_path, schema, 'test', metrics['trackeval'])
        assert parsed == {name: value for name, value in lookup[key].items() if name != 'id'}
        if key == CANONICAL:
            source = Path(plan['frozen_detection_sources']['rgb_test'])
            assert sha256(csv_path) == sha256(source / 'trackeval_output/pedestrian_detailed.csv')
            assert metrics['trackeval'] == read(source / 'metrics.json')['trackeval']
        row = dict(id=key, config=read(reference['config']), config_sha256=reference['config_sha256'],
            combined47=parsed['combined47'], per_sequence47=parsed['per_sequence47'],
            official_CSV=retain(csv_path, key + '.csv'), original_metrics=retain(output / 'metrics.json', key + '_metrics.json'))
        retain(Path(reference['config']), key + '_original_config.json')
        rows.append(row)
    for phase, jobs in state['phases'].items():
        for key, job in jobs.items():
            assert job['status'] == 'COMPLETE' and type(job['exit_code']) is int and job['exit_code'] == 0
            retain(Path(job['log']), phase + '__' + key + '_worker_log.txt')
    by_id = {row['id']: row for row in rows}
    baseline = by_id[CANONICAL]['combined47']['HOTA']
    primary = by_id[PRIMARY]['combined47']['HOTA']
    assert abs(baseline - 29.5289636096556) < 1e-9
    assert primary == original_report['primary_IR_HOTA']
    delta = primary - baseline
    assert delta == original_report['HOTA_gain_pp']
    all_fields = [item['metric'] for item in schema]
    comparisons = {}
    for key in [CANONICAL, FRESH]:
        control = by_id[key]
        comparisons[key] = dict(combined47_delta={field: by_id[PRIMARY]['combined47'][field] - control['combined47'][field] for field in all_fields},
            per_sequence47_delta=[dict(seq=a['seq'], **{field:a[field]-b[field] for field in all_fields})
                for a, b in zip(by_id[PRIMARY]['per_sequence47'], control['per_sequence47'])])
    report = dict(status='COMPLETE_INDEPENDENT_NATIVE_PRIMARY_FULL7_TEST_DELIVERY_AUDIT', UTC=utc(), split='test',
        videos=7, frames=15985, cases=3, rows=rows, official_fields=schema, official_family_field_counts=families,
        all_3_times8_times47_original_CSV_fields_checked=3*8*47,
        primary_IR=PRIMARY, original_RGB_baseline_HOTA=baseline, primary_IR_HOTA=primary, HOTA_gain_pp=delta,
        target_plus5_met=delta >= 5, target_plus10_met=delta >= 10,
        primary_vs_controls=comparisons, canonical_full_CSV_and39_exact_to_original_RGB=True,
        frozen_Test_sha256=FREEZE_SHA, original_normal_selection_sha256=NORMAL_SHA,
        frozen_Test_UTC=freeze['UTC'], original_controller_started_UTC=execution['UTC_started'],
        frozen_Test_written_before_original_production_controller_started=True,
        original_complete_worker_phases=state['phases'], original_completed_workers=sum(len(jobs) for jobs in state['phases'].values()),
        original_controller_actual_child_exit_code=0, original_controller_actual_parent_tool_exit_code_not_self_asserted=True,
        producer_bindings=producer_bindings, actual_MOT_file_hashes_and_rows_verified=True,
        source_hashes_verified_unchanged=registered['source_hashes'], source_audit_sha256=sha256(Path(__file__)),
        source_original_controller_sha256=CONTROLLER_SHA, fixed_seed=0, additional_seed_work=False,
        NN_or_GPU_work_by_audit=False, tracking_metrics_recomputed=False, Test_used_for_selection=False,
        independent_image_byte_replay=False, original_broader_study_or_removals_claimed_complete=False,
        supported_innovations=0, goal_met=False,
        limitations=['Historical Test was already exposed in this campaign.', 'RGB pretraining included present Dev videos.',
                    'PseudoIR is generated from RGB, not independent physical thermal measurement.',
                    'Producer delivery digests are checked as records, not independent replay of every image.',
                    'Shared bundle timing with gate observation is not per-method generation-inclusive FPS.'])
    # No publication file is created until every original condition validates.
    destination.mkdir()
    for path, data in copies:
        path.write_bytes(data)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    table = io.StringIO(newline='')
    writer = csv.DictWriter(table, fieldnames=['id', 'seq', 'split', *all_fields])
    writer.writeheader()
    for row in rows:
        for point in row['per_sequence47']:
            writer.writerow(dict(id=row['id'], **point))
    report_path.with_suffix('.csv').write_text(table.getvalue())
    md = '# 预先冻结伪红外主方案：完整七段Test\n\n'
    md += f'固定原始RGB baseline HOTA {baseline:.6f}；伪红外主方案 {primary:.6f}，绝对提升 **{delta:+.6f}点**。至少+5目标{ "达到" if delta >= 5 else "尚未达到" }，+10目标{ "达到" if delta >= 10 else "尚未达到" }。这是预先Dev冻结模型的正式结果，没有按Test改选。\n\n'
    fields = ['HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW']
    md += '| 条件 | ' + ' | '.join(fields) + ' |\n| --- | ' + ' | '.join(['---:']*len(fields)) + ' |\n'
    for row in rows:
        md += '| ' + row['id'] + ' | ' + ' | '.join(f"{row['combined47'][field]:.6f}" for field in fields) + ' |\n'
    md += '\n| 视频／官方汇总 | 原始RGB HOTA | 伪红外HOTA | 绝对点差 |\n| --- | ---: | ---: | ---: |\n'
    for a, b in zip(by_id[CANONICAL]['per_sequence47'], by_id[PRIMARY]['per_sequence47']):
        md += f"| {a['seq']} | {a['HOTA']:.6f} | {b['HOTA']:.6f} | {b['HOTA']-a['HOTA']:+.6f} |\n"
    md += '\n原始276列官方CSV、47字段、所有七视频与官方COMBINED、负值、原worker日志及来源原件均保留。汇总采用官方COMBINED，不按视频分数求均值。新鲜B1 RGB与历史规范B4 RGB均列出，固定主基线仍是后者。\n\n'
    md += '此为独立CPU交付审计，重新读取同一官方CSV并验证1128个原字段值、实际MOT文件与原始baseline CSV，不重新运行TrackEval或神经模型。原控制器和全部原worker实际退出0后才发布。输入字节摘要来自实际生产者记录；本审计不独立重放全部图像。共享两模型／门控观测时长不是单方法端到端FPS，固定离线生成成本不能被省略为已完成的全系统速度证明。\n\n'
    md += '后续原生IR记忆方案独立按Dev三个固定权重选择，不读取此Test分数改选择。扩展RGB结构对照和模块去除仍按用户最新范围暂缓，不把当前三条件报告称为旧广范围研究全部完成。独立创新计数仍0，研究Goal状态由完整证据另行审核。\n\n'
    md += '[完整JSON](PERCEPTUAL_MFT_20261126_NATIVE_PRIMARY_FULL_TEST.json) · [全部47字段CSV](PERCEPTUAL_MFT_20261126_NATIVE_PRIMARY_FULL_TEST.csv)。历史Test暴露、RGB预训练见Dev、伪IR源于RGB及单seed0限制保持。\n'
    report_path.with_suffix('.md').write_text(md)
    allow = [*destination.iterdir(), report_path, report_path.with_suffix('.csv'), report_path.with_suffix('.md')]
    for path in allow:
        text_bytes(path)
    frozen_json_path = base / 'full_Test_publication_allowlist.json'
    assert not frozen_json_path.exists()
    frozen_json_path.write_text(json.dumps([str(path.relative_to(ROOT)) for path in allow], indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], cases=3, videos=7, frames=15985, original_workers=report['original_completed_workers'],
        primary_IR_HOTA=primary, HOTA_gain_pp=delta, target_plus5_met=delta >= 5, report=str(report_path),
        publication_files=len(allow), GPU_work=False)), flush=True)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'publish', 'wait-publish'])
    args = parser.parse_args()
    _, _, work, _, _, _, _, _, cases = inputs()
    if args.mode == 'wait-publish':
        announced = False
        while True:
            ready, _, _ = readiness(work, cases)
            if ready['status'] == 'READY_ORIGINAL_PRIMARY_TEST_ACTUAL_EXIT_ZERO':
                break
            if not announced:
                print(json.dumps(ready), flush=True)
                announced = True
            time.sleep(15)
        publish()
    elif args.mode == 'check':
        print(json.dumps(readiness(work, cases)[0]), flush=True)
    else:
        publish()


if __name__ == '__main__':
    main()
