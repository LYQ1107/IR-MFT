#!/usr/bin/env python3
"""Publish all47 fields and exact official CSVs for fixed CPU public trackers."""
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_mft_public_trackers import context, read, sha, sources, write
from tools.export_perceptual_metrics47 import official_schema
from tools.fusion_memory_evaluation import official_csv47, validate_mot


def main():
    spec, plan, protocol, work, base = context()
    state, registration = read(base / 'state.json'), read(base / 'registered.json')
    receipt = read(base / 'controller_execution.json')
    if (state['status'] != 'COMPLETE_ALL8_DEV_AND_ALL8_TEST_CPU_PUBLIC_TRACKERS'
            or len(state['jobs']) != 16 or receipt['actual_controller_exit_code'] != 0
            or receipt['state_sha256'] != sha(base / 'state.json')
            or (Path('/proc') / str(state['controller']['pid'])).exists()
            or registration['source_hashes'] != sources(spec, work)):
        raise ValueError('Wait for the original 16 workers and actual controller completion')
    freeze = read(base / 'test_parameters_frozen.json')
    if (not freeze['all8_complete_Dev_before_Test']
            or freeze['registration_sha256'] != sha(base / 'registered.json')
            or freeze['identical_parameters'] != spec['trackers']):
        raise ValueError('Complete Dev and identical frozen Test parameters required')
    previous = Path(spec['infrastructure_retry']['previous_attempt_directory'])
    old_spec = read(previous / 'frozen_audit_mft_public_trackers_20261125.json')
    for key in ['reference_sources', 'streams', 'trackers', 'SORT_detection_threshold',
                'minimum_box_area', 'source_RGB_checkpoint_sha256', 'coordinate_policy']:
        if old_spec[key] != spec[key]:
            raise ValueError('An infrastructure retry changed scientific inputs/settings')
    retry_tracks_exact = {}
    for seq in protocol['dev']:
        suffix = Path('dev/sort__source_RGB_fresh/track_results') / (seq + '.txt')
        if sha(previous / suffix) != sha(base / suffix):
            raise ValueError('Rendering repair changed already completed source-RGB SORT tracks')
        retry_tracks_exact[seq] = sha(base / suffix)
    log_attempt = Path(spec['controller_log_retry']['previous_attempt_directory'])
    if sha(log_attempt / 'controller_execution_failure.json') != spec['controller_log_retry']['failure_receipt_sha256']:
        raise ValueError('Preserve the actual failed parent logging receipt')
    for key in ['reference_sources', 'streams', 'trackers', 'SORT_detection_threshold',
                'minimum_box_area', 'source_RGB_checkpoint_sha256', 'coordinate_policy']:
        if read(log_attempt / 'frozen_audit_mft_public_trackers_20261125.json')[key] != spec[key]:
            raise ValueError('A parent logging repair changed scientific inputs/settings')
    suffix = Path('dev/sort__source_RGB_fresh/trackeval_output/pedestrian_detailed.csv')
    if sha(log_attempt / suffix) != sha(base / suffix):
        raise ValueError('The parent logging repair changed the successful original full-Dev metrics')

    destination = ROOT / 'docs/fusion_research/perceptual_20261125_public_tracker_comparison'
    if destination.exists():
        raise ValueError('Preserve an earlier publication attempt')
    destination.mkdir()
    schema, counts = official_schema(plan['evaluation']['TrackEval_root'])
    catalog_path = ROOT / 'docs/fusion_research/perceptual_20261122_official_tables/index.json'
    catalog = {r['source']: r for r in read(catalog_path)['rows']}
    rows, raw_sources = [], []
    for split in ['dev', 'test']:
        for method in ['bytetrack', *spec['trackers']]:
            for stream, selection in spec['streams'].items():
                identifier = method + '__' + stream
                if method == 'bytetrack':
                    case = selection['Dev_case' if split == 'dev' else 'Test_case']
                    folder = work / ('structural_full_train_' + split) / case
                    original = read(folder / 'metrics.json')
                    source = folder / 'trackeval_output/pedestrian_detailed.csv'
                    published = catalog[str(source)]
                    if (sha(source) != published['sha256']
                            or sha(ROOT / published['file']) != published['sha256']):
                        raise ValueError('A previously published ByteTrack result changed')
                    detail = official_csv47(source, schema, split, original['trackeval'])
                    provenance = dict(existing_published_CSV=published['file'],
                        source_metrics_sha256=sha(folder / 'metrics.json'),
                        new_CPU_worker=False, original_case=case)
                    cost = None
                else:
                    key = split + '__' + identifier
                    job = state['jobs'][key]
                    folder = base / split / identifier
                    original = read(folder / 'metrics.json')
                    manifest = original['manifest']
                    if (job['status'] != 'COMPLETE' or job['exit_code'] != 0
                            or sha(folder / 'metrics.json') != job['metrics_sha256']
                            or sha(job['log']) != job['log_sha256']
                            or original['status'] != 'COMPLETE_CPU_PUBLIC_TRACKER_FULL_SPLIT'
                            or manifest['GT_inference_input'] or manifest['future_frame_input']
                            or manifest['registration_sha256'] != sha(base / 'registered.json')
                            or manifest['sources'] != registration['streams'][split][stream]
                            or manifest['source_versions'] != registration['source_hashes']):
                        raise ValueError('Actual worker, frozen source or complete metric lineage differs')
                    if split == 'dev' and freeze['complete_Dev_metrics'][key] != job['metrics_sha256']:
                        raise ValueError('The full Dev record changed after freezing Test')
                    for seq, coverage in manifest['coverage'].items():
                        track = folder / 'track_results' / (seq + '.txt')
                        if sha(track) != coverage['output_sha256']:
                            raise ValueError('Actual emitted MOT bytes changed')
                        validate_mot(track, coverage['frames'], coverage['output_rows'])
                        binding = manifest['sources'][seq]
                        if (sha(binding['path']) != binding['sha256']
                                or sha(binding['producer']) != binding['producer_sha256']):
                            raise ValueError('The source detector packets changed')
                    source = folder / 'trackeval_output/pedestrian_detailed.csv'
                    detail = official_csv47(source, schema, split, original['metrics39'])
                    if (detail['combined47'] != original['combined47']
                            or detail['per_sequence47'] != original['per_sequence47']):
                        raise ValueError('The original all47 result differs from the actual official CSV')
                    target = destination / (split + '__' + identifier + '.csv')
                    shutil.copyfile(source, target)
                    if sha(target) != sha(source):
                        raise ValueError('Published original CSV bytes differ')
                    log_target = destination / (split + '__' + identifier + '_worker_log.txt')
                    shutil.copyfile(job['log'], log_target)
                    if sha(log_target) != job['log_sha256']:
                        raise ValueError('Published actual worker log bytes differ')
                    provenance = dict(new_CPU_worker=True, source_metrics_sha256=job['metrics_sha256'],
                        actual_execution=job, official_CSV=str(target.relative_to(ROOT)),
                        actual_worker_log=str(log_target.relative_to(ROOT)))
                    cost = dict(tracker_update_seconds=manifest['tracker_update_seconds'],
                        tracker_update_FPS=manifest['frames'] / manifest['tracker_update_seconds'],
                        replay_seconds=manifest['replay_seconds'],
                        replay_FPS=manifest['frames'] / manifest['replay_seconds'],
                        replay_peak_RSS_GiB=manifest['peak_process_RSS_GiB'],
                        whole_worker_sampled_RSS_GiB=job['max_sampled_RSS_GiB'],
                        whole_worker_wall_seconds=job['wall_seconds'], scope=manifest['cost_scope'])
                    raw_sources.append(dict(split=split, method=method, stream=stream,
                        source=str(source), file=str(target.relative_to(ROOT)), sha256=sha(source),
                        original_columns=276))
                rows.append(dict(split=split, method=method, stream=stream,
                    combined47=detail['combined47'], per_sequence47=detail['per_sequence47'],
                    official_CSV_sha256=sha(source), provenance=provenance, CPU_cached_tracking_cost=cost))
    if len(rows) != 24 or len(raw_sources) != 16:
        raise ValueError('Require 8 existing ByteTrack rows and all16 new public tracker rows')
    empty_argv_snapshots = [k for k, value in state['jobs'].items()
        if value['process']['argv_sha256'] == 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855']
    proc_checks = [json.loads(line) for line in (base / 'independent_live_proc_checks.jsonl').read_text().splitlines()]
    for check in proc_checks:
        job = state['jobs'][check['job']]
        if (not check['matches_original_job'] or check['actual_argv'] != job['command']
                or check['pid'] != job['process']['pid']
                or check['start_ticks'] != job['process']['start_ticks']
                or check['actual_CUDA'] != ['CUDA_VISIBLE_DEVICES=']):
            raise ValueError('Independent live PID/argv/CUDA observation differs')
    lookup = {(r['split'], r['method'], r['stream']): r for r in rows}
    effects = []
    for split in ['dev', 'test']:
        for method in ['bytetrack', *spec['trackers']]:
            ir = lookup[split, method, 'primary_IR']['combined47']
            for control in ['source_RGB_fresh', 'matched_rgb', 'matched_gray']:
                other = lookup[split, method, control]['combined47']
                effects.append(dict(split=split, method=method, control=control,
                    delta={k: ir[k] - other[k] for k in ['HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW']}))
    compact_bindings = {split: {stream: {seq: {k: value[k] for k in [
        'path', 'sha256', 'frames', 'producer', 'producer_sha256', 'original_Dev_selected_case']}
        for seq, value in sequences.items()} for stream, sequences in streams.items()}
        for split, streams in registration['streams'].items()}
    historical = spec['historical_public_methods']
    encoded = subprocess.check_output(['git', 'show', historical['commit'] + ':' + historical['path']], cwd=ROOT)
    historical_target = destination / 'historical_public_methods_619cf27.md'
    with historical_target.open('xb') as handle:
        handle.write(encoded)
    failed_logs = {}
    for label, attempt in [('R1_graphics', previous), ('R2_parent_log', log_attempt)]:
        path = attempt / 'logs/dev__sort__source_RGB_fresh.txt'
        target = destination / ('original_' + label + '_worker_log.txt')
        shutil.copyfile(path, target)
        if sha(path) != sha(target):
            raise ValueError('Original infrastructure-attempt log changed')
        failed_logs[label] = dict(source=str(path), published=str(target.relative_to(ROOT)), sha256=sha(path))
    models = {}
    for stream, sequences in registration['streams']['dev'].items():
        model = next(iter(sequences.values()))['model']
        models[stream] = {k: v for k, v in model.items() if k in [
            'id', 'cue', 'components', 'seed', 'epoch', 'logical_Train_epoch', 'variant',
            'checkpoint', 'checkpoint_sha256', 'full_fit_proof_sha256']}
        models[stream]['shared_RGB_checkpoint_sha256'] = spec['source_RGB_checkpoint_sha256']
    report = dict(status='COMPLETE_FIXED_PUBLIC_TRACKER_FULL_DEV_TEST_NOT_GOAL_COMPLETION',
        specification=str(Path('configs/audit_mft_public_trackers_20261125.json')),
        specification_sha256=sha(ROOT / 'configs/audit_mft_public_trackers_20261125.json'),
        registration_sha256=sha(base / 'registered.json'),
        source_versions=registration['source_hashes'], source_detection_bindings=compact_bindings,
        source_models=models,
        actual_controller_receipt=receipt, state=state,
        independent_live_proc_checks=proc_checks,
        early_empty_proc_argv_snapshots=empty_argv_snapshots,
        engineering_checks=read(base / 'engineering.json')['checks'],
        first_cache_schema_engineering_failure=read(previous / 'engineering_attempt_r1_failure.json'),
        first_controller_graphics_failure=read(previous / 'controller_execution_failure.json'),
        second_controller_log_failure=read(log_attempt / 'controller_execution_failure.json'),
        original_infrastructure_attempt_logs=failed_logs,
        parent_log_retry_original_full_Dev_CSV_byte_exact=sha(log_attempt / suffix),
        graphics_retry_first_complete_Dev_tracks_byte_exact=retry_tracks_exact,
        Test_freeze=freeze, official_fields=schema, official_family_field_counts=counts,
        rows=rows, effects=effects, new_original_276_column_CSVs=raw_sources,
        historical_public_methods=dict(**historical, snapshot=str(historical_target.relative_to(ROOT)),
            snapshot_sha256=sha(historical_target)),
        prior_official_catalog_sha256=sha(catalog_path),
        scope='All8 public cases completed both complete4834frame Dev and15985frame seven-sequence Test. '
              'Four already published ByteTrack cases are references; no new GPU/detector/IR generation. '
              'One fixed default parameter set per public tracker, not a tuned best-of-method reproduction. '
              'RGB uses its original priority policy; IR/gray use their own original Dev-selected raw policies. '
              'SORT/OC-SORT API supplies no confidence, so emitted MOT confidence is1 and unused in evaluation. '
              'CPU tracker/replay costs exclude detector/fusion and cannot be called end-to-end FPS.',
        original_fusion_primary_unchanged=True, new_fusion_method_selected=False,
        fixed_seed=0, additional_seed_work=False, GPU_work=False,
        supported_innovations=0, goal_met=False,
        limitations=['Historical Test already exposed; no new blind-test claim.',
            'Source RGB pretraining saw the current Dev; pseudoIR derives from RGB.',
            'SORT max_age1 and OC-SORT max_age30 are their fixed defaults, not matched lifecycle tuning.',
            'These backend comparisons do not replace multimodal module removal/wrong/missing controls.',
            'Shared CPU host; measured cached tracker rates are not isolated or end-to-end rates.'])
    doc = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261125_PUBLIC_TRACKERS.json'
    write(doc, report)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=['split', 'method', 'stream', 'seq'] + [r['metric'] for r in schema], lineterminator='\n')
    writer.writeheader()
    for row in rows:
        for values in row['per_sequence47']:
            writer.writerow(dict(method=row['method'], stream=row['stream'], **values))
    csv_path = doc.with_suffix('.csv')
    with csv_path.open('x') as handle:
        handle.write(buffer.getvalue())
    lines = ['# 当前冻结检测输出上的公开跟踪方法对照', '',
        'SORT、OC-SORT 各四组输出，均先完成两段4834帧Dev，再冻结相同参数完成七段15985帧Test；'
        '16个新CPU worker均实际退出0。ByteTrack四组是先前已发布结果，未重新训练或替换。', '',
        '| 跟踪器 | 冻结检测输出 | Test HOTA | AssA | IDF1 | MOTA | IDSW |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        if row['split'] == 'test':
            m = row['combined47']
            lines.append(f"| {row['method']} | {row['stream']} | {m['HOTA']:.6f} | {m['AssA']:.6f} | {m['IDF1']:.6f} | {m['MOTA']:.6f} | {m['IDSW']} |")
    lines += ['', '每个跟踪器内，固定IR相对同预算RGB的HOTA差值：', '']
    for effect in effects:
        if effect['split'] == 'test' and effect['control'] == 'matched_rgb':
            lines.append(f"- {effect['method']}：{effect['delta']['HOTA']:+.6f}点。")
    primary = lookup['test', 'bytetrack', 'primary_IR']['combined47']['HOTA']
    lines += ['', f'原Dev选定主方法仍为ByteTrack＋IR，HOTA {primary:.6f}；'
        '本表不按Test切换跟踪器或主模型，目标34.529及2–3项独立贡献尚未达成。', '',
        '两种参考实现均从官方固定提交下载，保留原字节与许可通知，未复制进Git；'
        '[配置](../../configs/audit_mft_public_trackers_20261125.json)记录URL、SHA和全部参数。'
        'SORT使用max_age=1、OC-SORT使用max_age=30，均min_hits=3、IoU=0.3；'
        '输入置信阈值0.6，最小框面积100。没有外观ReID或测试集参数搜索。', '',
        '第一次工程检查过严，未接受原始缓存已有的anchor_indices/rgb_reference_scores；'
        '修正为与原重放器一致仅读取四项检测字段。第一次正式CPU调用生成两段完整轨迹后，'
        '因SORT选择TkAgg而在TrackEval绘图阶段退出1；保留全部失败资产，改用Agg后，'
        '这两段重跑轨迹与失败调用逐字节相同。第二次CPU父调度在首个worker成功退出0后，'
        '因日志字段重名退出1；修复日志后，该完整Dev官方CSV逐字节相同。'
        '两个父进程的实际退出与原始worker日志均保留。没有更改跟踪算法、检测输出或种子。', '',
        '原始276列CSV、逐序列及COMBINED的47字段、实际进程/退出码、来源SHA和CPU计时见'
        '[完整JSON](PERCEPTUAL_MFT_20261125_PUBLIC_TRACKERS.json)、'
        '[47字段CSV](PERCEPTUAL_MFT_20261125_PUBLIC_TRACKERS.csv)。'
        '新原始CSV位于[本次目录](perceptual_20261125_public_tracker_comparison/)。'
        'CPU计时只覆盖缓存跟踪；不包含检测器、融合网络或伪IR生成，不能作为端到端FPS。', '',
        '执行代码：[CPU完整Dev/Test驱动](../../tools/audit_mft_public_trackers.py)、'
        '[独立结果核对及导出](../../tools/audit_mft_public_tracker_results.py)。'
        'JSON保存每个实际worker的完整argv、PID、启动tick、退出码、日志SHA和结果SHA；'
        '调用固定CUDA_VISIBLE_DEVICES为空、OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=2、MKL_NUM_THREADS=2。'
        f'其中{len(empty_argv_snapshots)}份启动瞬间的proc argv为空，保留其真实空字节SHA，不把它称为期望argv核验通过。'
        '进程启动命令、实际worker成功退出及独立活跃进程抽查另行保留。', '',
        '旧分支已有BoT-SORT c0 HOTA30.310、HybridSORT c2 HOTA30.677，'
        '它们使用旧V3检测输出及鱼类RGB ReID，输入与本轮不同。'
        '[原文快照](perceptual_20261125_public_tracker_comparison/historical_public_methods_619cf27.md)'
        '固定于619cf27，不能当成本轮相同检测输入的对照。', '',
        '历史Test暴露、原RGB预训练见Dev、伪IR由RGB派生的限制保持。'
        '本次补齐公开方法比较，不增加已验证创新点计数。', '']
    with doc.with_suffix('.md').open('x') as handle:
        handle.write('\n'.join(lines))
    write(base / 'publication_verification.json', dict(status='COMPLETE_ACTUAL_FILE_AND_ALL47_AUDIT',
        report_sha256=sha(doc), official_new_CSVs=16, existing_ByteTrack_references=8,
        complete_new_Dev_cases=8, complete_new_Test_cases=8, actual_worker_exit0=16,
        all_complete_Dev_before_Test=True, scientific_parameters_unchanged_after_rendering_retry=True,
        retry_Dev_track_bytes_exact=True, source_code_sha256=sha(__file__),
        published_files={str(p.relative_to(ROOT)): sha(p) for p in sorted(destination.iterdir())},
        goal_met=False, supported_innovations=0))
    print(json.dumps(dict(status=report['status'], rows=len(rows), new_CSVs=len(raw_sources),
        report_sha256=sha(doc)), ensure_ascii=False))


if __name__ == '__main__':
    main()
