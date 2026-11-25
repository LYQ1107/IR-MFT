#!/usr/bin/env python3
"""Independently publish all47 fields from completed public comparisons."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_mft_additional_public_trackers import context, sources, SPEC
from tools.audit_mft_public_trackers import read, sha, write
from tools.export_perceptual_metrics47 import official_schema
from tools.fusion_memory_evaluation import official_csv47, validate_mot


def copy_exact(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError('Preserve an earlier published raw result: ' + str(target))
    else:
        target.write_bytes(data)
    if target.read_bytes() != data:
        raise ValueError('Raw publication is not byte exact')


def old_raw(row):
    provenance = row['provenance']
    relative = provenance.get('official_CSV') or provenance['existing_published_CSV']
    path = ROOT / relative
    if sha(path) != row['official_CSV_sha256']:
        raise ValueError('An earlier published ByteTrack/SORT/OC-SORT CSV changed')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['dev', 'complete'], required=True)
    args = parser.parse_args()
    spec, plan, work, base = context()
    state = read(base / 'state.json')
    registration = read(base / 'registered.json')
    freeze = read(base / 'test_parameters_frozen.json')
    if (registration['source_hashes'] != sources(spec) or
            not freeze['all8_complete_Dev_before_Test'] or
            freeze['registration_sha256'] != sha(base / 'registered.json') or
            freeze['identical_parameters'] != spec['trackers']):
        raise ValueError('Actual full Dev and its identical frozen Test parameters are required')
    if args.stage == 'complete':
        receipt = read(base / 'controller_execution.json')
        if (state['status'] != 'COMPLETE_ALL8_DEV_AND_ALL8_TEST_ADDITIONAL_CPU_PUBLIC_TRACKERS' or
                len(state['jobs']) != 16 or receipt['actual_controller_exit_code'] != 0 or
                receipt['state_sha256'] != sha(base / 'state.json')):
            raise ValueError('Wait for all16 actual workers and actual controller exit0')
        proc = Path('/proc') / str(state['controller']['pid'])
        if proc.exists():
            fields = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
            if int(fields[19]) == state['controller']['start_ticks'] and fields[0] != 'Z':
                raise ValueError('Original completed CPU controller is still running')
    else:
        receipt = None
    splits = ['dev', 'test'] if args.stage == 'complete' else ['dev']
    old_path = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261125_PUBLIC_TRACKERS.json'
    old_report = read(old_path)
    old_digest = sha(old_path)
    schema, family_counts = official_schema(plan['evaluation']['TrackEval_root'])
    rows = []
    for row in old_report['rows']:
        if row['split'] not in splits:
            continue
        detail = official_csv47(old_raw(row), schema, row['split'])
        if detail['combined47'] != row['combined47'] or detail['per_sequence47'] != row['per_sequence47']:
            raise ValueError('Previously published 47 fields changed')
        rows.append(row)
    destination = ROOT / 'docs/fusion_research/perceptual_20261126_additional_public_tracker_results'
    raw_files = []
    image_rollups = {}
    for split in splits:
        for method in spec['trackers']:
            for stream in spec['streams']:
                identifier = method + '__' + stream
                key = split + '__' + identifier
                job = state['jobs'][key]
                folder = base / split / identifier
                original = read(folder / 'metrics.json')
                manifest = original['manifest']
                if (job['status'] != 'COMPLETE' or job['exit_code'] != 0 or
                        sha(folder / 'metrics.json') != job['metrics_sha256'] or
                        sha(job['log']) != job['log_sha256'] or
                        not job['process']['expected_argv_matches'] or
                        original['status'] != 'COMPLETE_CPU_ADDITIONAL_PUBLIC_TRACKER_FULL_SPLIT' or
                        manifest['GT_inference_input'] or manifest['future_frame_input'] or
                        manifest['appearance_network_used'] or manifest['GPU_work'] or
                        manifest['registration_sha256'] != sha(base / 'registered.json') or
                        manifest['source_versions'] != registration['source_hashes'] or
                        manifest['sources'] != registration['streams'][split][stream]):
                    raise ValueError('Actual worker or complete inference/evaluation provenance differs')
                if split == 'dev' and freeze['complete_Dev_metrics'][key] != job['metrics_sha256']:
                    raise ValueError('Dev metrics changed after Test freezing')
                for sequence, coverage in manifest['coverage'].items():
                    tracks = folder / 'track_results' / (sequence + '.txt')
                    if sha(tracks) != coverage['output_sha256']:
                        raise ValueError('Actual emitted MOT bytes changed')
                    validate_mot(tracks, coverage['frames'], coverage['output_rows'])
                    if method == 'botsort':
                        if coverage['RGB_camera_compensation_frames'] != coverage['frames']:
                            raise ValueError('ORB camera compensation did not receive every actual RGB frame')
                        image_rollups.setdefault((split, sequence), set()).add(coverage['RGB_content_rollup_sha256'])
                    binding = manifest['sources'][sequence]
                    if sha(binding['path']) != binding['sha256'] or sha(binding['producer']) != binding['producer_sha256']:
                        raise ValueError('Frozen detector packets or original producer changed')
                source = folder / 'trackeval_output/pedestrian_detailed.csv'
                detail = official_csv47(source, schema, split, original['metrics39'])
                if detail['combined47'] != original['combined47'] or detail['per_sequence47'] != original['per_sequence47']:
                    raise ValueError('New original metrics do not match the actual 276-column CSV')
                target = destination / (key + '.csv')
                log_target = destination / (key + '_worker_log.txt')
                copy_exact(source, target)
                copy_exact(Path(job['log']), log_target)
                frames = manifest['frames']
                cost = dict(tracker_update_seconds=manifest['tracker_update_seconds'],
                    tracker_update_FPS=frames/manifest['tracker_update_seconds'],
                    replay_seconds=manifest['replay_seconds'], replay_FPS=frames/manifest['replay_seconds'],
                    replay_peak_RSS_GiB=manifest['peak_process_RSS_GiB'],
                    whole_worker_sampled_RSS_GiB=job['max_sampled_RSS_GiB'],
                    whole_worker_wall_seconds=job['wall_seconds'], scope=manifest['cost_scope'])
                rows.append(dict(split=split, method=method, stream=stream,
                    combined47=detail['combined47'], per_sequence47=detail['per_sequence47'],
                    official_CSV_sha256=sha(source), CPU_cached_tracking_cost=cost,
                    provenance=dict(new_CPU_worker=True, actual_execution=job,
                        source_metrics_sha256=job['metrics_sha256'],
                        official_CSV=str(target.relative_to(ROOT)), actual_worker_log=str(log_target.relative_to(ROOT))),
                    scalar_score_compatibility=manifest['scalar_score_compatibility']))
                raw_files.append(dict(split=split, method=method, stream=stream,
                    file=str(target.relative_to(ROOT)), sha256=sha(source), original_columns=276))
    if any(len(values) != 1 for values in image_rollups.values()):
        raise ValueError('BoT-SORT camera-motion RGB inputs differ across the four detector streams')
    if len(rows) != 20*len(splits) or len(raw_files) != 8*len(splits):
        raise ValueError('Require all five backends and four frozen streams in every requested split')
    lookup = {(row['split'],row['method'],row['stream']):row for row in rows}
    effects = []
    for split in splits:
        for method in ['bytetrack','sort','ocsort','botsort','hybridsort']:
            ir = lookup[split,method,'primary_IR']['combined47']
            for control in ['source_RGB_fresh','matched_rgb','matched_gray']:
                other = lookup[split,method,control]['combined47']
                effects.append(dict(split=split, method=method, control=control,
                    delta={name:ir[name]-other[name] for name in ['HOTA','DetA','AssA','IDF1','MOTA','IDSW']}))
    report = dict(status=('COMPLETE_ADDITIONAL_PUBLIC_TRACKER_DEV_ONLY_NOT_TEST_COMPLETION'
                          if args.stage=='dev' else 'COMPLETE_ADDITIONAL_PUBLIC_TRACKER_FULL_DEV_TEST'),
        stage=args.stage, specification=str(SPEC.relative_to(ROOT)), specification_sha256=sha(SPEC),
        registration_sha256=sha(base/'registered.json'), original_controller_receipt=receipt,
        earlier_public_report=str(old_path.relative_to(ROOT)), earlier_public_report_sha256=old_digest,
        source_versions=registration['source_hashes'], Test_freeze=freeze,
        official_fields=schema, official_family_field_counts=family_counts, rows=rows, effects=effects,
        new_original_276_column_CSVs=raw_files, all_BoT_RGB_camera_inputs_equal_across_streams=True,
        original_fusion_primary_unchanged=True, new_fusion_method_selected=False,
        source_publication_tool=str(Path(__file__).relative_to(ROOT)), source_publication_tool_sha256=sha(Path(__file__)),
        fixed_seed=0, additional_seed_work=False, GPU_work=False, supported_innovations=0, goal_met=False,
        scope='Fixed tracking-backend comparison on frozen complete-Dev-selected detector streams. '
              'BoT-SORT and HybridSORT use their disclosed no-ReID presets/compatibility adapters. '
              'Not native person-ReID replication or a new fusion contribution. '
              'Earlier V3/fishRGB-ReID results stay separate. No Test reselection. '
              'CPU tracking costs exclude neural detector/fusion and IR generation. '
              'Historical Test exposure, RGB pretraining seeing Dev and RGB-derived pseudoIR remain disclosed.')
    stem='PERCEPTUAL_MFT_20261126_ADDITIONAL_PUBLIC_TRACKERS_'+('DEV' if args.stage=='dev' else 'COMPLETE')
    docs=ROOT/'docs/fusion_research'
    write(docs/(stem+'.json'), report)
    flat=docs/(stem+'.csv')
    fields=[item['metric'] for item in schema]
    with flat.open('x', newline='') as handle:
        writer=csv.writer(handle, lineterminator='\n')
        writer.writerow(['split','method','stream',*fields])
        for row in rows:
            writer.writerow([row['split'],row['method'],row['stream'],*[row['combined47'][field] for field in fields]])
    lines=['# 五种公开跟踪后端：'+('完整Dev' if args.stage=='dev' else '完整Dev／Test')+'比较', '',
        '使用同一组冻结的源RGB／同预算RGB／灰度／IR检测包。原ByteTrack主方案保持；公开后端不计为原创贡献。', '']
    for split in splits:
        lines += [('完整Dev：2段4834帧。' if split=='dev' else '完整Test：7段15985帧。'), '',
            '| 后端 | 源RGB HOTA | 同预算RGB | 灰度 | IR | IR−RGB | IR−灰度 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for method in ['bytetrack','sort','ocsort','botsort','hybridsort']:
            values=[lookup[split,method,stream]['combined47']['HOTA'] for stream in spec['streams']]
            lines.append('| '+method+' | '+' | '.join(f'{value:.6f}' for value in values)
                         +f' | {values[3]-values[1]:+.6f} | {values[3]-values[2]:+.6f} |')
        lines += ['']
    lines += ['BoT-SORT保留ORB、无ReID；HybridSORT采用官方MOT17无ReID的HMIoU／TCM／低分关联预设。'
              'MFT统一输出面积>100，BoT不应用person宽高比过滤；其25fps参数及分数布局兼容均按登记保留。', '',
        '表中快照／priority或raw策略沿用原ByteTrack完整Dev选择，没有针对其他后端重选检测器或做参数搜索。'
        '官方47字段、逐序列及COMBINED均在JSON，新增原276列CSV及实际worker日志按字节保存。', '',
        'CPU update包含BoT的ORB，replay另含RGB读图／解码；均不含检测器、融合或IR生成，不能作为端到端FPS。'
        '较早的BoT／HybridSORT V3＋鱼类RGB ReID结果使用不同输入，继续单独保留。', '',
        ('本页仅完成Dev审计。全部8项固定Test继续运行，包含负Dev条件；原7段Test主方案仍29.834177，目标34.529未达。'
         if args.stage=='dev' else '全部16个新worker与原CPU控制器均实际退出0。原7段Test主方案仍29.834177，不按本次Test改选方法；目标与独立贡献要求仍待完成。'), '',
        '历史Test已暴露、RGB预训练见Dev及RGB派生伪IR限制保持。独立创新计数仍0，Goal保持active。', '',
        f'[完整JSON]({stem}.json) · [47字段CSV]({stem}.csv)']
    write_text=docs/(stem+'.md')
    with write_text.open('x') as handle:
        handle.write('\n'.join(lines)+'\n')
    if registration['source_hashes'] != sources(spec) or sha(old_path) != old_digest:
        raise ValueError('Sources or prior results changed during publication')
    print(json.dumps(dict(status=report['status'], rows=len(rows), new_raw_CSVs=len(raw_files),
                          report_sha256=sha(docs/(stem+'.json')))), flush=True)


if __name__=='__main__':
    main()
