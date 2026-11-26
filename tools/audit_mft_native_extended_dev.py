#!/usr/bin/env python3
"""Audit original native stages9/12 without fitting, inference, or selection.

Every case, including negative arms, must have completed in the original queue.
Earlier raw proofs are referenced, not duplicated. Later complete Dev producers
must consume the same encoded RGB/original-IR byte stream as the reference
published before stage9. This checks recorded consumption digests, not an
independent replay or per-image enumeration. No Test results are read.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_mft_native_full_dev_stage import (  # noqa: E402
    CANONICAL, FRESH, LENGTHS, MAX_TEXT_BYTES, complete_jobs, read, require,
    sha, text_bytes,
)

DOCS = ROOT / 'docs/fusion_research'
PLAN = ROOT / 'configs/perceptual_mft_20261122.json'
STUDY = ROOT / 'configs/perceptual_native_pyramid_study_20261124.json'
BASE = DOCS / 'PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV.json'
BASE_SHA = '63426a8689f27ce662557b2df1d737db1d55d18af7e716bdd1374bdc290d688e'
HELPER_SHA = '971ff4788f011d7f4e70778986aa059ecfab7d1abd7d2b9485cf59df1a83f693'
REFERENCE = DOCS / 'perceptual_20261126_native_minimum6_full_dev/input_bytes_reference_before_stage9.json'
REFERENCE_SHA = '295025b0f2e08e76016e6061782e4fadd7c247136e0050565f6b2346a940d933'
CASE_DIR = ROOT / 'configs/perceptual_native_pyramid_dev_cases_20261124'


def case_ids(first, last):
    result = set()
    for mode in ('quality', 'temporal'):
        for cue in ('rgb', 'gray', 'ir'):
            for epoch in range(first, last + 1):
                key = f'NP_{mode}_{cue}_e{epoch:02d}'
                result.add(key)
                if cue == 'ir':
                    result.update((key + '_off', key + '_roll'))
    return result


def expected_jobs(stage):
    result = {}
    for end in range(6, stage + 1, 3):
        phase = f'dev_normal_{end:02d}'
        result[phase + '_NN'] = set(LENGTHS)
        result[phase + '_OFFICIAL'] = case_ids(1 if end == 6 else end - 2, end)
        if end == 6:
            result[phase + '_CANONICAL'] = {'source'}
            result[phase + '_SOURCE_OFFICIAL'] = {FRESH, CANONICAL}
        else:
            result[f'NORMAL_TRAIN_{end:02d}'] = {
                mode + '_' + cue for mode in ('quality', 'temporal')
                for cue in ('rgb', 'gray', 'ir')}
    return result


def paths(stage):
    if stage == 6:
        return BASE, DOCS / 'perceptual_20261126_native_minimum6_full_dev'
    return (DOCS / f'PERCEPTUAL_MFT_20261126_NATIVE_STAGE{stage}_FULL_DEV.json',
            DOCS / f'perceptual_20261126_native_stage{stage}_full_dev')


def fixed_inputs():
    require(sha(ROOT / 'tools/audit_mft_native_full_dev_stage.py') == HELPER_SHA,
            'Source-bound minimum6 audit helper changed')
    require(sha(BASE) == BASE_SHA and sha(REFERENCE) == REFERENCE_SHA,
            'Published minimum6/reference bytes changed')
    plan, study, reference = read(PLAN), read(STUDY), read(REFERENCE)
    require(sha(PLAN) == study['main_plan_sha256'] and
            sha(ROOT / plan['protocol']) == study['protocol_sha256'], 'Frozen protocol changed')
    require(read(ROOT / plan['protocol'])['dev'] == list(LENGTHS) and
            reference['future_Dev_stage_input_digest_must_match_per_sequence'] is True and
            set(reference['sequences']) == set(LENGTHS), 'Wrong fixed Dev input reference')
    return plan, study, Path(plan['work_root']), reference


def readiness(stage):
    _, _, work, _ = fixed_inputs()
    state = read(work / 'native_pyramid_study_state.json')
    require('FAIL' not in state['status'], 'Preserve original failure: ' + state['status'])
    expected = expected_jobs(stage)
    phases = {name: state.get('phases', {}).get(name, {}) for name in expected}
    decision = work / f'native_pyramid_Dev_stage_{stage:02d}.json'
    result = dict(status='WAITING_FOR_ALL_ORIGINAL_EXTENDED_DEV_AND_FROZEN_DECISION',
        stage=stage, controller_phase=state['status'], decision_exists=decision.exists(),
        phases={name: dict(expected=len(ids), present=len(phases[name]),
            complete=sum(j.get('status') == 'COMPLETE' and type(j.get('exit_code')) is int
                         and j['exit_code'] == 0 for j in phases[name].values()))
            for name, ids in expected.items()},
        GPU_work=False, Test_results_read=False, publication_files_created=False,
        new_accuracy_claimed=False, supported_innovations=0, goal_met=False)
    if decision.exists():
        complete_jobs(phases, expected)
        value = read(decision)
        require(value['status'] == 'COMPLETE_ALL_REGISTERED_DEV_AT_STAGE' and
                value['Test_read_for_selection'] is False and
                value['all_required_off_full_metrics_CSV_tracks_exact'] is True,
                'Original extended Dev decision unqualified')
        result['status'] = 'READY_ORIGINAL_EXTENDED_DEV_FOR_INDEPENDENT_AUDIT'
    return result, state, phases, decision


def same_image_bytes(sequence, producer, reference):
    frozen = reference['sequences'][sequence]
    require(producer['frames'] == frozen['frames'] == LENGTHS[sequence] and
            producer['original_image_bytes_delivery_sha256'] ==
            frozen['original_image_bytes_delivery_sha256'],
            'Encoded RGB/original-IR delivery digest differs: ' + sequence)


def publish(stage, engineering_only=False):
    from tools.export_perceptual_metrics47 import official_schema
    from tools.fusion_memory_evaluation import official_csv47, validate_mot
    from tools.native_pyramid_study_rules import best_dev, budget_decision

    require(stage in (9, 12) or (stage == 6 and engineering_only), 'Unregistered export stage')
    plan, study, work, reference = fixed_inputs()
    ready, state, phases, decision_path = readiness(stage)
    require(ready['status'].startswith('READY_'), 'Wait for the complete original extended Dev stage')
    registration = read(work / 'native_pyramid_pipeline_registered.json')
    require(state['source_hashes'] == registration['source_hashes'] and
            state['registry_sha256'] == sha(work / 'native_pyramid_pipeline_registered.json') and
            registration['study_sha256'] == sha(STUDY), 'Original registration changed')
    for name, digest in registration['source_hashes'].items():
        require(sha(ROOT / name) == digest, 'Frozen scientific source changed: ' + name)
    for name, digest in registration['artifact_hashes'].items():
        require(sha(name) == digest, 'Frozen artifact changed: ' + name)
    require(state['physical_gpus'] == ['4', '5'] and state['maximum_cards'] == 2 and
            state['fixed_seed'] == 0 and not state['additional_seed_work'], 'Resource/seed policy changed')
    prior_stage = 6 if engineering_only else stage - 3
    previous_path, _ = paths(prior_stage)
    previous = read(previous_path)
    previous_decision = work / f'native_pyramid_Dev_stage_{prior_stage:02d}.json'
    require(previous['stage'] == prior_stage and
            previous['original_stage_decision_sha256'] == sha(previous_decision), 'Prior audit/decision changed')
    frozen = read(decision_path)
    require(frozen['study_sha256'] == sha(STUDY) and
            frozen['rows'][:len(read(previous_decision)['rows'])] == read(previous_decision)['rows'],
            'Historical Dev rows were changed by the later stage')
    normal = [r for r in frozen['rows'] if not r.get('baseline') and r['intervention'] == 'normal']
    require(frozen['decision'] == budget_decision(normal, stage, intervention_coverage_complete=True),
            'Original common-budget decision differs')
    required_ids = case_ids(1, stage) | {FRESH, CANONICAL}
    require(len(frozen['rows']) == len(required_ids) == 10 * stage + 2 and
            {r['id'] for r in frozen['rows']} == required_ids, 'Missing/extra cumulative Dev condition')
    output, destination = paths(stage)
    if not engineering_only:
        require(not destination.exists() and not any(output.with_suffix(s).exists()
                for s in ('.json', '.csv', '.md')), 'Preserve earlier publication')
    copies, producer_bindings, cases = [], {}, {}

    def retain(source, name):
        source, target = Path(source), destination / name
        data = text_bytes(source)
        copies.append((source, target, data))
        return dict(file=str(target.relative_to(ROOT)), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))

    retain(decision_path, 'original_frozen_stage_decision.json')
    retain(REFERENCE, 'input_bytes_reference_frozen_before_stage9.json')
    for end in range(6, stage + 1, 3):
        spec_path = work / f'native_pyramid_specs/dev_normal_{end:02d}.json'
        spec = read(spec_path)
        interval_ids = case_ids(1 if end == 6 else end - 2, end)
        models = {row['id']: row for row in spec['models']}
        require(len(models) == len(spec['models']) and set(models) == interval_ids | {FRESH} and
                spec['status'] == 'FROZEN_BEFORE_NATIVE_OBSERVATION_PRODUCTION' and
                spec['split'] == 'dev' and spec['study_sha256'] == sha(STUDY), 'Wrong original stage specification')
        if end == stage:
            retain(spec_path, 'original_specification.json')
        for key in interval_ids | ({FRESH, CANONICAL} if end == 6 else set()):
            case_path = CASE_DIR / (key + '.json')
            case = read(case_path)
            require(case['id'] == key and case['split'] == 'dev' and case['seed'] == 0 and
                    case['study_sha256'] == sha(STUDY) and case['specification_sha256'] == sha(spec_path) and
                    Path(case['specification']) == spec_path and case['stream_directory'] == spec['stream_directory'],
                    'Original per-condition configuration differs')
            if key != CANONICAL:
                require(all(case[k] == v for k, v in models[key].items()), 'Case/model mismatch')
            if not case.get('baseline'):
                require(case['removal'] is None and not case['geometry_fixed'] and
                        sha(case['checkpoint']) == case['checkpoint_sha256'] and
                        sha(ROOT / 'configs/perceptual_native_pyramid_fit_cases_20261124' /
                            (case['fit_id'] + '.json')) == case['fit_config_sha256'], 'Changed snapshot/fit identity')
            cases[key] = (case_path, case)
            if end == stage:
                retain(case_path, key + '_original_config.json')
        for sequence, count in LENGTHS.items():
            for canonical in ([False, True] if end == 6 else [False]):
                p = Path(spec['stream_directory']) / 'sequence_manifests' / (sequence + ('.canonical.json' if canonical else '.json'))
                producer = read(p)
                require(producer['status'] == ('COMPLETE_NATIVE_CANONICAL_SOURCE_WITHOUT_GT' if canonical else
                        'COMPLETE_NATIVE_SEQUENTIAL_OBSERVATIONS_WITHOUT_GT') and
                        producer['sequence'] == sequence and producer['frames'] == count and producer['split'] == 'dev' and
                        producer['specification_sha256'] == sha(spec_path) and producer['study_sha256'] == sha(STUDY) and
                        producer['GT_read'] is False and producer['future_frame_input'] is False and
                        producer['previous_outputs_rewritten'] is False and
                        set(producer['streams']) == ({CANONICAL} if canonical else set(models)), 'Incomplete/noncausal producer')
                packet = Path(plan['frozen_detection_sources']['rgb_dev']) / 'detections' / (sequence + '.npz')
                require(producer['canonical_detection_sha256'] == sha(packet), 'Fixed RGB source changed')
                if not canonical:
                    same_image_bytes(sequence, producer, reference)
                    old = read(reference['sequences'][sequence]['producer'])
                    require(sha(reference['sequences'][sequence]['producer']) ==
                            reference['sequences'][sequence]['producer_sha256'], 'Original input-reference producer changed')
                    require(producer['physical_gpu'] in ['4', '5'] and producer['independent_input_batch_size'] == 1 and
                            producer['source_RGB_sha256'] == study['source_RGB_sha256'], 'Wrong GPU/batch/teacher')
                    for field in ('track_sha256', 'detection_packet_sha256'):
                        require(producer['streams'][FRESH][field] == old['streams'][FRESH][field], 'Fresh RGB anchor changed')
                    for key in interval_ids:
                        gate = producer['gate_statistics'][key]
                        require(gate['frames'] == count and len(gate['levels']) == 3 and
                                gate['available_frames'] + gate['missing_frames'] == count, 'Incomplete gate coverage')
                        if key.endswith('_off'):
                            require(gate['missing_frames'] == count, 'Missing condition not actually unavailable')
                            for field in ('track_sha256', 'detection_packet_sha256'):
                                require(producer['streams'][key][field] == producer['streams'][FRESH][field],
                                        'Missing condition differs from actual fresh RGB emission')
                producer_bindings[p] = (producer, sha(p))
                if end == stage:
                    retain(p, p.name)
    schema, families = official_schema(plan['evaluation']['TrackEval_root'])
    previous_rows = {row['id']: row for row in previous['rows']}
    rows = []
    for row in frozen['rows']:
        key = row['id']
        case_path, case = cases[key]
        metric_path = work / 'native_pyramid_evaluation/dev' / key / 'metrics.json'
        original = read(metric_path)
        manifest = original['manifest']
        require(original['status'] == 'COMPLETE' and row['source_metrics_sha256'] == sha(metric_path) and
                row['metrics'] == original['trackeval'] and Path(row['source_metrics']) == metric_path and
                all(row[k] == v for k, v in case.items()), 'Frozen metrics/config changed')
        require(manifest['status'] == 'INFERENCE_COMPLETE_BEFORE_GT_EVALUATION' and
                manifest['config'] == case and manifest['config_sha256'] == sha(case_path) and
                manifest['study_sha256'] == sha(STUDY) and manifest['plan_sha256'] == sha(PLAN) and
                manifest['source_versions'] == registration['source_hashes'] and
                manifest['split'] == 'dev' and manifest['frames'] == 4834 and manifest['sequences'] == list(LENGTHS) and
                manifest['GT_inference_input'] is False and manifest['future_frame_input'] is False and
                manifest['previous_outputs_rewritten'] is False, 'Inference provenance changed')
        for sequence, count in LENGTHS.items():
            binding, coverage = manifest['sources'][sequence], manifest['coverage'][sequence]
            producer, producer_sha = producer_bindings[Path(binding['manifest_path'])]
            stream = producer['streams'][key]
            require(binding['manifest_sha256'] == producer_sha and coverage['frames'] == stream['frames'] == count and
                    coverage['output_rows'] == stream['track_rows'] and coverage['output_sha256'] ==
                    stream['track_sha256'] == sha(stream['track_path']) and coverage['detection_packet_sha256'] ==
                    stream['detection_packet_sha256'], 'Full MOT/producer coverage mismatch')
            validate_mot(stream['track_path'], count, stream['track_rows'])
        csv_path = metric_path.parent / 'trackeval_output/pedestrian_detailed.csv'
        detail = official_csv47(csv_path, schema, 'dev', original['trackeval'])
        if key in previous_rows:
            prior = previous_rows[key]
            require(prior['combined47'] == detail['combined47'] and prior['per_sequence47'] == detail['per_sequence47'] and
                    prior['config_sha256'] == sha(case_path) and prior['official_CSV']['sha256'] ==
                    sha(ROOT / prior['official_CSV']['file']) == sha(csv_path) and prior['original_metrics']['sha256'] ==
                    sha(ROOT / prior['original_metrics']['file']) == sha(metric_path), 'Previously published evidence changed')
            rows.append(copy.deepcopy(prior))
        else:
            rows.append(dict(id=key, split='dev', config=case, config_sha256=sha(case_path),
                combined47=detail['combined47'], per_sequence47=detail['per_sequence47'],
                official_CSV=retain(csv_path, key + '.csv'), original_metrics=retain(metric_path, key + '_original_metrics.json'),
                original_official_metrics39=original['trackeval'], original_coverage=manifest['coverage'],
                original_source_manifest_references=manifest['sources']))
    lookup = {r['id']: r for r in rows}
    for row in rows:
        if row['config'].get('intervention') == 'unavailable':
            fresh = lookup[FRESH]
            require(all(row[k] == fresh[k] for k in ('combined47', 'per_sequence47', 'original_coverage')) and
                    row['official_CSV']['sha256'] == fresh['official_CSV']['sha256'], 'Unavailable all47/full CSV/MOT mismatch')
    for phase, jobs in phases.items():
        for key, job in jobs.items():
            end = int(phase.split('_')[2]) if phase.startswith('dev_normal_') else int(phase[-2:])
            spec_path = work / f'native_pyramid_specs/dev_normal_{end:02d}.json'
            if phase.startswith('NORMAL_TRAIN_'):
                tail = ['fit', '--fit-id', key, '--through-epoch', str(end)]
                script, slots = 'tools/train_native_pyramid.py', ['4', '5']
            else:
                tail = (['produce', '--split', 'dev', '--sequence', key, '--specification', str(spec_path)]
                    if phase.endswith('_NN') else ['canonical', '--split', 'dev', '--specification', str(spec_path)]
                    if phase.endswith('_CANONICAL') else ['evaluate', '--split', 'dev', '--config', str(cases[key][0])])
                script, slots = 'tools/evaluate_native_pyramid.py', ['4', '5'] if phase.endswith('_NN') else ['CPU0', 'CPU1']
            require(job['command'][1:] == ['-u', script, *tail] and job['slot'] in slots,
                    'Original command/slot mismatch')
            if end == stage:
                retain(job['log'], phase + '__' + key + '_worker_log.txt')
    own = {mode + '_' + cue: best_dev([r for r in normal if r['mode'] == mode and r['cue'] == cue])['id']
           for mode in ('quality', 'temporal') for cue in ('rgb', 'gray', 'ir')}
    report = dict(status='COMPLETE_INDEPENDENT_EXTENDED_FULL_DEV_AUDIT_NOT_FINAL_STUDY',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), stage=stage, split='dev', frames=4834,
        sequences=LENGTHS, cases=len(rows), all_registered_cases_included=True, rows=rows,
        official_fields=schema, official_family_field_counts=families, six_own_rounded_official_Dev_best=own,
        original_frozen_budget_decision=frozen['decision'], original_stage_decision_sha256=sha(decision_path),
        previous_audit=str(previous_path.relative_to(ROOT)), previous_audit_sha256=sha(previous_path),
        original_stage_worker_completion=phases, original_controller_phase_at_audit=state['status'],
        original_controller_exit_code=None, registered_sources_verified_unchanged=registration['source_hashes'],
        study_sha256=sha(STUDY), registration_sha256=sha(work / 'native_pyramid_pipeline_registered.json'),
        fixed_image_byte_reference=str(REFERENCE.relative_to(ROOT)), fixed_image_byte_reference_sha256=REFERENCE_SHA,
        every_stage_recorded_original_image_byte_digest_matches_reference=True,
        independent_image_byte_replay_or_per_image_list=False,
        unavailable_all47_per_sequence_full_CSV_full_MOT_exact_to_fresh_RGB=True,
        packet_hash_scope='Producer emission digests; discarded detection packets are not independently replayed.',
        fixed_seed=0, additional_seed_work=False, GPU_work=False, Test_results_read=False,
        final_model_selected_by_this_audit=False, new_Test_accuracy_claimed=False, supported_innovations=0, goal_met=False,
        limitations=study['limitations'], source_audit_tool=str(Path(__file__).relative_to(ROOT)),
        source_audit_tool_sha256=sha(__file__))
    final = read(work / 'native_pyramid_study_state.json')
    require({k: final['phases'][k] for k in phases} == phases and sha(decision_path) ==
            report['original_stage_decision_sha256'], 'Completed original stage changed during audit')
    for source, target, data in copies:
        require(source.read_bytes() == data, 'Original proof changed during audit: ' + str(source))
    encoded = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode()
    require(len(encoded) < MAX_TEXT_BYTES and len({t for _, t, _ in copies}) == len(copies), 'Oversized/duplicate output')
    if engineering_only:
        # Exercise the complete later-audit path on real completed stage6 data.
        # Never write over stage6 or reinterpret this as a stage9 completion.
        require(len(rows) == 62 and all(row['combined47'] == previous_rows[row['id']]['combined47'] and
                row['per_sequence47'] == previous_rows[row['id']]['per_sequence47'] for row in rows),
                'Engineering path differs from actual complete minimum6 results')
        return dict(actual_completed_stage=6, cases=len(rows), actual_original_jobs_checked=65,
            all47_and_per_sequence_exact_published_minimum6=True,
            source_manifest_MOT_and_checkpoint_hashes_verified=True,
            output_files_written=0, GPU_work=False, new_extended_accuracy_claimed=False)
    destination.mkdir()
    for _, target, data in copies:
        with target.open('xb') as handle:
            handle.write(data)
    with output.open('xb') as handle:
        handle.write(encoded)
    import csv
    with output.with_suffix('.csv').open('x', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        fields = [f['metric'] for f in schema]
        writer.writerow(['split', 'case', *fields])
        writer.writerows(['dev', row['id'], *[row['combined47'][f] for f in fields]] for row in rows)
    lines = [f'# 原生融合：完整{stage}轮 Dev 独立核对', '',
        f'两段完整Dev共4834帧，累计{len(rows)}项；包括所有正常、错位与缺失条件及两种RGB。', '',
        '| 同模式各自Dev最佳 | RGB HOTA | 灰度 HOTA | IR HOTA | IR−RGB | IR−灰度 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for mode in ('quality', 'temporal'):
        values = [lookup[own[mode + '_' + cue]]['combined47']['HOTA'] for cue in ('rgb', 'gray', 'ir')]
        lines.append('| ' + mode + ' | ' + ' | '.join(f'{v:.6f}' for v in values) +
                     f' | {values[2] - values[0]:+.6f} | {values[2] - values[1]:+.6f} |')
    ir = max(lookup[own[m + '_ir']]['combined47']['HOTA'] for m in ('quality', 'temporal'))
    rgb = max(lookup[own[m + '_rgb']]['combined47']['HOTA'] for m in ('quality', 'temporal'))
    lines += ['', f'跨两种结构的最强IR与最强RGB差为{ir - rgb:+.6f}点，必须同时保留。'
        '各自检查点排序沿用原官方39字段舍入规则；CSV未舍入值不用于重选。', '',
        '全部阶段记录的RGB／原始伪IR图像字节消费摘要均与九轮Dev前冻结的六轮参考一致。'
        '这是生产者消费摘要核对，不是独立图像重放或逐图清单。全部缺失条件的47字段、逐序列、完整CSV与MOT精确等于新鲜B1 RGB。', '',
        '原共同预算决定：' + json.dumps(frozen['decision'], ensure_ascii=False) + '。', '',
        '该文件仅为完整Dev阶段核对；没有运行新推理、训练、TrackEval或读取Test，不重新选择最终方案，'
        '不代表收敛、独立模块贡献或Test提升五点。原研究控制器仍须独立核对最终实际退出码。', '',
        f'[全部指标]({output.name}) · [47字段总表]({output.with_suffix(".csv").name})']
    with output.with_suffix('.md').open('x') as handle:
        handle.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=report['status'], stage=stage, cases=len(rows),
        raw_files=len(copies), report_bytes=len(encoded), original_budget_decision=frozen['decision'], goal_met=False)), flush=True)


def engineering(stage):
    _, _, _, reference = fixed_inputs()
    base = read(BASE)
    actual_path = publish(6, engineering_only=True)
    rejected = []
    for sequence in LENGTHS:
        value = read(reference['sequences'][sequence]['producer'])
        same_image_bytes(sequence, value, reference)
        corrupt = copy.deepcopy(value)
        corrupt['original_image_bytes_delivery_sha256'] = '0' * 64
        require(corrupt['streams'][FRESH] == value['streams'][FRESH], 'Engineering fresh stream unexpectedly changed')
        try:
            same_image_bytes(sequence, corrupt, reference)
        except ValueError:
            rejected.append(sequence + ':changed_image_bytes_despite_unchanged_RGB_output')
        else:
            raise ValueError('Changed image bytes with identical RGB outputs were accepted')
    ready = readiness(stage)[0]
    if ready['status'].startswith('WAITING_'):
        try:
            publish(stage)
        except ValueError as error:
            require(str(error) == 'Wait for the complete original extended Dev stage', 'Wrong live-stage failure')
        else:
            raise ValueError('Incomplete original extended stage was published')
        output, folder = paths(stage)
        require(not output.exists() and not folder.exists(), 'Incomplete stage emitted publication files')
    print(json.dumps(dict(status='COMPLETE_CPU_ENGINEERING_NOT_EXTENDED_ACCURACY',
        actual_minimum6_full_CSVs_checked=len(base['rows']), fields_each=47,
        actual_complete_audit_path=actual_path,
        changed_image_digest_cases_rejected=rejected, synthetic_changes_are_not_new_production_evidence=True,
        current_original_stage=ready, stage=stage, GPU_work=False, Test_results_read=False,
        new_extended_accuracy_claimed=False, supported_innovations=0, goal_met=False)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, choices=(9, 12), required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--readiness', action='store_true')
    action.add_argument('--publish', action='store_true')
    action.add_argument('--engineering', action='store_true')
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only auditor requires explicitly empty CUDA visibility')
    if args.readiness:
        print(json.dumps(readiness(args.stage)[0]), flush=True)
    elif args.engineering:
        engineering(args.stage)
    else:
        publish(args.stage)


if __name__ == '__main__':
    main()
