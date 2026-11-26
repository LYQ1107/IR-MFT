#!/usr/bin/env python3
"""Publish minimum-six-epoch native Dev evidence after its original jobs finish.

CPU only. Does not evaluate tracks, fit/select models, launch jobs, read Test
results, or alter the registered experiment. Incomplete original stages fail
before any publication directory is created. The original rounded-39-field
budget decision remains authoritative; full-precision CSV values are descriptive.
"""
import argparse
import copy
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN = ROOT / 'configs/perceptual_mft_20261122.json'
STUDY = ROOT / 'configs/perceptual_native_pyramid_study_20261124.json'
CONFIG_PUBLICATION = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261126_NATIVE_FULL_DEV_CONFIGS_REGISTERED.json'
FRESH = 'NP_source_fresh_RGB'
CANONICAL = 'NP_source_canonical_RGB'
LENGTHS = {'BT-005': 1834, 'SN-013': 3000}
PHASE = 'dev_normal_06'
MAX_TEXT_BYTES = 2 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def complete_jobs(phases, expected_ids):
    """Require original subprocess-poll completion, including negative arms."""
    for phase, ids in expected_ids.items():
        entries = phases.get(phase, {})
        require(set(entries) == set(ids), 'Missing/unexpected original jobs: ' + phase)
        for key, job in entries.items():
            require(job['id'] == key and job['status'] == 'COMPLETE' and
                    type(job.get('exit_code')) is int and job['exit_code'] == 0 and
                    type(job.get('pid')) is int and job['pid'] > 0,
                    'Original worker not actually complete with numeric exit0: ' + key)


def inputs():
    plan, study, published = read(PLAN), read(STUDY), read(CONFIG_PUBLICATION)
    work = Path(plan['work_root'])
    require(sha(PLAN) == study['main_plan_sha256'] and sha(STUDY) == published['study_sha256'],
            'Frozen study/main plan changed')
    protocol = read(ROOT / plan['protocol'])
    require(sha(ROOT / plan['protocol']) == study['protocol_sha256'] and
            protocol['dev'] == list(LENGTHS), 'Frozen full Dev split changed')
    require(published['counts'] == dict(baseline=2, normal=36,
            IR_image_spatial_roll=12, unavailable=12) and len(published['cases']) == 62,
            'Require exactly all62 prepublished original Dev cases')
    spec_path = work / 'native_pyramid_specs/dev_normal_06.json'
    spec = read(spec_path)
    require(sha(spec_path) == published['specification_sha256'] and spec['split'] == 'dev' and
            spec['study_sha256'] == sha(STUDY), 'Frozen specification changed')
    models = {row['id']: row for row in spec['models']}
    cases = {}
    for reference in published['cases']:
        path = ROOT / reference['file']
        value = read(path)
        require(sha(path) == reference['sha256'] and value['id'] == reference['id'] and
                value['seed'] == 0 and value['split'] == 'dev' and
                value['specification_sha256'] == sha(spec_path) and
                value['study_sha256'] == sha(STUDY) and
                value['stream_directory'] == spec['stream_directory'] and
                Path(value['specification']) == spec_path, 'A prepublished case changed')
        if value['id'] in models:
            require(all(value[k] == v for k, v in models[value['id']].items()),
                    'Case differs from actual inference specification')
        else:
            require(value['id'] == CANONICAL and value.get('canonical') is True and
                    value.get('baseline') == 'historical_canonical_RGB_B4',
                    'Unexpected non-produced case')
        require(value['id'] not in cases, 'Duplicate published case')
        cases[value['id']] = (path, value)
    require(set(cases) == set(models) | {CANONICAL} and len(models) == 61,
            'Incomplete or extra native output streams')
    return plan, study, work, published, spec_path, spec, cases


def expected_jobs(cases):
    return {PHASE + '_NN': set(LENGTHS), PHASE + '_CANONICAL': {'source'},
            PHASE + '_SOURCE_OFFICIAL': {FRESH, CANONICAL},
            PHASE + '_OFFICIAL': set(cases) - {FRESH, CANONICAL}}


def readiness(work, cases):
    path = work / 'native_pyramid_study_state.json'
    state = read(path)
    expected = expected_jobs(cases)
    phases = {key: state.get('phases', {}).get(key, {}) for key in expected}
    stage_path = work / 'native_pyramid_Dev_stage_06.json'
    result = dict(status='WAITING_FOR_ORIGINAL_COMPLETE_FULL_STAGE6_DEV',
        stage=6, original_controller_phase=state['status'],
        stage_decision_exists=stage_path.exists(), new_accuracy_claimed=False,
        Test_results_read=False, model_selection_performed=False,
        publication_files_created=False, goal_met=False,
        phases={key: dict(expected=len(ids), present=len(phases[key]),
            complete=sum(v.get('status') == 'COMPLETE' and type(v.get('exit_code')) is int and
                         v['exit_code'] == 0 for v in phases[key].values()))
            for key, ids in expected.items()})
    if 'FAIL' in state['status']:
        raise ValueError('Preserve original failed native controller: ' + state['status'])
    if stage_path.exists():
        complete_jobs(phases, expected)
        stage = read(stage_path)
        require(stage['status'] == 'COMPLETE_ALL_REGISTERED_DEV_AT_STAGE' and
                stage['Test_read_for_selection'] is False and
                stage['all_required_off_full_metrics_CSV_tracks_exact'] is True and
                stage['study_sha256'] == sha(STUDY), 'Unqualified original full Dev decision')
        result['status'] = 'READY_ORIGINAL_FULL_STAGE6_DEV_REQUIRES_INDEPENDENT_AUDIT'
    return result, state, phases, stage_path


def text_bytes(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Regular text proof required')
    data = path.read_bytes()
    data.decode('utf-8')
    require(b'\x00' not in data and len(data) < MAX_TEXT_BYTES, 'Binary or oversized publication rejected')
    return data


def publish():
    from tools.export_perceptual_metrics47 import official_schema
    from tools.fusion_memory_evaluation import official_csv47, validate_mot
    from tools.native_pyramid_study_rules import budget_decision, best_dev

    plan, study, work, published, spec_path, spec, cases = inputs()
    ready, state, phases, stage_path = readiness(work, cases)
    require(ready['status'].startswith('READY_'), 'Wait for every original full Dev worker and frozen decision')
    stage, registration = read(stage_path), read(work / 'native_pyramid_pipeline_registered.json')
    require(registration['study_sha256'] == sha(STUDY) and
            state['source_hashes'] == registration['source_hashes'] and
            state['registry_sha256'] == sha(work / 'native_pyramid_pipeline_registered.json'),
            'Original registration changed')
    for name, digest in registration['source_hashes'].items():
        require(sha(ROOT / name) == digest, 'Registered source changed: ' + name)
    for name, digest in registration['artifact_hashes'].items():
        require(sha(name) == digest, 'Registered artifact changed: ' + name)
    require(state['physical_gpus'] == ['4', '5'] and state['maximum_cards'] == 2 and
            state['fixed_seed'] == 0 and not state['additional_seed_work'], 'Resource/seed policy changed')
    frozen_rows = {row['id']: row for row in stage['rows']}
    require(len(stage['rows']) == len(frozen_rows) == 62 and set(frozen_rows) == set(cases),
            'Original frozen decision does not include every case')
    normal = [row for row in stage['rows'] if not row.get('baseline') and row['intervention'] == 'normal']
    require(stage['decision'] == budget_decision(normal, 6, intervention_coverage_complete=True),
            'Original rounded-official Dev budget decision differs')
    schema, family_counts = official_schema(plan['evaluation']['TrackEval_root'])
    folder = Path(spec['stream_directory'])
    stem = 'PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV'
    docs = ROOT / 'docs/fusion_research'
    destination = docs / 'perceptual_20261126_native_minimum6_full_dev'
    require(not destination.exists() and not any((docs / (stem + suffix)).exists()
            for suffix in ['.json', '.csv', '.md']), 'Preserve earlier publication')
    copies, rows, manifests = [], [], {}

    def retain(source, name):
        source, target = Path(source), destination / name
        data = text_bytes(source)
        copies.append((source, target, data))
        return dict(file=str(target.relative_to(ROOT)), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))

    retain(spec_path, 'original_specification.json')
    retain(stage_path, 'original_frozen_stage6_decision.json')
    for sequence, count in LENGTHS.items():
        for canonical in [False, True]:
            source = folder / 'sequence_manifests' / (sequence + ('.canonical.json' if canonical else '.json'))
            value = read(source)
            require(value['status'] == ('COMPLETE_NATIVE_CANONICAL_SOURCE_WITHOUT_GT' if canonical else
                    'COMPLETE_NATIVE_SEQUENTIAL_OBSERVATIONS_WITHOUT_GT') and value['split'] == 'dev' and
                    value['sequence'] == sequence and value['frames'] == count and
                    value['GT_read'] is False and value['future_frame_input'] is False and
                    value['previous_outputs_rewritten'] is False and
                    value['study_sha256'] == sha(STUDY) and
                    value['specification_sha256'] == sha(spec_path), 'Incomplete/noncausal original producer')
            require(set(value['streams']) == ({CANONICAL} if canonical else {r['id'] for r in spec['models']}),
                    'Unexpected/missing producer stream')
            packet_path = Path(plan['frozen_detection_sources']['rgb_dev']) / 'detections' / (sequence + '.npz')
            require(value['canonical_detection_sha256'] == sha(packet_path), 'Fixed RGB detection source changed')
            if not canonical:
                require(value['physical_gpu'] in ['4', '5'] and value['independent_input_batch_size'] == 1,
                        'Wrong actual GPU/batch size')
                require(value['source_RGB_sha256'] == study['source_RGB_sha256'], 'RGB teacher identity changed')
                for model in spec['models']:
                    if model.get('baseline'):
                        continue
                    gate = value['gate_statistics'][model['id']]
                    require(gate['frames'] == count and len(gate['levels']) == 3 and
                            gate['available_frames'] + gate['missing_frames'] == count,
                            'Gate statistics do not cover every actual frame')
                    if model['intervention'] == 'unavailable':
                        require(gate['missing_frames'] == count, 'Unavailable intervention not actually missing')
                        for key in ['track_sha256', 'detection_packet_sha256']:
                            require(value['streams'][model['id']][key] == value['streams'][FRESH][key],
                                    'Unavailable modality differs from fresh RGB at emission')
            manifests[sequence, canonical] = (value, retain(source, source.name))
    for phase, jobs in phases.items():
        for key, job in jobs.items():
            command = [str(v) for v in job['command']]
            expected_tail = (['produce', '--split', 'dev', '--sequence', key, '--specification', str(spec_path)]
                if phase.endswith('_NN') else ['canonical', '--split', 'dev', '--specification', str(spec_path)]
                if phase.endswith('_CANONICAL') else ['evaluate', '--split', 'dev', '--config', str(cases[key][0])])
            require(len(command) == 3 + len(expected_tail) and command[1:3] == ['-u', 'tools/evaluate_native_pyramid.py'] and
                    command[3:] == expected_tail and job['slot'] in (['4', '5'] if phase.endswith('_NN') else ['CPU0', 'CPU1']),
                    'Actual original worker command/slot differs')
            retain(job['log'], phase + '__' + key + '_worker_log.txt')
    for key, (case_path, case) in cases.items():
        metric_path = work / 'native_pyramid_evaluation/dev' / key / 'metrics.json'
        original = read(metric_path)
        manifest = original['manifest']
        frozen = frozen_rows[key]
        require(original['status'] == 'COMPLETE' and frozen['source_metrics_sha256'] == sha(metric_path) and
                frozen['metrics'] == original['trackeval'] and Path(frozen['source_metrics']) == metric_path and
                all(frozen[k] == value for k, value in case.items()), 'Original frozen metrics/config changed')
        require(manifest['status'] == 'INFERENCE_COMPLETE_BEFORE_GT_EVALUATION' and
                manifest['split'] == 'dev' and manifest['config'] == case and manifest['config_sha256'] == sha(case_path) and
                manifest['study_sha256'] == sha(STUDY) and manifest['plan_sha256'] == sha(PLAN) and
                manifest['sequences'] == list(LENGTHS) and manifest['frames'] == 4834 and
                manifest['source_versions'] == registration['source_hashes'] and
                manifest['GT_inference_input'] is False and manifest['future_frame_input'] is False and
                manifest['previous_outputs_rewritten'] is False, 'Official inference provenance changed')
        for sequence, count in LENGTHS.items():
            producer, reference = manifests[sequence, bool(case.get('canonical'))]
            binding, coverage = manifest['sources'][sequence], manifest['coverage'][sequence]
            stream = producer['streams'][key]
            require(binding['manifest_sha256'] == sha(binding['manifest_path']) == reference['sha256'] and
                    coverage['frames'] == stream['frames'] == count and
                    coverage['output_rows'] == stream['track_rows'] and
                    coverage['output_sha256'] == stream['track_sha256'] == sha(stream['track_path']) and
                    coverage['detection_packet_sha256'] == stream['detection_packet_sha256'],
                    'Full causal MOT/producer coverage changed')
            validate_mot(stream['track_path'], count, stream['track_rows'])
        csv_path = metric_path.parent / 'trackeval_output/pedestrian_detailed.csv'
        detail = official_csv47(csv_path, schema, 'dev', original['trackeval'])
        rows.append(dict(id=key, split='dev', config=case, config_sha256=sha(case_path),
            combined47=detail['combined47'], per_sequence47=detail['per_sequence47'],
            official_CSV=retain(csv_path, key + '.csv'),
            original_metrics=retain(metric_path, key + '_original_metrics.json'),
            original_official_metrics39=original['trackeval'], original_coverage=manifest['coverage'],
            original_source_manifest_references=manifest['sources']))
    lookup = {row['id']: row for row in rows}
    fresh = lookup[FRESH]
    canonical_source = Path(plan['frozen_detection_sources']['rgb_dev'])
    require(lookup[CANONICAL]['original_official_metrics39'] == read(canonical_source / 'metrics.json')['trackeval'] and
            lookup[CANONICAL]['official_CSV']['sha256'] ==
            sha(canonical_source / 'trackeval_output/pedestrian_detailed.csv'),
            'Canonical source baseline is not exactly reproduced')
    for row in rows:
        if row['config'].get('intervention') == 'unavailable':
            require(row['combined47'] == fresh['combined47'] and row['per_sequence47'] == fresh['per_sequence47'] and
                    row['official_CSV']['sha256'] == fresh['official_CSV']['sha256'] and
                    row['original_coverage'] == fresh['original_coverage'],
                    'Missing-modality all47/full-CSV/full-MOT identity failed')
    effects = []
    for mode in ['quality', 'temporal']:
        for epoch in range(1, 7):
            ir = lookup[f'NP_{mode}_ir_e{epoch:02d}']
            for control in [CANONICAL, FRESH, f'NP_{mode}_rgb_e{epoch:02d}',
                            f'NP_{mode}_gray_e{epoch:02d}', ir['id'] + '_roll', ir['id'] + '_off']:
                other = lookup[control]
                effects.append(dict(IR=ir['id'], control=control,
                    delta={field: ir['combined47'][field] - other['combined47'][field]
                           for field in ['HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW']}))
    own_best = {mode + '_' + cue: best_dev([r for r in normal if r['mode'] == mode and r['cue'] == cue])['id']
                for mode in ['quality', 'temporal'] for cue in ['rgb', 'gray', 'ir']}
    report = dict(status='COMPLETE_INDEPENDENT_MINIMUM6_FULL_DEV_AUDIT_NOT_FINAL_STUDY',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), stage=6, split='dev',
        sequences=LENGTHS, frames=4834, all62_cases_included=True, official_fields=schema,
        official_family_field_counts=family_counts, rows=rows, paired_snapshot_effects=effects,
        six_own_rounded_official_Dev_best=own_best,
        original_frozen_budget_decision=stage['decision'],
        original_stage_decision_sha256=sha(stage_path),
        prepublished_config_manifest_sha256=sha(CONFIG_PUBLICATION),
        study_sha256=sha(STUDY), specification_sha256=sha(spec_path),
        registration_sha256=sha(work / 'native_pyramid_pipeline_registered.json'),
        registered_sources_verified_unchanged=registration['source_hashes'],
        original_stage_worker_completion=phases, original_controller_phase_at_audit=state['status'],
        original_controller_exit_code=None,
        original_controller_completion_required_for_final_study=True,
        unavailable_all47_per_sequence_full_CSV_full_MOT_exact_to_fresh_RGB=True,
        packet_hash_scope='Original producer checks every unavailable frame against fresh RGB and records their '
            'identical rolling packet digests. Packets are not persisted, so this CPU audit does not independently '
            'reconstruct those discarded detections.',
        fixed_seed=0, additional_seed_work=False, GPU_work=False, Test_results_read=False,
        Test_based_selection=False, final_model_selected_by_this_audit=False,
        new_Test_accuracy_claimed=False, supported_innovations=0, goal_met=False,
        scientific_scope='Only minimum-six-epoch full Dev evidence. Follow the original 6/9/12 common-budget '
            'decision; all six normal conditions require full Test including negative arms. No module is counted '
            'as an independent innovation from Dev differences. This is not convergence, complete Test, '
            'individual runtime, final selection, or attainment of the five-point goal.',
        historical_Test_previously_exposed=True, RGB_pretraining_saw_Dev=True,
        pseudoIR_derived_from_RGB=True, limitations=study['limitations'],
        source_audit_tool=str(Path(__file__).relative_to(ROOT)), source_audit_tool_sha256=sha(Path(__file__)))
    # Finish every source check before emitting any publication file.
    final = read(work / 'native_pyramid_study_state.json')
    require({key: final['phases'][key] for key in phases} == phases and
            sha(stage_path) == report['original_stage_decision_sha256'], 'Completed original stage changed during audit')
    for source, target, data in copies:
        require(source.read_bytes() == data, 'Original raw proof changed during audit: ' + str(source))
    encoded = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode()
    require(len(encoded) < MAX_TEXT_BYTES and len({target for _, target, _ in copies}) == len(copies),
            'Oversized report or duplicate publication destination')
    destination.mkdir()
    for source, target, data in copies:
        with target.open('xb') as handle:
            handle.write(data)
    with (docs / (stem + '.json')).open('xb') as handle:
        handle.write(encoded)
    fields = [item['metric'] for item in schema]
    with (docs / (stem + '.csv')).open('x', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['split', 'case', *fields])
        for row in rows:
            writer.writerow(['dev', row['id'], *[row['combined47'][field] for field in fields]])
    lines = ['# 原生辅助金字塔：最低六轮完整 Dev', '',
        '两段完整Dev共4834帧，保留36个正常检查点、12个实际IR错位、12个缺失模态与两种RGB基线。'
        '原控制器的全部65个生产/规范源/官方评估worker均已实际退出0；本审计不要求整个研究控制器已退出。', '',
        '| 同模式各自Dev最佳 | RGB HOTA | 灰度 HOTA | IR HOTA | IR−RGB | IR−灰度 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for mode in ['quality', 'temporal']:
        values = [lookup[own_best[mode + '_' + cue]]['combined47']['HOTA'] for cue in ['rgb', 'gray', 'ir']]
        lines.append('| ' + mode + ' | ' + ' | '.join(f'{v:.6f}' for v in values) +
                     f' | {values[2]-values[0]:+.6f} | {values[2]-values[1]:+.6f} |')
    lines += ['', '最佳检查点沿用原官方39字段的舍入值与既定排序；上表显示对应CSV的未舍入值。'
        '原统一训练预算决定：' + json.dumps(stage['decision'], ensure_ascii=False) + '。', '',
        'JSON保留全部47字段、逐序列及COMBINED、同轮对照和IR错位/缺失差值；'
        '原276列CSV、原metrics、完整生产清单、冻结Dev决定及worker日志按字节保存。'
        '未发布图像、权重、检测包或轨迹。缺失模态全部47字段、逐序列、完整CSV及MOT均精确等于新B1 RGB。', '',
        '这是完整Dev阶段证据，尚不代表完整Test、收敛、独立创新或+5目标完成。'
        '后续严格沿原6/9/12规则推进，负结果同样进入必要Test。历史Test暴露、'
        'RGB预训练见Dev及伪IR由RGB生成等限制保持。', '',
        f'[全部62路指标]({stem}.json) · [47字段总表]({stem}.csv)']
    with (docs / (stem + '.md')).open('x') as handle:
        handle.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=report['status'], cases=len(rows), raw_files=len(copies),
        report_bytes=len(encoded), original_budget_decision=stage['decision'], goal_met=False)), flush=True)


def engineering():
    """Real previously published full Dev CSV and adversarial queue evidence."""
    import tempfile
    from tools.export_perceptual_metrics47 import official_schema
    from tools.fusion_memory_evaluation import official_csv47
    plan, study, work, published, spec_path, spec, cases = inputs()
    expected = expected_jobs(cases)
    jobs = {phase: {key: dict(id=key, status='COMPLETE', exit_code=0, pid=index + 1)
                   for index, key in enumerate(ids)} for phase, ids in expected.items()}
    complete_jobs(jobs, expected)
    rejected = []
    changes = ['missing_negative_arm', 'worker_still_running', 'nonnull_failure_exit',
               'unknown_exit', 'boolean_exit_is_not_numeric_exit', 'unexpected_case']
    phase = PHASE + '_OFFICIAL'
    negative = next(key for key in jobs[phase] if key.endswith('_off'))
    for change in changes:
        corrupt = copy.deepcopy(jobs)
        if change == 'missing_negative_arm': del corrupt[phase][negative]
        elif change == 'worker_still_running': corrupt[phase][negative]['status'] = 'RUNNING'
        elif change == 'nonnull_failure_exit': corrupt[phase][negative]['exit_code'] = 1
        elif change == 'unknown_exit': corrupt[phase][negative]['exit_code'] = None
        elif change == 'boolean_exit_is_not_numeric_exit': corrupt[phase][negative]['exit_code'] = False
        else: corrupt[phase]['invented'] = dict(id='invented', status='COMPLETE', exit_code=0, pid=1)
        try:
            complete_jobs(corrupt, expected)
        except ValueError:
            rejected.append(change)
        else:
            raise ValueError('Invalid queue evidence was accepted: ' + change)
    old_path = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261126_ADDITIONAL_PUBLIC_TRACKERS_COMPLETE.json'
    old = read(old_path)
    source = next(row for row in old['rows'] if row['split'] == 'dev' and row['method'] == 'botsort' and
                  row['stream'] == 'primary_IR')
    csv_path = ROOT / source['provenance']['official_CSV']
    schema, counts = official_schema(plan['evaluation']['TrackEval_root'])
    detail = official_csv47(csv_path, schema, 'dev')
    require(sha(csv_path) == source['official_CSV_sha256'] and
            detail['combined47'] == source['combined47'] and detail['per_sequence47'] == source['per_sequence47'],
            'Real previously published full Dev47 conversion changed')
    with tempfile.TemporaryDirectory(prefix='native_dev_export_engineering_', dir=work) as folder:
        bad_csv = Path(folder) / 'partial.csv'
        lines = csv_path.read_text().splitlines(keepends=True)
        bad_csv.write_text(''.join(lines[:-1]))
        try:
            official_csv47(bad_csv, schema, 'dev')
        except ValueError:
            rejected.append('partial_real_Dev_CSV_without_COMBINED')
        else:
            raise ValueError('Partial actual CSV was accepted')
    live_ready = readiness(work, cases)[0]
    live_rejected = None
    if live_ready['status'].startswith('WAITING_'):
        try:
            publish()
        except ValueError as error:
            require(str(error) == 'Wait for every original full Dev worker and frozen decision',
                    'Live-stage gate failed for an unrelated reason')
            live_rejected = True
        else:
            raise ValueError('Actual incomplete native stage was published')
        docs = ROOT / 'docs/fusion_research'
        require(not (docs / 'perceptual_20261126_native_minimum6_full_dev').exists() and
                not (docs / 'PERCEPTUAL_MFT_20261126_NATIVE_MINIMUM6_FULL_DEV.json').exists(),
                'Incomplete live stage created publication files')
    print(json.dumps(dict(status='COMPLETE_CPU_EXPORT_ENGINEERING_NOT_NEW_NATIVE_ACCURACY',
        negative_cases_rejected=rejected, actual_previous_full_Dev47_byte_bound=True,
        synthetic_queue_fixtures_are_not_actual_completion_receipts=True,
        actual_incomplete_stage_publication_rejected_before_output=live_rejected,
        actual_previous_report_sha256=sha(old_path), actual_previous_CSV_sha256=sha(csv_path),
        prepublished_new_cases=len(cases), original_current_stage_ready=live_ready,
        GPU_work=False, Test_results_read=False, new_native_accuracy_claimed=False,
        publication_files_created=False, goal_met=False)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check-ready', action='store_true')
    action.add_argument('--engineering', action='store_true')
    action.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit empty CUDA_VISIBLE_DEVICES required')
    if args.check_ready:
        plan, study, work, published, spec_path, spec, cases = inputs()
        print(json.dumps(readiness(work, cases)[0]), flush=True)
    elif args.engineering:
        engineering()
    else:
        publish()


if __name__ == '__main__':
    main()
