#!/usr/bin/env python3
"""Frozen native-IR parent, three fixed causal-memory weights, Dev-only selection.

This is the explicit reduced user scope of 2026-11-26. It does not complete,
release or fabricate evidence for the older two-family F4/control study.
Actual completion/exits of the current original-baseline Test own GPU priority.
"""
import argparse
import copy
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.fusion_memory_evaluation import evaluate_observations, official_csv47
from tools.fusion_memory_observation import FixedSequenceImages, LENGTHS, observe_sequence
from tools.fusion_memory_online import OnlineFusionMemory, visual_routes
from tools.fusion_memory_model_loading import load_fusion_snapshot_cpu, load_source_rgb_cpu
from tools.launch_native_pyramid_study import Queue
from tools.native_pyramid_experiment import context, frozen_json, gpu_settings, read, registry
from tools.native_pyramid_online_inference import OnlineNativeDetector
from tools.native_pyramid_stream_tracks import IndependentByteTrack, mot_line
from tools.native_pyramid_training_loop import guard_storage
from tools.neck72_runtime import require_host_memory
from tools.probe_mft_native_memory_priority import SPEC, tensor_digest, visual_and_cases
from tools.run_pseudo_ir_comparison import sha256

PREFIX = 'PMM_20261126'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def locations():
    plan, study, protocol, calibration, work, checkpoint = context()
    return plan, study, protocol, calibration, work, checkpoint, work / 'native_memory_priority_20261126'


def identity(pid):
    proc = Path('/proc') / str(pid)
    try:
        fields = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
        return dict(PID=pid, start_ticks=int(fields[19]), kernel_state=fields[0],
                    raw_argv_sha256=hashlib.sha256((proc / 'cmdline').read_bytes()).hexdigest())
    except (FileNotFoundError, ProcessLookupError):
        return None


def registered():
    plan, study, protocol, calibration, work, checkpoint, base = locations()
    registry(work)
    value = read(base / 'pipeline_registered.json')
    assert value['status'] == 'REGISTERED_NATIVE_IR_ONLY_MEMORY_PRIORITY'
    assert value['specification_sha256'] == sha256(SPEC)
    for name, expected in value['source_hashes'].items():
        assert sha256(ROOT / name) == expected, name
    for name, expected in value['artifact_hashes'].items():
        assert sha256(name) == expected, name
    return value


def predecessor_ready(work, *, required=False):
    base = work / 'original_baseline_priority_20261126'
    execution = read(base / 'controller.execution.json')
    state = read(base / 'state.json')
    if 'FAILED' in state['status'] or execution.get('actual_child_exit_code') not in (None, 0):
        raise RuntimeError('Preserve actual original-baseline priority failure before memory CUDA.')
    expected = execution['identity']
    current = identity(expected['PID'])
    same_live = (current is not None and current['start_ticks'] == expected['start_ticks'] and
                 current['kernel_state'] != 'Z')
    ready = (not same_live and type(execution.get('actual_child_exit_code')) is int and
             execution['actual_child_exit_code'] == 0 and
             state['status'] == 'COMPLETE_PRIMARY_ORIGINAL_BASELINE_TEST_REQUIRES_ASSESSMENT' and
             (base / 'full_Test_results.json').is_file() and state['phases'] and
             all(rows and all(j['status'] == 'COMPLETE' and type(j.get('exit_code')) is int and
                             j['exit_code'] == 0 and not identity(j['pid'])
                             for j in rows.values()) for rows in state['phases'].values()))
    if required:
        assert ready, 'Current original-baseline Test retains exclusive GPU priority.'
    return bool(ready)


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plan, study, protocol, calibration, work, checkpoint, base = locations()
    assert not (base / 'pipeline_registered.json').exists()
    spec = read(SPEC)
    assert spec['appearance_weights'] == [0., .05, .1] and spec['fixed_seed'] == 0
    proof_path, execution_path = base / 'CPU_engineering_r1.json', base / 'CPU_engineering_r1.execution.json'
    proof, execution = read(proof_path), read(execution_path)
    assert proof['status'] == 'COMPLETE_REAL_TRAIN_CPU_NATIVE_MEMORY_BRIDGE_ENGINEERING_NOT_GPU_QUALIFICATION'
    assert proof['specification_sha256'] == sha256(SPEC)
    assert execution['actual_child_exit_code'] == 0 and execution['source_unchanged'] is True
    assert proof['source_sha256'] == execution['source_sha256'] == sha256(ROOT / 'tools/probe_mft_native_memory_priority.py')
    assert sha256(execution['log']) == execution['log_sha256']
    visual, cases, arm, progress = visual_and_cases(spec, work)
    sources = {}
    for name in ['native_pyramid_pipeline_registered.json', 'task_utility/pipeline_registered.json']:
        for source, expected in read(work / name)['source_hashes'].items():
            assert source not in sources or sources[source] == expected
            assert sha256(ROOT / source) == expected
            sources[source] = expected
    for name in ['tools/run_mft_native_memory_priority.py', 'tools/probe_mft_native_memory_priority.py', str(SPEC.relative_to(ROOT))]:
        sources[name] = sha256(ROOT / name)
    artifacts = {str(path): sha256(path) for path in [proof_path, execution_path,
        Path(execution['log']), work / 'native_pyramid_normal_selection_frozen.json',
        ROOT / spec['parent_Dev_report'], Path(visual['model']['checkpoint']), checkpoint,
        work / 'correlation_cache/calibration.json']}
    anchors = {}
    for sequence in LENGTHS['dev']:
        path = work / 'native_pyramid_streams/dev_normal_06/sequence_manifests' / (sequence + '.json')
        value = read(path)
        assert value['GT_read'] is False and value['frames'] == LENGTHS['dev'][sequence]
        assert visual['id'] in value['streams']
        anchors[sequence] = dict(path=str(path), sha256=sha256(path))
        artifacts[str(path)] = sha256(path)
    value = dict(status='REGISTERED_NATIVE_IR_ONLY_MEMORY_PRIORITY', UTC=utc(),
        specification_sha256=sha256(SPEC), source_hashes=sources, artifact_hashes=artifacts,
        visual_parent=visual, normal_cases=cases, Dev_image_anchors=anchors,
        scope='Single already Dev-frozen native IR parent; original RGB baseline remains fixed; no extra RGB architecture or removal fit.',
        original_broader_F4_completed=False, new_NN_fits=0, fixed_seed=0,
        physical_gpus=['4', '5'], maximum_cards=2, supported_innovations=0, goal_met=False)
    frozen_json(base / 'pipeline_registered.json', value)
    make_specification('dev', cases, anchors)
    print(json.dumps(dict(status=value['status'], source_files=len(sources), cases=len(cases),
                         GPU_work=False, Test_accuracy_read=False)), flush=True)


def make_specification(split, cases, anchors):
    plan, study, protocol, calibration, work, checkpoint, base = locations()
    path = base / 'specifications' / (split + '.json')
    value = dict(status='FROZEN_NATIVE_MEMORY_PRIORITY_BEFORE_CURRENT_SEQUENCE_PRODUCTION',
        split=split, cases=cases, stream_directory=str(base / 'streams' / split),
        image_source_anchors=anchors, pipeline_registration_sha256=sha256(base / 'pipeline_registered.json'),
        selected_from_complete_Dev_only=True, original_broader_F4_completed=False)
    frozen_json(path, value)
    return path


def load_models(cases, study, checkpoint, work, spec):
    import torch
    visual, all_cases, arm, progress = visual_and_cases(spec, work)
    assert all(case in all_cases for case in cases)
    routes, _ = visual_routes(cases)
    assert len(routes) == 1
    teacher = load_source_rgb_cpu(checkpoint, study['source_RGB_sha256'])
    module = load_fusion_snapshot_cpu(visual['model']['checkpoint'], visual['model']['checkpoint_sha256'],
                                      'native', visual['model'], arm, progress)
    return teacher, {next(iter(routes)): module}, visual


def engineering():
    import numpy as np
    import torch
    registration = registered()
    plan, study, protocol, calibration, work, checkpoint, base = locations()
    predecessor_ready(work, required=True)
    require_host_memory(8 * 2**30, maximum_workers=2)
    teacher, modules, visual = load_models(registration['normal_cases'], study, checkpoint, work, read(SPEC))
    reference_module = copy.deepcopy(next(iter(modules.values())))
    physical, uuid = gpu_settings()
    teacher = teacher.cuda().half().eval().requires_grad_(False)
    modules = {key: value.cuda().float().eval().requires_grad_(False) for key, value in modules.items()}
    reference_module = reference_module.cuda().float().eval().requires_grad_(False)
    before = dict(teacher=tensor_digest(teacher), native=tensor_digest(next(iter(modules.values()))))
    cases = registration['normal_cases']
    engine = OnlineFusionMemory(teacher, cases, modules, (800, 1440), calibration)
    reference = OnlineNativeDetector(teacher, [visual['model']], {visual['id']: reference_module}, (800, 1440), calibration)
    tracker = IndependentByteTrack()
    images = FixedSequenceImages(plan, 'train', 'BT-003', frames=12)
    started = time.monotonic()
    records, total = [], 0
    try:
        for frame in range(1, 13):
            rgb, ir, source = images.read_frame(frame, True)
            expected, _ = reference.step(rgb, ir, 'BT-003', frame)
            expected = expected[visual['id']]
            original_mot = ''.join(mot_line(row) for row in tracker.update(expected, frame))
            rows, packets, _ = engine.step(rgb, ir, 'BT-003', frame)
            assert all(np.array_equal(value, expected) for value in packets.values())
            assert ''.join(mot_line(row) for row in rows[cases[0]['id']]) == original_mot
            total += len(expected)
            records.append(dict(frame=frame, detections=len(expected), zero_MOT_exact=True, packets_exact=True, source=source))
        assert total > 0
        assert tensor_digest(teacher) == before['teacher']
        assert tensor_digest(next(iter(modules.values()))) == before['native']
        statistics = {key: dict(value.statistics) for key, value in engine.trackers.items()}
        assert all(statistics[c['id']]['changed_edges'] > 0 for c in cases[1:])
        registered()
        result = dict(status='COMPLETE_REAL_GPU_NATIVE_MEMORY_PRIORITY_ENGINEERING', UTC=utc(),
            physical_gpu=physical, GPU_uuid=str(uuid), actual_identity=identity(os.getpid()), frames=12,
            input_hw=[800, 1440], records=records, memory_statistics=statistics,
            source_teacher_and_native_weights_unchanged=True, zero_visual_and_literal_MOT_exact=True,
            all_memory_cases_same_actual_detector=True, GT_read=False, Test_accuracy_read=False,
            no_training_or_optimizer_updates=True, seconds=time.monotonic()-started,
            peak_gpu_GiB=torch.cuda.max_memory_allocated() / 2**30,
            pipeline_registration_sha256=sha256(base / 'pipeline_registered.json'))
        frozen_json(base / ('gpu_engineering_' + physical + '.json'), result)
        print(json.dumps(dict(status=result['status'], physical_gpu=physical, frames=12)), flush=True)
    finally:
        engine.close()


def require_engineering(base):
    state = read(base / 'state.json')
    jobs = state['phases'][PREFIX + '_GPU_ENGINEERING']
    assert set(jobs) == {'4', '5'}
    for physical, job in jobs.items():
        assert job['status'] == 'COMPLETE' and type(job.get('exit_code')) is int and job['exit_code'] == 0
        proof = read(base / ('gpu_engineering_' + physical + '.json'))
        assert proof['status'] == 'COMPLETE_REAL_GPU_NATIVE_MEMORY_PRIORITY_ENGINEERING'
        assert proof['pipeline_registration_sha256'] == sha256(base / 'pipeline_registered.json')


def specification(split):
    registration = registered()
    plan, study, protocol, calibration, work, checkpoint, base = locations()
    path = base / 'specifications' / (split + '.json')
    spec = read(path)
    assert spec['split'] == split and spec['pipeline_registration_sha256'] == sha256(base / 'pipeline_registered.json')
    assert all(case in registration['normal_cases'] for case in spec['cases'])
    assert set(spec['image_source_anchors']) == set(LENGTHS[split])
    if split == 'dev':
        assert spec['cases'] == registration['normal_cases']
    else:
        freeze = read(base / 'Test_selection_frozen.json')
        assert freeze['specification_sha256'] == sha256(path) and freeze['Test_read_for_selection'] is False
        assert freeze['complete_Dev_sha256'] == sha256(base / 'complete_Dev.json')
        assert spec['cases'] == freeze['cases']
    return registration, path, spec


def produce(split, sequence):
    import torch
    registration, path, spec = specification(split)
    plan, study, protocol, calibration, work, checkpoint, base = locations()
    predecessor_ready(work, required=True)
    require_engineering(base)
    assert sequence in LENGTHS[split]
    require_host_memory(8 * 2**30, maximum_workers=2)
    teacher, modules, visual = load_models(spec['cases'], study, checkpoint, work, read(SPEC))
    physical, uuid = gpu_settings()
    teacher = teacher.cuda().half().eval().requires_grad_(False)
    modules = {key: value.cuda().float().eval().requires_grad_(False) for key, value in modules.items()}
    engine = OnlineFusionMemory(teacher, spec['cases'], modules, (800, 1440), calibration)
    anchor = spec['image_source_anchors'][sequence]
    assert sha256(anchor['path']) == anchor['sha256']
    expected = read(anchor['path'])['original_image_bytes_delivery_sha256']
    provenance = dict(specification_sha256=sha256(path),
        pipeline_registration_sha256=sha256(base / 'pipeline_registered.json'),
        source_versions=registration['source_hashes'], physical_gpu=physical, GPU_uuid=str(uuid),
        actual_producer_identity=identity(os.getpid()),
        scope='Previously qualified native IR primary only; current user baseline-priority scope.',
        original_broader_F4_completed=False, GT_input=False, future_input=False)
    def resources():
        guard_storage(work)
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20 < 8
    result = observe_sequence(engine, FixedSequenceImages(plan, split, sequence), spec['cases'],
        spec['stream_directory'], expected, provenance, resource_check=resources)
    original = read(anchor['path'])['streams'][visual['id']]
    for case in spec['cases']:
        assert result['streams'][case['id']]['detection_packet_sha256'] == original['detection_packet_sha256']
        if case['appearance_weight'] == 0:
            assert result['streams'][case['id']]['track_sha256'] == original['track_sha256']
    registered()
    frozen_json(base / 'producer_validation' / split / (sequence + '.json'),
        dict(status='COMPLETE_NATIVE_MEMORY_SAME_DETECTOR_AND_ZERO_MOT_ANCHOR_CHECK',
             all_case_detector_packets_exact_to_native_parent=True, zero_MOT_bytes_exact=True,
             sequence=sequence, split=split, frames=LENGTHS[split][sequence],
             producer_sha256=sha256(Path(spec['stream_directory']) / 'sequence_manifests' / (sequence + '.json'))))
    print(json.dumps(dict(status=result['status'], sequence=sequence, frames=result['frames'], physical_gpu=physical)), flush=True)


def evaluate(split, case_id):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    registration, path, spec = specification(split)
    plan, study, protocol, calibration, work, checkpoint, base = locations()
    cases = [case for case in spec['cases'] if case['id'] == case_id]
    assert len(cases) == 1
    for sequence in LENGTHS[split]:
        validation = read(base / 'producer_validation' / split / (sequence + '.json'))
        assert validation['zero_MOT_bytes_exact'] and validation['all_case_detector_packets_exact_to_native_parent']
    result = evaluate_observations(path, cases[0], spec['stream_directory'],
        sha256(base / 'pipeline_registered.json'), registration['source_hashes'],
        spec['image_source_anchors'], base / 'evaluation' / split / case_id, plan)
    registered()
    print(json.dumps(dict(status=result['status'], case=case_id, split=split)), flush=True)


class PriorityQueue(Queue):
    def __init__(self, work, state, base):
        super().__init__(work, state)
        self.path = base / 'state.json'

    def save(self):
        self.state['UTC'] = utc()
        super().save()


def run_phase(queue, split):
    registration, path, spec = specification(split)
    plan, study, protocol, calibration, work, checkpoint, base = locations()
    command = [sys.executable, '-u', str(Path(__file__).relative_to(ROOT))]
    tasks = [dict(id=sequence, command=command + ['produce', '--split', split, '--sequence', sequence])
             for sequence in sorted(LENGTHS[split], key=lambda key: -LENGTHS[split][key])]
    def producer_checked(job):
        value = read(base / 'producer_validation' / split / (job['id'] + '.json'))
        assert value['zero_MOT_bytes_exact'] and value['all_case_detector_packets_exact_to_native_parent']
        assert value['frames'] == LENGTHS[split][job['id']]
    queue.jobs(PREFIX + '_' + split.upper() + '_NN', tasks, True, producer_checked)
    tasks = [dict(id=case['id'], command=command + ['evaluate', '--split', split, '--case', case['id']]) for case in spec['cases']]
    def evaluation_checked(job):
        value = read(base / 'evaluation' / split / job['id'] / 'metrics.json')
        assert value['status'] == 'COMPLETE_F4_OFFICIAL_METRICS'
        assert value['manifest']['frames'] == sum(LENGTHS[split].values())
    queue.jobs(PREFIX + '_' + split.upper() + '_OFFICIAL', tasks, False, evaluation_checked)
    return [read(base / 'evaluation' / split / case['id'] / 'metrics.json') for case in spec['cases']]


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    registration = registered()
    plan, study, protocol, calibration, work, checkpoint, base = locations()
    with (base / 'state.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (base / 'state.json').exists(), 'Inspect an earlier original attempt instead of restarting.'
        state = dict(status='WAITING_FOR_ORIGINAL_BASELINE_PRIORITY_TEST_ACTUAL_EXIT',
            controller_pid=os.getpid(), phases={}, data_root=plan['data_root'], fixed_seed=0,
            physical_gpus=['4', '5'], maximum_cards=2, additional_seed_work=False,
            scope='Native IR parent only, three memory weights; extra RGB architectures/removals deferred.',
            supported_innovations=0, goal_met=False)
        queue = PriorityQueue(work, state, base)
        queue.save()
        try:
            while not predecessor_ready(work):
                registered()
                time.sleep(15)
            command = [sys.executable, '-u', str(Path(__file__).relative_to(ROOT))]
            def checked(job):
                value = read(base / ('gpu_engineering_' + job['id'] + '.json'))
                assert value['status'] == 'COMPLETE_REAL_GPU_NATIVE_MEMORY_PRIORITY_ENGINEERING'
                assert value['physical_gpu'] == job['id'] and value['zero_visual_and_literal_MOT_exact']
            queue.jobs(PREFIX + '_GPU_ENGINEERING', [dict(id=gpu, physical_gpu=gpu, command=command + ['engineering'])
                       for gpu in ['4', '5']], True, checked)
            values = run_phase(queue, 'dev')
            zero = next(value for value in values if value['case']['appearance_weight'] == 0)
            parent = read(ROOT / read(SPEC)['parent_Dev_report'])
            expected = next(row for row in parent['rows'] if row['id'] == registration['visual_parent']['id'])
            assert zero['combined47'] == expected['combined47']
            # The parser includes COMBINED in per_sequence47; the parent stores only videos.
            assert zero['per_sequence47'][:-1] == expected['per_sequence47']
            chosen = min(values, key=lambda value: (*(-value['combined47'][key] for key in ['HOTA', 'AssA', 'IDF1']),
                                                     value['case']['appearance_weight']))
            gain = chosen['combined47']['HOTA'] - zero['combined47']['HOTA']
            complete = dict(status='COMPLETE_ALL3_NATIVE_MEMORY_PRIORITY_DEV', UTC=utc(),
                rows=values, chosen_case=chosen['case'], Dev_gain_over_identical_visual_parent_pp=gain,
                zero_all47_and_per_sequence_exact=True, Test_read_for_selection=False,
                new_NN_training=False, supported_innovations=0, goal_met=False)
            frozen_json(base / 'complete_Dev.json', complete)
            if chosen['case']['appearance_weight'] == 0 or gain <= 0:
                state.update(status='COMPLETE_NATIVE_MEMORY_NO_POSITIVE_DEV_GAIN_NO_NEW_TEST',
                             complete_Dev_sha256=sha256(base / 'complete_Dev.json'), Dev_gain_pp=gain)
                queue.save()
                print(json.dumps(dict(status=state['status'], Dev_gain_pp=gain)), flush=True)
                return
            anchors = {}
            for sequence in LENGTHS['test']:
                path = work / 'native_pyramid_streams/test_original_baseline_priority_20261126/sequence_manifests' / (sequence + '.json')
                value = read(path)
                assert value['GT_read'] is False and value['frames'] == LENGTHS['test'][sequence]
                anchors[sequence] = dict(path=str(path), sha256=sha256(path))
            cases = [zero['case'], chosen['case']]
            spec_path = make_specification('test', cases, anchors)
            frozen_json(base / 'Test_selection_frozen.json', dict(status='FROZEN_NATIVE_MEMORY_FROM_COMPLETE_DEV_BEFORE_TEST',
                UTC=utc(), cases=cases, chosen_case=chosen['case'],
                complete_Dev_sha256=sha256(base / 'complete_Dev.json'), specification_sha256=sha256(spec_path),
                Test_read_for_selection=False, historical_Test_previously_exposed=True,
                source_native_normal_selection_unchanged=True, supported_innovations=0, goal_met=False))
            test_values = run_phase(queue, 'test')
            from tools.export_perceptual_metrics47 import official_schema
            source = work / 'native_pyramid_evaluation/test/NP_source_canonical_RGB'
            original = read(source / 'metrics.json')
            schema, _ = official_schema(plan['evaluation']['TrackEval_root'])
            baseline = official_csv47(source / 'trackeval_output/pedestrian_detailed.csv', schema, 'test', original['trackeval'])
            selected = next(value for value in test_values if value['case']['id'] == chosen['case']['id'])
            delta = selected['combined47']['HOTA'] - baseline['combined47']['HOTA']
            assert abs(baseline['combined47']['HOTA'] - 29.5289636096556) < 1e-9
            report = dict(status='COMPLETE_DEV_FROZEN_NATIVE_MEMORY_FULL7_TEST', UTC=utc(), rows=test_values,
                original_RGB_baseline=baseline, primary_case=chosen['case'],
                Test_selection_sha256=sha256(base / 'Test_selection_frozen.json'),
                HOTA_gain_over_original_RGB_pp=delta, target_plus5_met=delta >= 5, target_plus10_met=delta >= 10,
                Test_used_for_selection=False, historical_Test_previously_exposed=True,
                prior_broader_F4_not_completed=True, supported_innovations=0, goal_met=False)
            frozen_json(base / 'full_Test_results.json', report)
            state.update(status='COMPLETE_NATIVE_MEMORY_TEST_REQUIRES_SCIENTIFIC_ASSESSMENT',
                         report_sha256=sha256(base / 'full_Test_results.json'), HOTA_gain_over_original_RGB_pp=delta)
            queue.save()
            print(json.dumps(dict(status=state['status'], HOTA_gain_over_original_RGB_pp=delta)), flush=True)
        except Exception as error:
            state.update(status='FAILED_PRESERVE_NATIVE_MEMORY_PRIORITY_ATTEMPT', error=repr(error))
            queue.save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['register', 'run', 'engineering', 'produce', 'evaluate', 'preflight'])
    parser.add_argument('--split', choices=['dev', 'test'])
    parser.add_argument('--sequence')
    parser.add_argument('--case')
    args = parser.parse_args()
    if args.mode == 'register':
        register()
    elif args.mode == 'run':
        run()
    elif args.mode == 'engineering':
        engineering()
    elif args.mode == 'produce':
        assert args.split and args.sequence
        produce(args.split, args.sequence)
    elif args.mode == 'evaluate':
        assert args.split and args.case
        evaluate(args.split, args.case)
    else:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        value = registered()
        plan, study, protocol, calibration, work, checkpoint, base = locations()
        print(json.dumps(dict(status='VALIDATED_NATIVE_MEMORY_PRIORITY_CPU_PREFLIGHT',
            predecessor_actually_finished=predecessor_ready(work), normal_cases=len(value['normal_cases']),
            GPU_work=False, Test_accuracy_read=False)), flush=True)


if __name__ == '__main__':
    main()
