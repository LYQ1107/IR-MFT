#!/usr/bin/env python3
"""Reduced baseline-priority paired Fisher/identity-persistence study.

Reuses the qualified frozen November-IR visual parent and read-only low-level
observation/evaluation components. Does not release older deferred queues.
"""
import argparse
import copy
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.fusion_memory_evaluation import evaluate_observations, official_csv47
from tools.fusion_memory_model_loading import load_fusion_snapshot_cpu, load_source_rgb_cpu
from tools.fusion_memory_observation import FixedSequenceImages, LENGTHS, observe_sequence
from tools.mft_fisher_online import FisherOnline, fixed_cases
from tools.native_pyramid_experiment import context, frozen_json, gpu_settings, read
from tools.native_pyramid_online_inference import OnlineNativeDetector
from tools.native_pyramid_stream_tracks import IndependentByteTrack, mot_line
from tools.native_pyramid_training_loop import guard_storage
from tools.probe_mft_native_memory_priority import SPEC as PARENT_SPEC, tensor_digest, visual_and_cases
from tools.run_mft_native_memory_priority_v3 import PriorityQueue, identity, predecessor_ready
from tools.run_mft_native_memory_priority_v3 import registered as prior_registered
from tools.run_pseudo_ir_comparison import sha256

SPEC = ROOT / 'configs/perceptual_fisher_memory_20261127.json'
PREFIX = 'FISHER_20261127_R1'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def locations():
    values = context()
    return (*values, values[4] / 'fisher_memory_20261127')


def require_predecessors(work):
    predecessor_ready(work, required=True)
    base = work / 'native_memory_priority_20261126_v3'
    execution, state = read(base / 'controller.execution.json'), read(base / 'state.json')
    assert execution['actual_child_exit_code'] == 0 and not identity(execution['identity']['PID'])
    assert state['status'] == 'COMPLETE_NATIVE_MEMORY_NO_POSITIVE_DEV_GAIN_NO_NEW_TEST'
    assert all(j['status'] == 'COMPLETE' and type(j['exit_code']) is int and j['exit_code'] == 0
               and not identity(j['pid']) for phase in state['phases'].values() for j in phase.values())
    runtime = read(work / 'native_primary_individual_runtime_r1.execution.json')
    assert runtime['actual_child_exit_code'] == 0 and not identity(runtime['identity']['PID'])


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plan, study, protocol, calibration, work, checkpoint, base = locations()
    require_predecessors(work)
    parent = prior_registered()
    fit_path = base / 'fit_complete.json'
    fit = read(fit_path)
    execution = read(work / 'fisher_projection_fit_r3.execution.json')
    assert execution['actual_child_exit_code'] == 0 and execution['source_unchanged']
    assert not identity(execution['identity']['PID'])
    assert fit['status'] == 'COMPLETE_SAMPLED_TRAIN_ONLY_FISHER_PROJECTION'
    assert fit['Dev_fit_rows'] == fit['Test_fit_rows'] == 0
    assert fit['specification_sha256'] == sha256(SPEC)
    assert sha256(fit['model_path']) == fit['model_sha256']
    sources = dict(parent['source_hashes'])
    own = ['tools/run_mft_fisher_memory.py', 'tools/mft_fisher_online.py',
        'tools/mft_identity_projection.py', 'tools/fit_mft_identity_projection.py',
        'tools/fit_mft_identity_projection_r2.py',
        'configs/perceptual_fisher_memory_sparse_train_policy_20261127.json',
        'yolox/tracker/fisher_identity_tracker.py', str(SPEC.relative_to(ROOT))]
    for name in own:
        sources[name] = sha256(ROOT / name)
    for name, expected in sources.items():
        assert sha256(ROOT / name) == expected
    visual, _, _, _ = visual_and_cases(read(PARENT_SPEC), work)
    spec = read(SPEC)
    assert visual['id'] == spec['primary_visual_id']
    assert visual['model']['checkpoint_sha256'] == spec['primary_checkpoint_sha256']
    anchors = {}
    for seq in LENGTHS['dev']:
        p = work / 'native_pyramid_streams/dev_normal_06/sequence_manifests' / (seq+'.json')
        value = read(p)
        assert value['frames'] == LENGTHS['dev'][seq] and value['GT_read'] is False
        anchors[seq] = dict(path=str(p), sha256=sha256(p))
    artifacts = {str(p): sha256(p) for p in [fit_path, Path(fit['model_path']),
        base / 'fit_registered.json', base / 'fit_registered_r2.json', work / 'fisher_projection_fit_r1.execution.json',
        work / 'fisher_projection_fit_r2.execution.json', work / 'fisher_projection_fit_r3.execution.json',
        work / 'native_primary_individual_runtime_r1.execution.json',
        work / 'native_memory_priority_20261126_v3/controller.execution.json',
        work / 'native_memory_priority_20261126_v3/complete_Dev.json']}
    value = dict(status='REGISTERED_PAIRED_FISHER_NATIVE_PRIMARY_ONLY_STUDY', UTC=utc(),
        specification_sha256=sha256(SPEC), source_hashes=sources, artifact_hashes=artifacts,
        visual_parent=visual, normal_cases=fixed_cases(spec), Dev_image_anchors=anchors,
        parent_registration_sha256=sha256(work / 'native_memory_priority_20261126_v3/pipeline_registered.json'),
        fit_sha256=sha256(fit_path), physical_gpus=['4', '5'], fixed_seed=0,
        supported_innovations=0, goal_met=False)
    assert not (base / 'pipeline_registered.json').exists()
    frozen_json(base / 'pipeline_registered.json', value)
    make_specification('dev', value['normal_cases'], anchors)
    print(json.dumps(dict(status=value['status'], source_files=len(sources), cases=len(value['normal_cases']))), flush=True)


def registered():
    plan, study, protocol, calibration, work, checkpoint, base = locations()
    prior_registered()
    value = read(base / 'pipeline_registered.json')
    assert value['status'] == 'REGISTERED_PAIRED_FISHER_NATIVE_PRIMARY_ONLY_STUDY'
    assert value['specification_sha256'] == sha256(SPEC)
    for name, expected in value['source_hashes'].items():
        assert sha256(ROOT / name) == expected, name
    for name, expected in value['artifact_hashes'].items():
        assert sha256(name) == expected, name
    return value


def make_specification(split, cases, anchors):
    plan, study, protocol, calibration, work, checkpoint, base = locations()
    path = base / 'specifications' / (split+'.json')
    value = dict(status='FROZEN_PAIRED_FISHER_CASES_BEFORE_SEQUENCE_PRODUCTION', split=split,
        cases=cases, stream_directory=str(base / 'streams' / split), image_source_anchors=anchors,
        pipeline_registration_sha256=sha256(base / 'pipeline_registered.json'),
        no_per_video_policy=True, fixed_seed=0, Test_read_for_selection=False)
    frozen_json(path, value)
    return path


def specification(split):
    registration = registered()
    plan, study, protocol, calibration, work, checkpoint, base = locations()
    path = base / 'specifications' / (split+'.json')
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


def load_models(study, checkpoint, work):
    import numpy as np
    visual, _, arm, progress = visual_and_cases(read(PARENT_SPEC), work)
    teacher = load_source_rgb_cpu(checkpoint, study['source_RGB_sha256'])
    module = load_fusion_snapshot_cpu(visual['model']['checkpoint'], visual['model']['checkpoint_sha256'],
                                    'native', visual['model'], arm, progress)
    fit = read(work / 'fisher_memory_20261127/fit_complete.json')
    assert sha256(fit['model_path']) == fit['model_sha256']
    with np.load(fit['model_path']) as saved:
        parameters = {key: saved[key].copy() for key in ['center', 'projection']}
    return teacher, module, visual, parameters


def engineering():
    import cv2
    import numpy as np
    import torch
    from tools.build_correlation_cache import auxiliary_input, read_pair
    from yolox.tracker.foveated_regions import exact_half_preproc
    registration = registered()
    plan, study, protocol, calibration, work, checkpoint, base = locations()
    require_predecessors(work)
    teacher, module, visual, parameters = load_models(study, checkpoint, work)
    reference_module = copy.deepcopy(module)
    physical, uuid = gpu_settings()
    teacher = teacher.cuda().half().eval().requires_grad_(False)
    module = module.cuda().float().eval().requires_grad_(False)
    reference_module = reference_module.cuda().float().eval().requires_grad_(False)
    before = dict(teacher=tensor_digest(teacher), module=tensor_digest(module))
    cases = registration['normal_cases']
    engine = FisherOnline(teacher, module, visual['model'], cases, calibration, parameters, read(SPEC)['association'])
    reference = OnlineNativeDetector(teacher, [visual['model']], {visual['id']: reference_module}, (800, 1440), calibration)
    baseline = IndependentByteTrack()
    fit = read(base / 'fit_complete.json')
    anchors = []
    # Verify real immutable cache bytes, not only pathname provenance.
    sequences = protocol['train'][0::2] if physical == '4' else protocol['train'][1::2]
    started = time.monotonic()
    with torch.inference_mode():
        for seq in sequences:
            anchor = fit['cache_first_frame_anchors'][seq]
            rgb, ir = read_pair(plan, seq, anchor['frame'])
            rgb_value, _ = exact_half_preproc(rgb, (480, 864), (.485, .456, .406), (.229, .224, .225))
            norm = calibration['statistics']['ir']
            ir_value, _ = auxiliary_input(ir, (480, 864), norm['mean'], norm['std'])
            checks = {}
            for cue, value in [('rgb', rgb_value), ('ir', ir_value)]:
                feature = teacher.backbone(torch.from_numpy(value[None].copy()).cuda())[0]
                actual = hashlib.sha256(feature[0].cpu().numpy().tobytes()).hexdigest()
                assert actual == anchor['P3_sha256'][cue], (seq, cue, actual, anchor['P3_sha256'][cue])
                checks[cue] = actual
            anchors.append(dict(sequence=seq, frame=anchor['frame'], RGB_IR_P3_byte_exact=True, hashes=checks))
    images = FixedSequenceImages(plan, 'train', 'BT-003', frames=12)
    frame_checks, first = [], None
    try:
        for frame in range(1, 13):
            rgb, ir, _ = images.read_frame(frame, True)
            expected, _ = reference.step(rgb, ir, 'BT-003', frame)
            observed = expected[visual['id']]
            rows, packets, _ = engine.step(rgb, ir, 'BT-003', frame)
            assert all(np.array_equal(p, observed) for p in packets.values())
            assert ''.join(map(mot_line, rows[cases[0]['id']])) == ''.join(map(mot_line, baseline.update(observed, frame)))
            if frame == 1:
                first = (rgb.copy(), ir.copy(), observed.copy(), {k: ''.join(map(mot_line, v)) for k,v in rows.items()})
            for tracker in engine.trackers.values():
                for track in tracker.core.tracked_stracks+tracker.core.lost_stracks:
                    if track.memory is not None:
                        assert np.isfinite(track.memory).all() and abs(float(np.linalg.norm(track.memory))-1) < 1e-5
                    assert track.observation is None
            frame_checks.append(dict(frame=frame, detections=len(observed), zero_MOT_exact=True, all_detector_packets_exact=True))
        statistics = {key:dict(t.statistics) for key,t in engine.trackers.items()}
        other = FixedSequenceImages(plan, 'train', 'BT-001', frames=1)
        rgb, ir, _ = other.read_frame(1, True)
        engine.step(rgb, ir, 'BT-001', 1)
        rows, packets, _ = engine.step(first[0], first[1], 'BT-003', 1)
        assert all(np.array_equal(p, first[2]) for p in packets.values())
        assert {k: ''.join(map(mot_line, v)) for k,v in rows.items()} == first[3]
        try:
            engine.step(first[0], first[1], 'BT-003', 3)
        except ValueError:
            skipped_frame_refused = True
        else:
            raise AssertionError('A skipped future frame was accepted')
        assert tensor_digest(teacher) == before['teacher'] and tensor_digest(module) == before['module']
        registered()
        value = dict(status='COMPLETE_REAL_GPU_PAIRED_FISHER_ENGINEERING', UTC=utc(),
            physical_gpu=physical, GPU_uuid=str(uuid), actual_identity=identity(os.getpid()),
            real_Train_cache_anchors=anchors, frames=12, records=frame_checks, memory_statistics=statistics,
            zero_visual_and_literal_MOT_exact=True, source_and_fusion_weights_unchanged=True,
            sequence_reset_then_return_exact=True, skipped_future_frame_refused=skipped_frame_refused,
            GT_read=False, Test_accuracy_read=False, new_NN_training=False,
            seconds=time.monotonic()-started, pipeline_registration_sha256=sha256(base/'pipeline_registered.json'))
        frozen_json(base / ('gpu_engineering_'+physical+'.json'), value)
        print(json.dumps(dict(status=value['status'], physical_gpu=physical, frames=12, cache_sequences=len(anchors))), flush=True)
    finally:
        engine.close()


def require_engineering(base):
    jobs = read(base / 'state.json')['phases'][PREFIX+'_GPU_ENGINEERING']
    assert set(jobs) == {'4', '5'}
    for gpu, job in jobs.items():
        assert job['status'] == 'COMPLETE' and type(job['exit_code']) is int and job['exit_code'] == 0
        assert not identity(job['pid'])
        value = read(base / ('gpu_engineering_'+gpu+'.json'))
        assert value['status'] == 'COMPLETE_REAL_GPU_PAIRED_FISHER_ENGINEERING'
        assert value['zero_visual_and_literal_MOT_exact'] and value['sequence_reset_then_return_exact']


def produce(split, sequence):
    registration, path, spec = specification(split)
    plan, study, protocol, calibration, work, checkpoint, base = locations()
    require_predecessors(work)
    require_engineering(base)
    teacher, module, visual, parameters = load_models(study, checkpoint, work)
    physical, uuid = gpu_settings()
    teacher = teacher.cuda().half().eval().requires_grad_(False)
    module = module.cuda().float().eval().requires_grad_(False)
    engine = FisherOnline(teacher, module, visual['model'], spec['cases'], calibration, parameters, read(SPEC)['association'])
    anchor = spec['image_source_anchors'][sequence]
    assert sha256(anchor['path']) == anchor['sha256']
    original = read(anchor['path'])
    provenance = dict(specification_sha256=sha256(path), pipeline_registration_sha256=sha256(base/'pipeline_registered.json'),
        source_versions=registration['source_hashes'], physical_gpu=physical, GPU_uuid=str(uuid),
        actual_producer_identity=identity(os.getpid()), GT_input=False, future_input=False,
        scope='Native IR primary plus sampled-Train paired Fisher appearance; no new RGB architecture')
    value = observe_sequence(engine, FixedSequenceImages(plan, split, sequence), spec['cases'],
        spec['stream_directory'], original['original_image_bytes_delivery_sha256'], provenance,
        resource_check=lambda:guard_storage(work))
    parent = original['streams'][visual['id']]
    for case in spec['cases']:
        assert value['streams'][case['id']]['detection_packet_sha256'] == parent['detection_packet_sha256']
        if case['appearance_weight'] == 0:
            assert value['streams'][case['id']]['track_sha256'] == parent['track_sha256']
    registered()
    frozen_json(base/'producer_validation'/split/(sequence+'.json'), dict(status='COMPLETE_PAIRED_FISHER_PRODUCER_PARITY',
        sequence=sequence, split=split, frames=LENGTHS[split][sequence], all_detector_packets_exact=True,
        zero_MOT_bytes_exact=True, source_manifest_sha256=sha256(Path(spec['stream_directory'])/'sequence_manifests'/(sequence+'.json'))))
    print(json.dumps(dict(status=value['status'], sequence=sequence, frames=value['frames'], physical_gpu=physical)), flush=True)


def evaluate(split, case_id):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    registration, path, spec = specification(split)
    plan, study, protocol, calibration, work, checkpoint, base = locations()
    require_engineering(base)
    for sequence in LENGTHS[split]:
        value = read(base/'producer_validation'/split/(sequence+'.json'))
        assert value['all_detector_packets_exact'] and value['zero_MOT_bytes_exact']
    case = next(c for c in spec['cases'] if c['id'] == case_id)
    value = evaluate_observations(path, case, spec['stream_directory'], sha256(base/'pipeline_registered.json'),
        registration['source_hashes'], spec['image_source_anchors'], base/'evaluation'/split/case_id, plan)
    registered()
    print(json.dumps(dict(status=value['status'], split=split, case=case_id)), flush=True)


def run_phase(queue, split):
    registration, path, spec = specification(split)
    plan, study, protocol, calibration, work, checkpoint, base = locations()
    command = [sys.executable, '-u', str(Path(__file__).relative_to(ROOT))]
    tasks = [dict(id=seq, command=command+['produce','--split',split,'--sequence',seq])
        for seq in sorted(LENGTHS[split], key=lambda s:-LENGTHS[split][s])]
    def produced(job):
        value = read(base/'producer_validation'/split/(job['id']+'.json'))
        assert value['zero_MOT_bytes_exact'] and value['all_detector_packets_exact']
    queue.jobs(PREFIX+'_'+split.upper()+'_NN', tasks, True, produced)
    tasks = [dict(id=c['id'], command=command+['evaluate','--split',split,'--case',c['id']])for c in spec['cases']]
    def evaluated(job):
        value = read(base/'evaluation'/split/job['id']/'metrics.json')
        assert value['manifest']['frames'] == sum(LENGTHS[split].values())
    queue.jobs(PREFIX+'_'+split.upper()+'_OFFICIAL', tasks, False, evaluated)
    return [read(base/'evaluation'/split/c['id']/'metrics.json')for c in spec['cases']]


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    registration = registered()
    plan, study, protocol, calibration, work, checkpoint, base = locations()
    require_predecessors(work)
    with (base/'state.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (base/'state.json').exists(), 'Preserve prior attempts'
        state = dict(status='STARTING_PAIRED_FISHER_GPU_ENGINEERING', controller_pid=os.getpid(), phases={}, data_root=plan['data_root'],
            physical_gpus=['4','5'], maximum_cards=2, fixed_seed=0, additional_seed_work=False,
            user_priority='Original direct RGB baseline +5 HOTA; +10 stretch', supported_innovations=0, goal_met=False)
        queue = PriorityQueue(work, state, base)
        queue.save()
        try:
            command = [sys.executable,'-u',str(Path(__file__).relative_to(ROOT))]
            def checked(job):
                value = read(base/('gpu_engineering_'+job['id']+'.json'))
                assert value['status'] == 'COMPLETE_REAL_GPU_PAIRED_FISHER_ENGINEERING'
                assert value['zero_visual_and_literal_MOT_exact']
            queue.jobs(PREFIX+'_GPU_ENGINEERING', [dict(id=g,physical_gpu=g,command=command+['engineering'])for g in ['4','5']], True, checked)
            rows = run_phase(queue, 'dev')
            zero = next(r for r in rows if r['case']['appearance_weight'] == 0)
            parent = read(work/'native_memory_priority_20261126_v3/complete_Dev.json')['rows'][0]
            assert zero['combined47'] == parent['combined47'] and zero['per_sequence47'] == parent['per_sequence47']
            chosen = min(rows,key=lambda r:(*(-r['combined47'][k] for k in ['HOTA','AssA','IDF1']),r['case']['lost_buffer'],r['case']['appearance_weight']))
            gain = chosen['combined47']['HOTA']-zero['combined47']['HOTA']
            frozen_json(base/'complete_Dev.json',dict(status='COMPLETE_ALL5_PAIRED_FISHER_DEV',UTC=utc(),rows=rows,
                chosen_case=chosen['case'],Dev_gain_over_identical_visual_parent_pp=gain,
                zero_all47_and_per_sequence_exact=True,Test_read_for_selection=False,
                new_NN_training=False,supported_innovations=0,goal_met=False))
            if chosen['case']['appearance_weight'] == 0 or gain <= 0:
                state.update(status='COMPLETE_FISHER_NO_POSITIVE_DEV_GAIN_NO_NEW_TEST',Dev_gain_pp=gain)
                queue.save()
                return
            anchors = {}
            for sequence in LENGTHS['test']:
                p = work/'native_pyramid_streams/test_original_baseline_priority_20261126/sequence_manifests'/(sequence+'.json')
                assert read(p)['frames'] == LENGTHS['test'][sequence] and read(p)['GT_read'] is False
                anchors[sequence] = dict(path=str(p),sha256=sha256(p))
            cases = [zero['case'],chosen['case']]
            spec_path = make_specification('test',cases,anchors)
            frozen_json(base/'Test_selection_frozen.json',dict(status='FROZEN_PAIRED_FISHER_FROM_COMPLETE_DEV_BEFORE_TEST',UTC=utc(),
                cases=cases,chosen_case=chosen['case'],complete_Dev_sha256=sha256(base/'complete_Dev.json'),
                specification_sha256=sha256(spec_path),Test_read_for_selection=False,historical_Test_previously_exposed=True,
                supported_innovations=0,goal_met=False))
            test = run_phase(queue,'test')
            selected = next(r for r in test if r['case']['id'] == chosen['case']['id'])
            from tools.export_perceptual_metrics47 import official_schema
            folder = work/'native_pyramid_evaluation/test/NP_source_canonical_RGB'
            schema,_ = official_schema(plan['evaluation']['TrackEval_root'])
            csv = folder/'trackeval_output/pedestrian_detailed.csv'
            source = Path(plan['frozen_detection_sources']['rgb_test'])/'trackeval_output/pedestrian_detailed.csv'
            assert sha256(csv) == sha256(source) == 'c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
            baseline = official_csv47(csv,schema,'test',read(folder/'metrics.json')['trackeval'])
            delta = selected['combined47']['HOTA']-baseline['combined47']['HOTA']
            frozen_json(base/'full_Test_results.json',dict(status='COMPLETE_DEV_FROZEN_PAIRED_FISHER_FULL7_TEST',UTC=utc(),
                rows=test,original_RGB_baseline=baseline,primary_case=chosen['case'],HOTA_gain_over_original_RGB_pp=delta,
                target_plus5_met=delta>=5,target_plus10_met=delta>=10,Test_used_for_selection=False,
                historical_Test_previously_exposed=True,supported_innovations=0,goal_met=False))
            state.update(status='COMPLETE_FISHER_FULL_TEST_REQUIRES_SCIENTIFIC_ASSESSMENT',HOTA_gain_over_original_RGB_pp=delta)
            queue.save()
        except Exception as error:
            state.update(status='FAILED_PRESERVE_PAIRED_FISHER_ATTEMPT',error=repr(error))
            queue.save()
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['register','run','engineering','produce','evaluate'])
    p.add_argument('--split',choices=['dev','test'])
    p.add_argument('--sequence')
    p.add_argument('--case')
    a = p.parse_args()
    if a.mode == 'register': register()
    elif a.mode == 'run': run()
    elif a.mode == 'engineering': engineering()
    elif a.mode == 'produce': produce(a.split,a.sequence)
    elif a.mode == 'evaluate': evaluate(a.split,a.case)


if __name__ == '__main__':
    main()
