#!/usr/bin/env python3
"""Actual fixed-Train CPU engineering for the frozen native-IR memory bridge.

Reads no GT or Dev/Test metrics beyond the already frozen parent Dev record.
No CUDA, training, optimizer, new seed, image copy or feature cache is used.
CPU input geometry is smaller than formal inference; this is not GPU parity.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch

from tools.fusion_memory_design import make_case
from tools.fusion_memory_model_loading import load_fusion_snapshot_cpu, load_source_rgb_cpu
from tools.fusion_memory_observation import FixedSequenceImages
from tools.fusion_memory_online import OnlineFusionMemory, visual_routes
from tools.native_pyramid_design import snapshot_row
from tools.native_pyramid_experiment import context, read, registry
from tools.native_pyramid_online_inference import OnlineNativeDetector
from tools.native_pyramid_stream_tracks import IndependentByteTrack, mot_line
from tools.native_pyramid_training_loop import guard_storage
from tools.run_pseudo_ir_comparison import sha256

SPEC = ROOT / 'configs/perceptual_native_memory_priority_20261126.json'


def tensor_digest(module):
    h = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        h.update(key.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def visual_and_cases(spec, work):
    normal_path = work / 'native_pyramid_normal_selection_frozen.json'
    assert sha256(normal_path) == spec['original_normal_selection_sha256']
    selected = read(normal_path)
    assert selected['Test_read_for_selection'] is False
    original = selected['selection']['primary']
    assert original['id'] == spec['primary_id'] == 'NP_temporal_ir_e06'
    arm = read(ROOT / 'configs/perceptual_native_pyramid_fit_cases_20261124/temporal_ir.json')
    assert sha256(original['checkpoint']) == original['checkpoint_sha256']
    progress = read(work / 'native_pyramid_fits/temporal_ir/progress.json')
    assert progress['completed_full_Train_epochs'] == 9
    model = snapshot_row(work, original['fit_id'], original['epoch'])
    visual = dict(family='native', id=model['id'], cue='ir', baseline=False, model=model,
                  module_kwargs=arm['module_kwargs'])
    cases = [make_case(visual, 'byte' if weight == 0 else 'fused_uniform', weight)
             for weight in spec['appearance_weights']]
    return visual, cases, arm, progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    assert not args.output.exists(), 'Preserve an earlier actual attempt.'
    started = time.monotonic()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    spec = read(SPEC)
    plan, study, protocol, calibration, work, checkpoint = context()
    registry(work)
    assert spec['primary_id'] == 'NP_temporal_ir_e06'
    assert spec['appearance_weights'] == [0.0, 0.05, 0.1]
    assert spec['formal_input_hw'] == [800, 1440]
    assert sha256(ROOT / spec['parent_Dev_report']) == spec['parent_Dev_report_sha256']
    initial_spec_sha = sha256(SPEC)
    initial_source_sha = sha256(Path(__file__))
    storage = guard_storage(work)
    visual, cases, arm, progress = visual_and_cases(spec, work)
    routes, _ = visual_routes(cases)
    assert len(routes) == 1
    route_id = next(iter(routes))
    module = load_fusion_snapshot_cpu(visual['model']['checkpoint'], visual['model']['checkpoint_sha256'],
                                      'native', visual['model'], arm, progress)
    reference_module = copy.deepcopy(module)
    teacher = load_source_rgb_cpu(checkpoint, study['source_RGB_sha256'])
    before = dict(teacher=tensor_digest(teacher), auxiliary=tensor_digest(module))
    engine = OnlineFusionMemory(teacher, cases, {route_id: module}, spec['CPU_probe_input_hw'], calibration)
    reference = OnlineNativeDetector(teacher, [visual['model']], {visual['id']: reference_module},
                                     spec['CPU_probe_input_hw'], calibration)
    baseline = IndependentByteTrack()
    images = FixedSequenceImages(plan, 'train', 'BT-003', frames=3)
    fields, first_rows, first_packet, source_rows = [], None, None, []
    accepted_memories, normalized_memories = 0, 0
    try:
        for frame in range(1, 4):
            rgb, ir, source = images.read_frame(frame, True)
            source_rows.append(source)
            expected, _ = reference.step(rgb, ir, 'BT-003', frame)
            native = expected[visual['id']]
            original_mot = ''.join(mot_line(row) for row in baseline.update(native, frame))
            rows, packets, metadata = engine.step(rgb, ir, 'BT-003', frame)
            assert all(np.array_equal(value, native) for value in packets.values())
            zero_mot = ''.join(mot_line(row) for row in rows[cases[0]['id']])
            assert zero_mot == original_mot
            if frame == 1:
                first_packet = native.copy()
                first_rows = {key: ''.join(mot_line(row) for row in value) for key, value in rows.items()}
                first_pixels = rgb.copy(), ir.copy()
            for tracker in engine.trackers.values():
                for track in tracker.core.tracked_stracks + tracker.core.lost_stracks:
                    value = getattr(track, 'fused_memory', None)
                    if value is not None:
                        accepted_memories += 1
                        assert np.isfinite(value).all() and abs(float(np.linalg.norm(value)) - 1) < 1e-5
                        normalized_memories += 1
                    assert getattr(track, 'observation', None) is None
            fields.append(dict(frame=frame, detections=len(native), zero_literal_MOT_exact=True,
                every_memory_case_detector_packet_exact=True, GT_read=False, future_input=False))
        assert sum(row['detections'] for row in fields) > 0
        assert accepted_memories > 0
        statistics = {key: dict(tracker.statistics) for key, tracker in engine.trackers.items()}
        alternate = FixedSequenceImages(plan, 'train', 'BT-001', frames=1)
        rgb, ir, source = alternate.read_frame(1, True)
        source_rows.append(source)
        engine.step(rgb, ir, 'BT-001', 1)
        rows, packets, _ = engine.step(*first_pixels, 'BT-003', 1)
        assert all(np.array_equal(value, first_packet) for value in packets.values())
        assert {key: ''.join(mot_line(row) for row in value) for key, value in rows.items()} == first_rows
        assert tensor_digest(teacher) == before['teacher'] and tensor_digest(module) == before['auxiliary']
        assert not torch.cuda.is_initialized()
        assert sha256(SPEC) == initial_spec_sha and sha256(Path(__file__)) == initial_source_sha
        registry(work)
        guard_storage(work)
        result = dict(status='COMPLETE_REAL_TRAIN_CPU_NATIVE_MEMORY_BRIDGE_ENGINEERING_NOT_GPU_QUALIFICATION',
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_sha256=initial_source_sha,
            specification_sha256=initial_spec_sha, fixed_visual_parent=visual, cases=cases,
            input_hw=spec['CPU_probe_input_hw'], formal_input_hw=spec['formal_input_hw'],
            original_Train_prefix_frames=3, different_original_Train_source_frames=4,
            memory_engine_forward_calls=5, reference_visual_forward_calls=3,
            source_rows=source_rows, frame_checks=fields, memory_statistics=statistics,
            nonempty_memory_state_observations_including_repeats=accepted_memories,
            finite_unit_memory_state_observations=normalized_memories,
            switch_sequence_then_return_first_packet_and_all_MOT_exact=True,
            source_teacher_and_fusion_parameters_unchanged=True, actual_GPU_work=False,
            GT_read=False, Test_accuracy_read=False, parent_frozen_Dev_selection_read=True,
            new_Dev_or_Test_accuracy=False, training_or_optimizer_updates=0,
            new_seed_work=False, fixed_seed=0, new_image_weight_or_feature_disk_copies=0,
            formal_GPU_probe_still_required=True, memory_accuracy_benefit_established=False,
            supported_innovations=0, goal_met=False, initial_storage=storage,
            actual_wall_seconds=time.monotonic()-started,
            peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(dict(status=result['status'], accepted_memories=accepted_memories,
            actual_wall_seconds=result['actual_wall_seconds'], output=str(args.output))), flush=True)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
