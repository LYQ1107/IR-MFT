#!/usr/bin/env python3
"""Individual, uninstrumented B1 costs after full source/track-prefix identity.

Reads frozen models and inference manifests, never Test GT or accuracy values.
"""
import argparse
import gc
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from tools.native_pyramid_experiment import gpu_settings, load_teacher, read
from tools.native_pyramid_online_inference import OnlineNativeDetector
from tools.native_pyramid_stream_tracks import IndependentByteTrack, mot_line
from tools.neck72_runtime import require_host_memory, require_storage
from tools.observe_spatial_native import load_modules, read_image, require_test_freeze
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.spatial_native_study import STUDY, context, predecessor_ready, registered


def update_packet(digest, frame, values):
    digest.update(np.asarray([frame, len(values)], dtype='<i8').tobytes())
    digest.update(np.asarray(values, dtype='<f4').tobytes())


def actual_frame(plan, sequence, frame, needs_ir):
    rgb_path = Path(plan['data_root'])/'test'/sequence/'visible'/f'{frame:06d}.jpg'
    rgb, rgb_sha = read_image(rgb_path)
    ir, ir_sha = None, None
    if needs_ir:
        ir_path = Path(plan['generated_ir_root'])/'test'/sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
        ir, ir_sha = read_image(ir_path, scalar=True)
    return rgb, ir, rgb_sha, ir_sha


def check_prefix(engine, case, plan, sequence, reference):
    engine.reset(sequence)
    tracker = IndependentByteTrack()
    packets, tracks = hashlib.sha256(), hashlib.sha256()
    assert len(reference['prefix32_source_rows']) == 32
    for frame in range(1, 33):
        rgb, ir, rgb_sha, ir_sha = actual_frame(plan, sequence, frame, engine.needs_actual_IR_image)
        source = reference['prefix32_source_rows'][frame-1]
        assert source[:3] == [sequence, frame, rgb_sha]
        if engine.needs_actual_IR_image:
            assert source[3] == ir_sha
        observations, _ = engine.step(rgb, ir, sequence, frame)
        values = observations[case['id']]
        update_packet(packets, frame, values)
        for row in tracker.update(values, frame):
            tracks.update(mot_line(row).encode())
    assert packets.hexdigest() == reference['prefix32_packet_sha256'][case['id']]
    assert tracks.hexdigest() == reference['prefix32_literal_MOT_sha256'][case['id']]
    return dict(sequence=sequence, frames=32, exact_input_image_bytes=True,
        exact_detection_packet_prefix=True, exact_literal_MOT_prefix=True,
        packet_sha256=packets.hexdigest(), literal_MOT_sha256=tracks.hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', required=True)
    args = parser.parse_args()
    plan, study, protocol, calibration, work, checkpoint, base = context()
    registration = registered(work)
    assert predecessor_ready(work)
    freeze = require_test_freeze(base, args.specification)
    specification = read(args.specification)
    assert specification['split'] == 'test' and specification['models'] == freeze['models']
    require_host_memory(8*2**30, maximum_workers=1)
    require_storage(work)
    physical, uuid = gpu_settings()
    teacher = load_teacher(checkpoint, training=False)
    output = base/'individual_runtime.json'
    assert not output.exists()
    rows = []
    for case in specification['models']:
        modules, information = load_modules([case], study, base, registration, 'cuda')
        engine = OnlineNativeDetector(teacher, [case], modules, study['inference_input_hw'], calibration)
        equivalence, references = [], {}
        for sequence in protocol['historical_test']:
            source = Path(specification['stream_directory'])/'sequence_manifests'/(sequence+'.json')
            reference = read(source)
            assert reference['specification_sha256'] == sha256(args.specification)
            assert reference['status'] == 'COMPLETE_SPATIAL_B1_SOURCE_MATCHED_OBSERVATIONS_WITHOUT_GT'
            references[sequence] = reference
            check = check_prefix(engine, case, plan, sequence, reference)
            equivalence.append(dict(check, complete_source_manifest_sha256=sha256(source)))
        durations = []
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for sequence in protocol['historical_test']:
            engine.reset(sequence)
            tracker = IndependentByteTrack()
            measured_packets, measured_tracks = hashlib.sha256(), hashlib.sha256()
            for frame in range(1, 33):
                torch.cuda.synchronize()
                started = time.perf_counter()
                rgb, ir, rgb_sha, ir_sha = actual_frame(plan, sequence, frame, engine.needs_actual_IR_image)
                values, _ = engine.step(rgb, ir, sequence, frame)
                track_rows = tracker.update(values[case['id']], frame)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                if frame > 8:
                    durations.append(elapsed)
                source = references[sequence]['prefix32_source_rows'][frame-1]
                assert source[:3] == [sequence, frame, rgb_sha]
                if engine.needs_actual_IR_image:
                    assert source[3] == ir_sha
                update_packet(measured_packets, frame, values[case['id']])
                for track_row in track_rows:
                    measured_tracks.update(mot_line(track_row).encode())
            assert measured_packets.hexdigest() == references[sequence]['prefix32_packet_sha256'][case['id']]
            assert measured_tracks.hexdigest() == references[sequence]['prefix32_literal_MOT_sha256'][case['id']]
        assert len(durations) == 168 and all(t > 0 for t in durations)
        module = modules[case['id']]
        rows.append(dict(id=case['id'], model=case, source_equivalence=equivalence,
            source_equivalence_before_timing=True, actual_single_condition_at_a_time=True,
            actual_timed_run_input_packets_and_literal_MOT_also_exact=True,
            original_RGB_parameters=sum(p.numel() for p in teacher.parameters()),
            auxiliary_parameters=sum(p.numel() for p in module.parameters()) if module is not None else 0,
            total_input_frames=224, warmup_frames_each_sequence=8, measured_frames=168,
            seconds=sum(durations), FPS=168/sum(durations),
            mean_latency_ms=1000*sum(durations)/168, median_latency_ms=1000*sorted(durations)[84],
            peak_allocated_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_GPU_GiB=torch.cuda.max_memory_reserved()/2**30,
            physical_gpu=physical, GPU_uuid=uuid, independent_input_batch=1,
            includes_IO_decode_byte_hash_preprocessing_RGB_encoder_teacher_native_fusion_head_NMS_ByteTrack=True,
            excludes_weight_loading_and_offline_DiffV2IR_generation=True, instrumented_gate_observer=False,
            Test_GT_or_accuracy_input=False, measurement_repeats=1))
        print(dict(id=case['id'], measured_frames=168, FPS=rows[-1]['FPS']), flush=True)
        del engine, modules, information, module
        gc.collect()
        torch.cuda.empty_cache()
    registered(work)
    dump(output, dict(status='COMPLETE_SPATIAL_INDIVIDUAL_SOURCE_MATCHED_PREFIX_RUNTIME', rows=rows,
        study_sha256=sha256(STUDY), specification_sha256=sha256(args.specification),
        Test_selection_sha256=sha256(base/'Test_selection_frozen.json'),
        sample_scope='Seven32-frame prefixes,8warmup24timed pervideo;168timedframes percondition, not full15985-frameFPS.',
        source_equivalence_checked_before_any_timing_of_each_condition=True,
        offline_DiffV2IR_generation_and_weight_loading_excluded=True,
        GT_read=False, Test_accuracy_read=False, image_or_FPN_cache_created=False,
        fixed_seed=0, additional_seed_work=False, goal_met=False, supported_innovations=0))


if __name__ == '__main__':
    main()
