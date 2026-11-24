#!/usr/bin/env python3
"""Individual real-image online cost after frozen Dev choice, without Test GT."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from tools.evaluate_native_pyramid import load_modules, read_image, require_test_freeze
from tools.native_pyramid_experiment import STUDY, context, gpu_settings, load_teacher, predecessor_ready, read, registry
from tools.native_pyramid_online_inference import OnlineNativeDetector
from tools.native_pyramid_stream_tracks import IndependentByteTrack
from tools.native_pyramid_training_loop import guard_storage
from tools.run_pseudo_ir_comparison import dump, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', required=True)
    args = parser.parse_args()
    plan, study, protocol, calibration, work, checkpoint = context()
    registry(work)
    assert predecessor_ready(work)
    require_test_freeze(work, args.specification)
    specification = read(args.specification)
    assert specification['split'] == 'test'
    destination = work / 'native_pyramid_individual_runtime.json'
    assert not destination.exists()
    guard_storage(work)
    physical, uuid = gpu_settings()
    model = load_teacher(checkpoint, training=False)
    total_model = sum(p.numel() for p in model.parameters())
    rows = []
    for case in specification['models']:
        modules, information = load_modules([case], study)
        engine = OnlineNativeDetector(model, [case], modules, study['inference_input_hw'], calibration)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        durations, counts, packet_bytes, source_ids = [], [], [], []
        for sequence in protocol['historical_test']:
            tracker = IndependentByteTrack()
            for frame in range(1, 33):
                torch.cuda.synchronize()
                started = time.perf_counter()
                rgb_path = Path(plan['data_root']) / 'test' / sequence / 'visible' / f'{frame:06d}.jpg'
                rgb, rgb_sha = read_image(rgb_path)
                ir, ir_sha = None, None
                if engine.needs_actual_IR_image:
                    ir_path = Path(plan['generated_ir_root']) / 'test' / sequence / 'infrared_diffv2ir' / f'{frame:06d}.png'
                    ir, ir_sha = read_image(ir_path, scalar=True)
                values, _ = engine.step(rgb, ir, sequence, frame)
                tracker.update(values[case['id']], frame)
                torch.cuda.synchronize()
                seconds = time.perf_counter()-started
                if frame > 8:
                    durations.append(seconds)
                    counts.append(len(values[case['id']]))
                source_ids.append([sequence, frame, rgb_sha, ir_sha])
        import hashlib
        rows.append(dict(id=case['id'], source_images_sha256=hashlib.sha256(json.dumps(source_ids).encode()).hexdigest(),
            original_RGB_parameters=total_model, auxiliary_parameters=sum(p.numel() for p in modules[case['id']].parameters())
                if modules[case['id']] is not None else 0,
            warmup_frames_each_sequence=8, measured_frames=168, total_input_frames=224,
            sequences=protocol['historical_test'], seconds=sum(durations), FPS=len(durations)/sum(durations),
            mean_latency_ms=1000*sum(durations)/len(durations),
            median_latency_ms=1000*sorted(durations)[len(durations)//2],
            peak_allocated_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_GPU_GiB=torch.cuda.max_memory_reserved()/2**30,
            mean_output_detections=sum(counts)/len(counts),
            physical_gpu=physical, GPU_uuid=uuid, independent_input_batch=1,
            includes_image_IO_preprocessing_RGB_encoder_teacher_native_encoder_fusion_flow_NMS_and_ByteTrack=True,
            includes_offline_DiffV2IR_generation=False, measurement_repeats=1))
        print(json.dumps(dict(id=case['id'], measured_frames=168, FPS=rows[-1]['FPS'])), flush=True)
        del engine, modules, information
        gc.collect()
        torch.cuda.empty_cache()
    registry(work)
    dump(destination, dict(status='COMPLETE_NATIVE_INDIVIDUAL_RUNTIME', rows=rows,
        specification_sha256=sha256(args.specification), study_sha256=sha256(STUDY),
        source_RGB_sha256=study['source_RGB_sha256'], physical_gpu=physical, GPU_uuid=uuid,
        sample_scope='First32frames from each of7Testvideos; first8warmup,24measured. Stratified prefix cost sample, not full-Test per-method FPS.',
        scope='Single condition at a time; shared production FPS is reported separately. Other users may share the GPU; no seed repetitions or performance variance claim.',
        GT_read=False, Test_accuracy_read=False, image_or_FPN_cache_created=False,
        goal_met=False, supported_innovations=0))


if __name__ == '__main__': main()
