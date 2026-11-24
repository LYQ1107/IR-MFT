#!/usr/bin/env python3
"""Time one frozen neck72 condition from fresh current images, without GT."""
import argparse
import gc
import hashlib
from pathlib import Path
import sys
import time
import resource

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
import torchvision

from tools.eval_mft26ir import write_sequence_results
from tools.native_pyramid_training_loop import state_sha256
from tools.neck72_online_inputs import (FrozenFrontend, CurrentFeatures, CurrentPointStream,
                                       PrefixPackets, needs_ir)
from tools.neck72_runtime import require_storage, read
from tools.neck72_runtime_study import (SPEC, registered, cases, gpu_settings, frame_comparison, comparison_summary)
from tools.run_neck72_replay import load_sequence, tracker_for
from tools.run_pseudo_ir_comparison import dump, sha256


def input_image(plan, split, sequence, frame, ir_required):
    rgb_path = Path(plan['data_root']) / ('train' if split == 'dev' else 'test') / sequence / 'visible' / f'{frame:06d}.jpg'
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    ir_path, ir = None, None
    if ir_required:
        ir_path = Path(plan['generated_ir_root']) / split / sequence / 'infrared_diffv2ir' / f'{frame:06d}.png'
        ir = cv2.imread(str(ir_path), cv2.IMREAD_UNCHANGED)
        if ir is None or ir.ndim != 2 or ir.dtype != np.uint8 or ir.shape != rgb.shape[:2]:
            raise ValueError('Expected the fixed current uint8 scalar DiffV2IR image: '+str(ir_path))
    return rgb, ir, rgb_path, ir_path


def model_record(models):
    return {key: dict(parameters=sum(p.numel() for p in model.parameters()),
        state_sha256=state_sha256(model.state_dict())) for key, model in models.items()}


def source_track_prefix(work, split, case, sequence, frames, output):
    original = work / ('neck72_'+split) / case / 'track_results' / (sequence+'.txt')
    # Exactly the preserved formatter bytes; do not rebase IDs or regenerate
    # expected rows through another tracker, and never open a GT file.
    lines = [line for line in original.read_bytes().splitlines(keepends=True)
             if int(line.split(b',', 1)[0]) <= frames]
    output.write_bytes(b''.join(lines))
    return dict(full_source=str(original), full_source_sha256=sha256(original),
                prefix=str(output), prefix_sha256=sha256(output))


@torch.inference_mode()
def measure(case, split, plan, protocol, work, specification, output):
    config = case['config']; expected_frames = specification['sampling']['prefix_frames_per_sequence']
    warmup = specification['sampling']['warmup_frames_per_sequence']
    sequences = protocol['dev' if split == 'dev' else 'historical_test']
    frontend = FrozenFrontend(plan, config['detection_arm'], split, 'cuda')
    features = None if config['backend'] == 'byte' else CurrentFeatures(config, work, protocol)
    models = {'detector': frontend.model}
    if features is not None:
        models.update(features.models())
    initial_models = model_record(models)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    records = []; all_latencies = []
    for sequence in sequences:
        reference, reference_sources = load_sequence(plan, work, split, sequence, config)
        tracker = tracker_for(config)
        if config['backend'] != 'byte':
            initial_heads = model_record(dict(RGB_pair=tracker.model, native_query=tracker.native_model))
        else:
            initial_heads = {}
        buffers, point_stream = PrefixPackets(expected_frames), CurrentPointStream()
        rows, comparisons, durations, source_rows = [], [], [], []
        for frame in range(1, expected_frames+1):
            torch.cuda.synchronize(); started = time.perf_counter()
            rgb, ir, rgb_path, ir_path = input_image(plan, split, sequence, frame, needs_ir(config))
            ready = time.perf_counter()
            detections = frontend.predict(rgb, ir)
            torch.cuda.synchronize(); detected = time.perf_counter()
            values = {} if features is None else features.predict(rgb, ir, detections[:, :4])
            torch.cuda.synchronize(); encoded = time.perf_counter()
            points = {} if features is None else point_stream.step(frame, rgb, detections[:, :4])
            current = buffers.append(frame, detections, values, points)
            prepared = time.perf_counter()
            if features is None:
                rows.extend((frame, t.tlwh.copy(), t.track_id, t.score)
                    for t in tracker.update(detections, (1, 1), (1, 1)) if t.tlwh[2]*t.tlwh[3] > 100)
            else:
                rows.extend(tracker.update(frame, current))
            torch.cuda.synchronize(); finished = time.perf_counter()
            if frame > warmup:
                durations.append(dict(frame=frame, total_seconds=finished-started,
                    file_read_decode_seconds=ready-started, detector_preprocess_NN_NMS_transfer_seconds=detected-ready,
                    descriptor_crop_NN_alignment_seconds=encoded-detected,
                    RGB_point_LK_vote_packet_seconds=prepared-encoded, tracker_seconds=finished-prepared))
            # Verification and hashing are deliberately outside the timers.
            comparisons.append(frame_comparison(frame, current, reference, config, specification['numerics']['source_gates']))
            source_rows.append([sequence, frame, sha256(rgb_path), sha256(ir_path) if ir_path else None])
        actual = output / (sequence+'_fresh.txt'); write_sequence_results(str(actual), rows)
        original = output / (sequence+'_reference.txt')
        tracks = source_track_prefix(work, split, case['id'], sequence, expected_frames, original)
        exact = actual.read_bytes() == original.read_bytes()
        if features is not None:
            assert initial_heads == model_record(dict(RGB_pair=tracker.model, native_query=tracker.native_model))
        all_latencies.extend(v['total_seconds'] for v in durations)
        records.append(dict(sequence=sequence, input_frames=expected_frames, warmup_frames=warmup,
            measured_frames=len(durations), source_provenance=reference_sources,
            source_images_sha256=hashlib.sha256(str(source_rows).encode()).hexdigest(),
            reference_tracks=tracks, actual_track_sha256=sha256(actual), prefix_tracks_bit_exact=exact,
            source_frames_within_registered_limits=all(c['passed'] for c in comparisons),
            source_comparisons=comparison_summary(comparisons), measured_frame_components=durations,
            source_equivalence_passed=exact and all(c['passed'] for c in comparisons),
            head_parameters={key: value['parameters'] for key, value in initial_heads.items()},
            original_RGB_and_query_weights_BN_unchanged=True))
        print(case['id'], split, sequence, 'fresh_source_equivalence', records[-1]['source_equivalence_passed'], flush=True)
        del reference, tracker, buffers, point_stream
        gc.collect()
    assert initial_models == model_record(models)
    measured = len(sequences)*(expected_frames-warmup)
    assert measured == len(all_latencies) and sum(all_latencies) > 0
    equivalent = all(row['source_equivalence_passed'] for row in records)
    return dict(status='COMPLETE_NECK72_FRESH_IMAGE_RUNTIME_MEASUREMENT', id=case['id'], split=split,
        config=config, specification_sha256=sha256(SPEC), source_equivalence_passed=equivalent,
        valid_full_component_cost_for_preserved_accuracy=equivalent,
        diagnostic_only_until_source_equivalence=not equivalent, rows=records,
        measured_frames=measured, seconds=sum(all_latencies), measured_FPS=measured/sum(all_latencies),
        mean_latency_ms=1000*float(np.mean(all_latencies)), median_latency_ms=1000*float(np.median(all_latencies)),
        p90_latency_ms=1000*float(np.percentile(all_latencies, 90)), p95_latency_ms=1000*float(np.percentile(all_latencies, 95)),
        peak_own_allocated_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,
        peak_own_reserved_GPU_GiB=torch.cuda.max_memory_reserved()/2**30,
        peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        loaded_pixel_models=initial_models, original_pixel_weights_BN_unchanged=True,
        frontend_provenance=frontend.provenance, native_feature_provenance=features.provenance if features else None,
        actual_IR_image_read=needs_ir(config), input_batch=1, repeats=1, fixed_seed=0,
        cached_bank_substitution_for_inference=False, future_frame_input=False,
        GT_read=False, Test_accuracy_used=False, observer_hooks_enabled=False,
        includes_file_IO_decode_all_consumed_pixel_models_classical_points_and_tracking=True,
        includes_offline_DiffV2IR_generation=False, full_Test15985frame_FPS=False,
        host_GPU_and_filesystem_cache_shared_uncontrolled=True,
        scope=specification['scope'], timer_scope=specification['timers'],
        goal_met=False, supported_innovations=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    parser.add_argument('--split', choices=['dev', 'test'], required=True)
    args = parser.parse_args()
    plan, protocol, work, specification, registration = registered()
    physical, uuid = gpu_settings(work)
    frozen = cases(work); matches = [c for c in frozen['cases'] if c['id'] == args.case]
    assert len(matches) == 1
    if args.split == 'test':
        engineering = read(work / 'neck72_fresh_runtime_Dev_engineering.json')
        assert engineering['all_frozen_cases_passed']
        assert engineering['specification_sha256'] == sha256(SPEC)
        assert engineering['selection_sha256'] == sha256(work / 'neck72_Test_selection_frozen.json')
    output = work / 'neck72_fresh_runtime' / args.split / args.case
    assert not output.exists(), 'Preserve existing fresh-image runtime evidence'
    require_storage(work, specification['memory']['maximum_case_growth_bytes'])
    output.mkdir(parents=True)
    result = measure(matches[0], args.split, plan, protocol, work, specification, output)
    result.update(physical_gpu=physical, GPU_uuid=uuid, torch=str(torch.__version__),
        torchvision=str(torchvision.__version__), numpy=str(np.__version__), opencv=str(cv2.__version__),
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        source_sha256=sha256(__file__), runtime_registration_sha256=sha256(work / 'neck72_fresh_runtime_registered.json'),
        selection_sha256=sha256(work / 'neck72_Test_selection_frozen.json'))
    registered(); require_storage(work)
    dump(output / 'complete.json', result)
    print(result['status'], args.case, args.split, 'source_equivalence', result['source_equivalence_passed'], flush=True)


if __name__ == '__main__':
    main()
