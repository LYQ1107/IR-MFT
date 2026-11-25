#!/usr/bin/env python3
"""Actual single-model B4 costs for three already frozen full-Train controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import resource

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch

from tools.benchmark_structural_factorial import inputs
from tools.benchmark_structural_full_train import predict
from tools.extract_structural_full_train import context as fit_context, load_bundle
from tools.mft_cost_supplement import CASES, SPEC, Images, image_path, registered, references, route
from tools.native_pyramid_experiment import predecessor_ready, read
from tools.neck72_runtime import require_storage, require_host_memory
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.task_utility_queue import process_identity
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.byte_tracker import BYTETracker


def tracker():
    BaseTrack._count = 0
    return BYTETracker(SimpleNamespace(track_thresh=.6, track_buffer=30, match_thresh=.9, mot20=False), frame_rate=25)


def track_rows(engine, values):
    return [(t.track_id, *map(float, t.tlwh), float(t.score))
        for t in engine.update(values, (1, 1), (1, 1)) if t.tlwh[2] * t.tlwh[3] > 100]


def fingerprint(rows):
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def measure(model, adapter, plan, config, calibration, case, images, refs):
    records = []
    for sequence, path in refs.items():
        with np.load(path) as z:
            offsets, boxes, scores = z['offsets'], z['boxes'], z['scores']
            np.testing.assert_array_equal(z['frame_ids'][:32], np.arange(1, 33))
            reference = [np.c_[boxes[offsets[f-1]:offsets[f]], scores[offsets[f-1]:offsets[f]]].astype(np.float32)
                for f in range(1, 33)]
        reference_tracker = tracker()
        expected_rows = [track_rows(reference_tracker, values) for values in reference]
        live_tracker = tracker()
        delivered_rows = []
        total = preparation = compute = tracking = 0.
        cue = 'baseline' if case.get('baseline') else 'ir'
        for first in range(1, 33, 4):
            frames = list(range(first, first + 4))
            torch.cuda.synchronize()
            start = time.perf_counter()
            arrays, ratios, available = inputs(plan, config, calibration, sequence, frames, cue, images)
            if case.get('intervention') == 'unavailable':
                available = [False] * 4
            torch.cuda.synchronize()
            ready = time.perf_counter()
            values = predict(model, adapter, arrays, ratios, available, cue, 'raw')
            torch.cuda.synchronize()
            predicted = time.perf_counter()
            delivered_rows.extend(track_rows(live_tracker, value) for value in values)
            torch.cuda.synchronize()
            finished = time.perf_counter()
            if first > 8:
                total += finished - start
                preparation += ready - start
                compute += predicted - ready
                tracking += finished - predicted
            for frame, value in zip(frames, values):
                np.testing.assert_array_equal(value, reference[frame-1])
            del arrays, values
        if delivered_rows != expected_rows:
            raise ValueError('Independent actual NN-prefix and source-prefix MOT differ')
        records.append(dict(sequence=sequence, timed_frames=24, warmup_frames=8, NN_batch_size=4,
            seconds=total, input_preparation_transfer_seconds=preparation,
            NN_NMS_policy_transfer_seconds=compute, tracking_seconds=tracking,
            source32_arrays_exact=True, independent_prefix_MOT_exact=True,
            source_packet_sha256=sha256(path), MOT_prefix_sha256=fingerprint(delivered_rows)))
    return dict(per_sequence=records, timed_frames=48, NN_batch_size=4, timed_batches=12,
        FPS=48/sum(r['seconds'] for r in records),
        NN_NMS_policy_transfer_FPS=48/sum(r['NN_NMS_policy_transfer_seconds'] for r in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--memory-inputs', action='store_true')
    args = parser.parse_args()
    plan, work, base, registration = registered()
    state = read(base / 'state.json')
    identity = registration['native_waiter_identity']
    if process_identity(identity['pid']) != identity or not predecessor_ready(work):
        raise ValueError('Actual completed/exited legacy predecessor and original deferred waiter required')
    proc = Path('/proc') / str(identity['pid'])
    if (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0] not in ['T', 't']:
        raise ValueError('The original waiting native controller must remain deferred during these costs')
    if not state.get('native_waiter_deferred') or process_identity(state['controller_pid']) != state['controller_identity']:
        raise ValueError('The actual owning CPU cost scheduler must be alive')
    physical = os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical != '4':
        raise ValueError('Use physical4serially to match the original cost experiment')
    require_storage(work, growth_bytes=2**20)
    require_host_memory(8 * 2**30, maximum_workers=1)
    configpath, case, recipe = route(args.case)
    oldplan, study, studypath, config, oldconfigpath, _ = fit_context()
    if oldplan != plan:
        raise ValueError('Original fixed data plan changed')
    refs = references(plan, case)
    needs_ir = not bool(case.get('baseline'))
    image_hashes = {str(image_path(plan, s, f, m)): sha256(image_path(plan, s, f, m))
        for s in refs for f in range(1, 33) for m in (['rgb', 'ir'] if needs_ir else ['rgb'])}
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    expected = read(ROOT / 'configs/perceptual_gpu_handoff_20261123.json')['future_gpu_uuids'][physical]
    actual = str(torch.cuda.get_device_properties(0).uuid)
    if actual.lower().replace('gpu-', '') != expected.lower().replace('gpu-', ''):
        raise ValueError('Physical GPU UUID differs')
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(False)  # Original B4cost policy, no stronger retrospective claim.
    torch.cuda.set_per_process_memory_fraction(.45, 0)
    model, loaded, information, _ = load_bundle(dict(models=[recipe]), plan, study, studypath, config, oldconfigpath)
    images = Images(plan, case.get('intervention'), args.memory_inputs, needs_ir)
    preload_started = time.perf_counter()
    if args.memory_inputs:
        images.preload()
    preload_seconds = time.perf_counter() - preload_started if args.memory_inputs else 0.
    calibration = read(work / 'correlation_cache/calibration.json')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        result = measure(model, loaded[recipe['id']], plan, config, calibration, case, images, refs)
    if images.accesses != dict(rgb=64, ir=64 if needs_ir else 0):
        raise ValueError('All64original prefix frame deliveries required;canonical cannot read IR')
    if any(sha256(p) != s for p, s in image_hashes.items()):
        raise ValueError('Original prefix image bytes changed across measurement')
    registered()
    mode = 'memory' if args.memory_inputs else 'live'
    output = base / 'results' / (args.case + '_' + mode + '.json')
    if output.exists():
        raise ValueError('Preserve previous actual cost output')
    result.update(status='COMPLETE_ACTUAL_B4_FROZEN_FULL_TRAIN_CONTROL_COST', id=args.case,
        model_id=recipe['id'], config_sha256=sha256(configpath), memory_inputs=args.memory_inputs,
        specification_sha256=sha256(SPEC), registration_sha256=sha256(base / 'registered.json'),
        source_hashes=registration['source_hashes'], image_hashes=image_hashes,
        physical_gpu=physical, GPU_uuid=actual, own_process_identity=process_identity(os.getpid()),
        actual_device=str(next(model.parameters()).device), raw_precision=str(next(model.parameters()).dtype),
        checkpoint_sha256=information[recipe['id']].get('checkpoint_sha256'),
        peak_own_allocated_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,
        peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
        image_deliveries=images.accesses, decoded_RAM_cache_bytes=images.decoded_bytes,
        preload_seconds_excluded=preload_seconds, GT_read=False, Test_accuracy_read=False,
        new_fit=False, new_image_or_FPN_disk_cache=False, methods_measured_simultaneously=1,
        source_prefix_and_independent_MOT_exact=True, goal_met=False, supported_innovations=0,
        scope='Independent actual B4twoDevprefixes,8warmup+24timed/video;48timedframes. '
            'EachNNpacket matches its frozen actual reference;independent prefix MOT also exact. '
            'Includes old input preparation/transfer/NN/NMS/ByteTrack plus compactprefixrowcollection;excludes model loading,IRgeneration,hash/output. '
            'Memory mode excludes preload. OSfilecachewarmedbyoutside-timer source hashes;sharedhostuncontrolled. '
            'Unavailable intervention still computes the old IR path with availabilityFalse;notminimal missing-file fallback. '
            'RGBcanonical is an actual independent B4NN,notpacketreplayorassignedfreshFPS. NotB1latency/fullTestFPS.')
    if result['peak_process_RSS_GiB'] > 8:
        raise RuntimeError('Own cost worker exceeded8GiB')
    dump(output, result)
    print(json.dumps(dict(id=args.case, mode=mode, FPS=result['FPS'], status=result['status'])), flush=True)


if __name__ == '__main__':
    main()
