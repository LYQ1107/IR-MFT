#!/usr/bin/env python3
"""Single-condition B4 image/association prefix throughput, without gate hooks."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import torch

from tools.correlation_recovery_inference import CorrelationCaseDetector, load_adapters
from tools.correlation_recovery_workflow import SPEC, WORKFLOW, context, frame_packet, gpu_settings, packet, predecessor_ready, registered, require_probes
from tools.native_pyramid_stream_tracks import IndependentByteTrack
from tools.native_pyramid_training_loop import guard_storage
from tools.run_correlation_recovery_gpu import load_teacher, read_batch, require_test_freeze
from tools.run_pseudo_ir_comparison import dump, sha256


@torch.inference_mode()
def main():
    plan, spec, protocol, parent, calibration, work = context()
    workflow = registered()
    assert predecessor_ready(work)
    require_probes(work)
    require_test_freeze(work)
    physical, uuid = gpu_settings()
    cv2.setNumThreads(1)
    guard_storage(work)
    output = work / 'correlation_recovery_individual_runtime.json'
    assert not output.exists()
    model = load_teacher(parent)
    rows = []
    for case in spec['cases']:
        modules, _ = load_adapters([case], spec['repair_plan_sha256'], parent['source_RGB_sha256'], 'cuda')
        engine = CorrelationCaseDetector(model, [case], modules, parent['inference_input_hw'], calibration)
        measured, seconds, per_sequence = 0, 0., {}
        torch.cuda.reset_peak_memory_stats()
        for sequence in protocol['historical_test']:
            canonical_path = Path(plan['frozen_detection_sources']['rgb_test']) / 'detections' / (sequence+'.npz')
            canonical = packet(canonical_path)
            tracker = IndependentByteTrack()
            sequence_seconds = 0.
            for first in range(1, 33, 4):
                frames = list(range(first, first+4))
                torch.cuda.synchronize()
                started = time.perf_counter()
                rgb, ir = read_batch(plan, 'test', sequence, frames, engine.needs_IR)
                old = [frame_packet(canonical, frame) for frame in frames]
                result, _ = engine.batch(rgb, ir, old)
                for index, frame in enumerate(frames):
                    tracker.update(result[case['id']][index], frame)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                if first > 8:
                    sequence_seconds += elapsed
                    seconds += elapsed
                    measured += 4
            per_sequence[sequence] = dict(warmup_frames=8, measured_frames=24, measured_seconds=sequence_seconds)
        assert measured == 168 and seconds > 0
        rows.append(dict(id=case['id'], measured_frames=measured, seconds=seconds, FPS=measured/seconds,
            peak_allocated_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,
            source_parameters=sum(p.numel() for p in model.parameters()),
            auxiliary_parameters=sum(p.numel() for m in modules.values() if m is not None for p in m.parameters()),
            per_sequence=per_sequence, baseline_convention=case.get('convention')))
        del engine, modules
        guard_storage(work)
        print(json.dumps(dict(case=case['id'], measured_frames=measured)), flush=True)
    dump(output, dict(status='COMPLETE_CGR_SINGLE_CONDITION_B4_PREFIX_RUNTIME', rows=rows,
        workflow_sha256=sha256(WORKFLOW), source_specification_sha256=sha256(SPEC),
        physical_gpu=physical, GPU_uuid=uuid, fixed_seed=0, additional_seed_work=False,
        GT_read=False, Test_accuracy_read=False, observer_enabled=False,
        scope='Each frozen condition independently:7videos*32frames,8warmup/24measured each,168measured. '
            'Includes source file reads/decode/preprocessing, currentRGB encoder/teacher head, needed cue encoder, '
            'CGR fusion, NMS, historical RGB proposal lookup/priority and ByteTrack. '
            'Frozen canonical proposal NPZ is loaded outside timed batches; its original offline generation '
            'and DiffV2IR generation are excluded. Even canonical-convention control computes freshRGB then '
            'uses historical proposals. B4 throughput, not B1 online latency or fullTest per-method FPS. '
            'GPU and storage may be shared with other users; no repeat-seed variance claim.',
        goal_met=False, supported_innovations=0))


if __name__ == '__main__':
    main()
