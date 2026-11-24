#!/usr/bin/env python3
"""Synthetic-only P3-to-memory bridge proof with checkpoint/image-read traps."""
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch

from tools.fusion_memory_capture import FusionFeatureCapture
from tools.native_pyramid_train_data import detach_state
from tools.native_pyramid_training_loop import state_sha256
from tools.neck72_runtime import require_host_memory, require_storage
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.spatial_native_fusion import SpatialNativeFusion
from yolox.models.native_pyramid_fusion import NativePyramidFusion
from yolox.tracker.fusion_memory_tracker import FusionMemoryConfig, FusionMemoryTracker

SPEC = ROOT/'configs/perceptual_fusion_memory_capture_prototype_20261125.json'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    started = time.monotonic()
    specification = json.loads(SPEC.read_text())
    plan_path = ROOT/'configs/perceptual_mft_20261122.json'
    assert sha256(plan_path) == specification['main_plan_sha256']
    plan = json.loads(plan_path.read_text())
    work = Path(plan['work_root'])
    output = work/'fusion_memory_capture_CPU_proof.json'
    assert not output.exists()
    require_host_memory(2*2**30, maximum_workers=1)
    require_storage(work)
    log = work/'fusion_memory_capture_pytest.log'
    match = re.search(r'\b(\d+) passed\b', log.read_text())
    assert match and int(match.group(1)) == 15
    paths = ['tools/fusion_memory_capture.py', 'tools/probe_fusion_memory_capture.py',
        'tests/test_fusion_memory_capture.py', str(SPEC.relative_to(ROOT)),
        'tools/fusion_memory_features.py', 'yolox/tracker/fusion_memory_tracker.py',
        'yolox/tracker/byte_tracker.py', 'yolox/tracker/basetrack.py', 'yolox/tracker/matching.py',
        'yolox/tracker/kalman_filter.py', 'yolox/models/native_pyramid_fusion.py',
        'tools/spatial_native_fusion.py', 'tools/native_pyramid_train_data.py',
        'tools/native_pyramid_training_loop.py']
    sources = {name: sha256(ROOT/name) for name in paths}
    assert all(sources[name] == value for name, value in specification['frozen_dependencies'].items())
    registration_path = work/'native_spatial/pipeline_registered.json'
    registration = json.loads(registration_path.read_text())
    assert all(sha256(ROOT/name) == value for name, value in registration['source_hashes'].items())
    attempted = dict(checkpoint_loads=0, image_reads=0, image_decodes=0)
    def forbid(key):
        def function(*args, **kwargs):
            attempted[key] += 1
            raise AssertionError('No learned checkpoint or real image is permitted in this engineering proof')
        return function
    records = []
    with patch.object(torch, 'load', forbid('checkpoint_loads')), patch.object(cv2, 'imread', forbid('image_reads')), patch.object(cv2, 'imdecode', forbid('image_decodes')):
        for kind in ['quality', 'temporal', 'spatial']:
            torch.manual_seed(0)
            kwargs = dict(channels=(8, 16, 32))
            module = (SpatialNativeFusion(**kwargs, use_redundancy=True) if kind == 'spatial' else
                NativePyramidFusion(**kwargs, mode=kind, use_motion=False)).float().eval().requires_grad_(False)
            with torch.no_grad():
                module.output[0].bias.fill_(.125)  # Explicit synthetic constant, not a fitted parameter.
            weights = state_sha256(module.state_dict())
            capture = FusionFeatureCapture(module, feature_source='fused', quality_policy='attention_proxy')
            tracker = FusionMemoryTracker(FusionMemoryConfig(feature_source='fused', quality_mode='provided'))
            rgb = tuple(torch.randn(1, c, 64//s, 96//s) for c, s in zip((8, 16, 32), (8, 16, 32)))
            scores = tuple(torch.full_like(value[:, :1], .25) for value in rgb)
            cue = torch.randn(1, 3, 64, 96)
            boxes = np.array([[4., 2., 20., 10., .9], [60., 40., 65., 45., .8]], np.float32)
            input_boxes = boxes.copy()
            state, packet_trace = None, hashlib.sha256()
            try:
                for frame in range(1, 9):
                    capture.begin_frame('synthetic', frame, (32, 48), (64, 96), 2.)
                    with torch.no_grad():
                        _, state, _ = module(rgb, cue+frame*.01, scores, state=detach_state(state),
                            sequence='synthetic', frame=frame)
                    packet = capture.take_packet('synthetic', frame, boxes)
                    assert packet['rgb'].shape == packet['fused'].shape == (2, 8)
                    assert packet['auxiliary'].shape == (2, 64)
                    assert packet['auxiliary_available'].tolist() == [True, False]
                    assert packet['metadata']['quality_is_calibrated_reliability'] is False
                    assert capture.current is None and capture.rgb is None and capture.auxiliary is None
                    for name in ['rgb', 'auxiliary', 'fused']:
                        values = packet[name]
                        assert np.isfinite(values).all()
                        assert abs(float(np.linalg.norm(values[0]))-1.) < 1e-6 and not values[1].any()
                        packet_trace.update(values.tobytes())
                    tracks = tracker.update('synthetic', frame, boxes, packet)
                    assert tracks and all(np.isfinite(t.mean).all() for t in tracks)
                    np.testing.assert_array_equal(boxes, input_boxes)
                assert weights == state_sha256(module.state_dict())
                assert all(p.grad is None for p in module.parameters())
                expected_aux = 'current_filtered_native_state' if kind == 'temporal' else 'current_native_encoder'
                assert packet['metadata']['auxiliary_source'] == expected_aux
                records.append(dict(kind=kind, synthetic_frames=8, source_geometry=[32, 48], input_geometry=[64, 96],
                    RGB_channels=8, auxiliary_channels=64, fused_channels=8,
                    actual_returned_auxiliary_source=expected_aux, packet_trace_sha256=packet_trace.hexdigest(),
                    invalid_ROI_detection_retained=True, current_feature_references_released_each_frame=True,
                    module_parameters_and_gradients_unchanged=True, synthetic_enabled_memory_exercised=True))
            finally:
                capture.close()
    assert not any(attempted.values()) and not torch.cuda.is_initialized()
    assert all(sha256(ROOT/name) == value for name, value in sources.items())
    assert all(sha256(ROOT/name) == value for name, value in registration['source_hashes'].items())
    require_storage(work)
    result = dict(status='COMPLETE_SYNTHETIC_F4_ACTUAL_P3_CAPTURE_AND_MEMORY_BRIDGE_CPU_ONLY',
        source_hashes=sources, specification_sha256=sha256(SPEC), tests_passed=int(match.group(1)),
        test_log_sha256=sha256(log), records=records, guarded_attempt_counters=attempted,
        physical_GPU_work=False, learned_checkpoint_loads=0, actual_Train_Dev_Test_image_reads=0,
        actual_GT_read=False, learned_fusion_features_transferred=False, new_accuracy=False,
        new_formal_fits=False, formal_F4_launch_ready=False, full_four_direction_validation=False,
        existing323_sources_unchanged=True, parent_spatial_registration_sha256=sha256(registration_path),
        image_or_FPN_disk_cache=False, saved_weights=False, fixed_seed=0, additional_seed_work=False,
        peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
        seconds=time.monotonic()-started,
        scope='Known synthetic inputs and untrained modules with one explicit bias constant;24syntheticframes. '
            'Direct returned/filtered P3 vectors, causal capture and packet-to-memory interface only. '
            'No learned visual parent, new HOTA, real modality efficacy, GPU parity or final innovation.',
        goal_met=False, supported_innovations=0)
    assert result['peak_process_RSS_GiB'] <= 2
    dump(output, result)
    print(json.dumps(dict(status=result['status'], synthetic_frames=24, seconds=result['seconds'])), flush=True)


if __name__ == '__main__':
    main()
