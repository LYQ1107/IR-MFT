#!/usr/bin/env python3
"""Actual Train RGB CPU online bridge check; no learned fusion or GT/Test."""
import argparse
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
from tools.fusion_memory_design import make_case
from tools.fusion_memory_online import OnlineFusionMemory, visual_routes
from tools.fusion_memory_streams import FusionMemoryStreamWriter
from tools.native_pyramid_online_inference import OnlineNativeDetector
from tools.native_pyramid_stream_tracks import IndependentByteTrack, mot_line
from tools.native_pyramid_training_loop import state_sha256
from tools.neck72_runtime import require_host_memory, require_storage
from tools.probe_fusion_memory import state_digest
from yolox.exp import get_exp
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.fusion_memory_tracker import FusionMemoryConfig, FusionMemoryTracker

SPEC = ROOT/'configs/perceptual_f4_online_bridge_20261125.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def memory_digest(tracker):
    h = hashlib.sha256()
    for group in [tracker.core.tracked_stracks, tracker.core.lost_stracks, tracker.core.removed_stracks]:
        for track in group:
            for key in ['rgb_memory', 'aux_memory', 'fused_memory']:
                value = getattr(track, key)
                h.update(b'NONE' if value is None else value.tobytes())
            h.update(repr(track.memory_last_frame).encode())
            h.update(repr(track.memory_last_reliability).encode())
    return h.digest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pytest-receipt', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    started = time.monotonic()
    spec = read(SPEC)
    plan_path = ROOT/'configs/perceptual_mft_20261122.json'
    assert sha(plan_path) == spec['main_plan_sha256']
    plan = read(plan_path)
    work = Path(plan['work_root'])
    output = work/'fusion_memory_online_CPU_proof.json'
    stream_directory = work/'fusion_memory_online_CPU_tracks'
    assert not output.exists() and not stream_directory.exists()
    require_host_memory(4*2**30, maximum_workers=1)
    require_storage(work)
    assert sha(ROOT/'configs/perceptual_fusion_memory_study_20261125.json') == spec['F4_scientific_design_sha256']
    assert all(sha(ROOT/name) == value for name, value in spec['frozen_dependencies'].items())
    tests = read(args.pytest_receipt)
    assert tests['exit_code'] == 0 and all(sha(ROOT/name) == value for name, value in tests['source_hashes'].items())
    assert sha(tests['log']) == tests['log_sha256']
    match = re.search(r'\b(\d+) passed\b', Path(tests['log']).read_text())
    assert match and int(match.group(1)) == 14
    sources = {name: sha(ROOT/name) for name in ['tools/fusion_memory_online.py', 'tools/fusion_memory_streams.py',
        'tools/probe_f4_online.py', 'tests/test_fusion_memory_online.py', str(SPEC.relative_to(ROOT))]}
    sources.update(spec['frozen_dependencies'])
    calibration_path = work/'correlation_cache/calibration.json'
    parent = read(ROOT/'configs/perceptual_native_pyramid_study_20261124.json')
    assert sha(calibration_path) == parent['calibration_sha256']
    calibration = read(calibration_path)
    assert calibration['Dev_rows'] == calibration['Test_rows'] == 0
    reference = read(spec['source_content_reference'])['source_image_content_manifest']
    assert sha(reference['path']) == reference['sha256']
    content = {(row['sequence'], row['frame'], row['modality']): row for row in read(reference['path'])}
    paths = {frame: Path(plan['data_root'])/'train/BT-003/visible'/f'{frame:06d}.jpg' for frame in range(1, 9)}
    allowed = {p.resolve() for p in paths.values()}
    attempts = dict(forbidden_GT_Dev_Test_or_IR_reads=0, non_RGB_checkpoint_loads=0)
    def audit(event, values):
        if event != 'open' or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        if (path.name == 'gt.txt' or path.suffix in ['.npy', '.npz'] or 'test' in path.parts or
                (path.suffix.lower() in ['.jpg', '.jpeg', '.png'] and path not in allowed)):
            attempts['forbidden_GT_Dev_Test_or_IR_reads'] += 1
            raise AssertionError('Only eight fixed Train RGB images may enter this engineering proof')
    sys.addaudithook(audit)
    image_reads = []
    def image(frame):
        path = paths[frame]
        raw = path.read_bytes()
        row = content['BT-003', frame, 'rgb']
        digest = hashlib.sha256(raw).hexdigest()
        assert len(raw) == row['bytes'] and digest == row['sha256']
        result = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        assert result is not None
        image_reads.append(dict(frame=frame, bytes=len(raw), sha256=digest))
        return result
    checkpoint = Path(spec['source_RGB_checkpoint'])
    assert sha(checkpoint) == spec['source_RGB_sha256']
    loader, loads = torch.load, []
    def only_rgb(path, *a, **kw):
        if Path(path).resolve() != checkpoint.resolve():
            attempts['non_RGB_checkpoint_loads'] += 1
            raise AssertionError('No learned fusion checkpoint may load before parent Dev qualification')
        loads.append(str(path))
        return loader(path, *a, **kw)
    with patch.object(torch, 'load', only_rgb):
        teacher = get_exp(str(ROOT/'exps/yolox_mft26ir_visible.py'), None).get_model()
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        teacher.load_state_dict(saved['model'], strict=True)
        del saved
    teacher = teacher.float().eval().requires_grad_(False)
    initial = state_sha256(teacher.state_dict())
    sequence, input_hw = 'BT-003_Train_prefix', (480, 864)
    model = dict(id='F4_engineering_fresh_RGB', baseline='fresh_RGB_B1', cue='rgb', seed=0)
    visual = dict(id=model['id'], family='direct', cue='rgb', baseline=True, model=model)
    cases = [make_case(visual, 'byte', 0.)]+[make_case(visual, 'rgb_uniform', w) for w in [.05, .1]]
    references = {case['id']: FusionMemoryTracker(FusionMemoryConfig(
        appearance_weight=case['appearance_weight'], feature_source='rgb', quality_mode='uniform')) for case in cases}
    original = IndependentByteTrack()
    engine = OnlineNativeDetector(teacher, [model], {model['id']: None}, input_hw, calibration)
    capture = FusionFeatureCapture(None, feature_source='rgb', quality_policy='uniform')
    current, expected, reference_calls = {}, {}, [0]
    def reference_hook(module, args, output):
        reference_calls[0] += 1
        capture.capture_rgb_only(sequence, current['frame'], output[0])
    handle = teacher.backbone.register_forward_hook(reference_hook)
    global_count = BaseTrack._count
    reference_MOT = {case['id']: hashlib.sha256() for case in cases}
    original_MOT = hashlib.sha256()
    reference_stats, updates = {}, {c['id']: 0 for c in cases}
    try:
        for frame in range(1, 9):
            rgb = image(frame)
            ratio = min(input_hw[0]/rgb.shape[0], input_hw[1]/rgb.shape[1])
            current['frame'] = frame
            capture.begin_frame(sequence, frame, tuple(rgb.shape[:2]), input_hw, ratio)
            packets, _ = engine.step(rgb, None, sequence, frame)
            detections = packets[model['id']]
            feature = capture.take_packet(sequence, frame, detections)
            assert feature['rgb'].shape == (len(detections), 320)
            expected[frame] = dict(detections=detections.copy(), RGB_ROI_sha256=hashlib.sha256(feature['rgb'].tobytes()).hexdigest(), states={})
            base_rows = original.update(detections, frame)
            original_MOT.update(''.join(mot_line(r) for r in base_rows).encode())
            for case in cases:
                name, tracker = case['id'], references[case['id']]
                tracks = tracker.update(sequence, frame, detections, feature if case['appearance_weight'] else None)
                rows = [(frame, t.tlwh.copy(), t.track_id, t.score) for t in tracks if t.tlwh[2]*t.tlwh[3] > 100]
                reference_MOT[name].update(''.join(mot_line(r) for r in rows).encode())
                expected[frame]['states'][name] = (state_digest(tracker.core), memory_digest(tracker))
                updates[name] += sum(t.memory_last_frame['rgb'] == frame for t in tracker.core.tracked_stracks+tracker.core.lost_stracks)
                if not case['appearance_weight']:
                    assert state_digest(tracker.core) == state_digest(original.tracker)
            print(json.dumps(dict(reference_frame=frame, detections=len(detections))), flush=True)
        for name, tracker in references.items():
            reference_stats[name] = dict(tracker.statistics)
    finally:
        handle.remove()
        capture.close()
    assert reference_calls == [8]
    assert reference_MOT[cases[0]['id']].hexdigest() == original_MOT.hexdigest()
    routes, _ = visual_routes(cases)
    online = OnlineFusionMemory(teacher, cases, {key: None for key in routes}, input_hw, calibration)
    writer = FusionMemoryStreamWriter(stream_directory, sequence, [c['id'] for c in cases])
    take = FusionFeatureCapture.take_packet
    vector_count, encoded_frames, scalar_rows = [0], [0], []
    def check_packet(cap, seq, frame, values):
        packet = take(cap, seq, frame, values)
        assert hashlib.sha256(packet['rgb'].tobytes()).hexdigest() == expected[frame]['RGB_ROI_sha256']
        vector_count[0] += len(values)
        return packet
    def count_encoded(module, args, output):
        encoded_frames[0] += 1
        assert output[0].shape == (1, 320, 60, 108)
    encoder_handle = teacher.backbone.register_forward_hook(count_encoded)
    try:
        with patch.object(FusionFeatureCapture, 'take_packet', check_packet):
            for frame in range(1, 9):
                rows, detections, information = online.step(image(frame), None, sequence, frame)
                assert information['independent_input_batch_size'] == 1
                for case in cases:
                    name, tracker = case['id'], online.trackers[case['id']]
                    np.testing.assert_array_equal(detections[name], expected[frame]['detections'])
                    assert (state_digest(tracker.core), memory_digest(tracker)) == expected[frame]['states'][name]
                    assert all(t.observation is None for g in [tracker.core.tracked_stracks, tracker.core.lost_stracks,
                        tracker.core.removed_stracks] for t in g)
                writer.append(frame, rows, detections)
                scalar_rows.append(dict(frame=frame, detections=len(expected[frame]['detections']),
                    RGB_ROI_sha256=expected[frame]['RGB_ROI_sha256'], all_KF_ID_EMA_memories_exact=True))
                assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= 4*2**30
                print(json.dumps(dict(online_frame=frame, detections=len(expected[frame]['detections']), exact=True)), flush=True)
        streams = writer.close(8)
        for case in cases:
            name = case['id']
            assert streams[name]['track_sha256'] == reference_MOT[name].hexdigest() == sha(streams[name]['track_path'])
            assert online.trackers[name].statistics == reference_stats[name]
    finally:
        encoder_handle.remove()
        writer.abort()
        online.close()
    assert encoded_frames == [8] and len(image_reads) == 16 and loads == [str(checkpoint)]
    assert BaseTrack._count == global_count and not teacher.backbone._forward_hooks
    assert initial == state_sha256(teacher.state_dict()) and all(p.grad is None for p in teacher.parameters())
    assert not any(attempts.values()) and not torch.cuda.is_initialized()
    assert all(sha(ROOT/name) == value for name, value in sources.items())
    require_storage(work)
    records = [dict(case=case, stream=streams[case['id']],
        actual_RGB_memory_updates=updates[case['id']], **reference_stats[case['id']],
        same_MOT_as_zero=streams[case['id']]['track_sha256'] == original_MOT.hexdigest()) for case in cases]
    proof = dict(status='COMPLETE_REAL_TRAIN_RGB_ONLINE_F4_CPU_COMPOSITION', specification_sha256=sha(SPEC),
        source_hashes=sources, synthetic_tests_passed=14, pytest_receipt=str(args.pytest_receipt),
        pytest_receipt_sha256=sha(args.pytest_receipt), pytest_log_sha256=tests['log_sha256'],
        reference_input_content_manifest_sha256=reference['sha256'], source_RGB_checkpoint_sha256=sha(checkpoint),
        source_RGB_checkpoint_loads=1, unique_Train_frames=8, actual_reference_backbone_forwards=8,
        actual_composed_backbone_forwards=8, actual_RGB_ROI_vectors=vector_count[0], input_image_reads=image_reads,
        frames=scalar_rows, cases=records, guarded_attempt_counters=attempts,
        original_RGB_parameters_BN_gradients_unchanged=True, original_global_track_ID_counter_unchanged=True,
        all_current_RGB_ROI_values_exact_to_independent_reference=True,
        all_KF_ID_and_retained_EMA_memory_states_exact=True, all_case_MOT_bytes_exact=True,
        current_consumed_observation_refs_released=True, actual_learned_fusion_transferred=False,
        actual_IR_Dev_Test_images_read=False, GT_read=False, model_fit=False, new_GPU_work=False,
        saved_weights=False, new_image_FPN_ROI_disk_cache=False, new_official_accuracy=False,
        formal_F4_launch_ready=False, fixed_seed=0, additional_seed_work=False,
        seconds=time.monotonic()-started, peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        scope='Eight consecutive actual Train RGB images encoded in CPUFP32 at480x864 B1, two independent passes. '
              'Zero exact to original independentByteTrack; nonzero.05/.1 exact to the original memory component '
              'without observation cleanup. No learned currentIR parent, fullDev/Test,800x1440/GPU parity or efficacy.',
        goal_met=False, supported_innovations=0)
    assert proof['peak_RSS_GiB'] <= 4
    with output.open('x') as stream:
        json.dump(proof, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(output=str(output), seconds=proof['seconds'], peak_RSS_GiB=proof['peak_RSS_GiB'],
                         RGB_ROI_vectors=vector_count[0], new_accuracy=False)), flush=True)


if __name__ == '__main__':
    main()
