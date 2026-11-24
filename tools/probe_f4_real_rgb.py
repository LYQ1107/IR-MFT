#!/usr/bin/env python3
"""Real cached RGB P3 -> ROI -> nonzero memory, without fusion transfer or GT."""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from tools.fusion_memory_capture import FusionFeatureCapture
from tools.indexed_postprocess import indexed_postprocess
from tools.native_pyramid_training_loop import state_sha256
from tools.neck72_runtime import require_host_memory, require_storage
from tools.probe_fusion_memory import NoFeatures, append_tracks, state_digest
from yolox.models.yolo_head import YOLOXHead
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.fusion_memory_tracker import FusionMemoryConfig, FusionMemoryTracker
from yolox.utils.boxes import postprocess

SPEC = ROOT/'configs/perceptual_f4_real_rgb_engineering_20261125.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    started = time.monotonic()
    specification = read(SPEC)
    assert specification['frames'] == 20 and len(specification['clips']) == 5
    assert sha(ROOT/'configs/perceptual_fusion_track_memory_prototype_20261124.json') == specification['F4_prototype_sha256']
    assert sha(ROOT/'configs/perceptual_fusion_memory_capture_prototype_20261125.json') == specification['capture_prototype_sha256']
    plan_path = ROOT/'configs/perceptual_mft_20261122.json'
    assert sha(plan_path) == specification['main_plan_sha256']
    plan = read(plan_path)
    work = Path(plan['work_root'])
    output = work/'fusion_memory_real_RGB_CPU_proof.json'
    assert not output.exists()
    require_host_memory(2*2**30, maximum_workers=1)
    require_storage(work)
    guarded_attempts = dict(forbidden_GT_or_Test_reads=0, non_RGB_checkpoint_loads=0)
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0]))
        if path.name in ['gt.txt', 'targets.npy'] or 'test' in path.parts:
            guarded_attempts['forbidden_GT_or_Test_reads'] += 1
            raise AssertionError('Only cached Train RGB features and metadata may enter this proof')
    sys.addaudithook(audit)
    paths = ['tools/probe_f4_real_rgb.py', str(SPEC.relative_to(ROOT)),
        'tools/fusion_memory_capture.py', 'tools/fusion_memory_features.py',
        'tools/probe_fusion_memory.py', 'tools/indexed_postprocess.py',
        'yolox/utils/boxes.py', 'yolox/models/yolo_head.py', 'yolox/models/network_blocks.py',
        'yolox/tracker/fusion_memory_tracker.py', 'yolox/tracker/byte_tracker.py',
        'yolox/tracker/matching.py', 'yolox/tracker/kalman_filter.py', 'yolox/tracker/basetrack.py']
    sources = {name: sha(ROOT/name) for name in paths}
    previous_audit = read(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261125_MECHANISM_SOURCE_AUDIT.json')
    registries = []
    for record in previous_audit['registrations']:
        path = Path(record['registration'])
        assert sha(path) == record['registration_sha256']
        registry = read(path)
        assert all(sha(ROOT/name) == value for name, value in registry['source_hashes'].items())
        registries.append(registry)
    manifest_path = Path(specification['source_feature_manifest'])
    assert sha(manifest_path) == specification['source_feature_manifest_sha256']
    metadata = read(manifest_path)
    assert metadata['status'] == 'COMPLETE_TRAIN_FROZEN_FPN'
    assert metadata['sequence'] == 'BT-003' and not metadata['Dev_rows'] and not metadata['Test_rows']
    assert metadata['source_RGB_sha256'] == specification['source_RGB_sha256']
    arrays, feature_sources = {}, {}
    for level in [3, 4, 5]:
        info = metadata['outputs'][f'rgb_P{level}']
        assert sha(info['path']) == info['sha256']
        arrays[level] = np.load(info['path'], mmap_mode='r')
        feature_sources[f'P{level}'] = dict(path=info['path'], sha256=info['sha256'], shape=info['shape'])
    completed = read(specification['content_reference_manifest'])
    info = completed['source_image_content_manifest']
    assert sha(info['path']) == info['sha256']
    content = {(row['sequence'], row['frame'], row['modality']): row for row in read(info['path'])}
    checkpoint = Path(specification['source_RGB_checkpoint'])
    assert sha(checkpoint) == specification['source_RGB_sha256']
    loader = torch.load
    checkpoint_loads = []
    def only_rgb(path, *args, **kwargs):
        if Path(path) != checkpoint:
            guarded_attempts['non_RGB_checkpoint_loads'] += 1
            raise AssertionError('A learned fusion checkpoint is not yet eligible for F4')
        checkpoint_loads.append(str(path))
        return loader(path, *args, **kwargs)
    with patch.object(torch, 'load', only_rgb):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    head = YOLOXHead(1, width=1.25).float().eval().requires_grad_(False)
    head.load_state_dict({key.removeprefix('head.'): value for key, value in saved['model'].items()
                         if key.startswith('head.')}, strict=True)
    del saved
    initial = state_sha256(head.state_dict())
    args = SimpleNamespace(track_thresh=.6, track_buffer=30, match_thresh=.9, mot20=False)
    records, raw_sources = [], []
    seen, total_detections, captured_vectors = [], 0, 0
    with torch.no_grad():
        for clip_index, clip in enumerate(specification['clips']):
            assert len(clip['frames']) == len(clip['indices']) == 4
            assert clip['frames'] == list(range(clip['frames'][0], clip['frames'][0]+4))
            sequence = f'BT-003_Train_clip_{clip_index:02d}'
            BaseTrack._count = 0
            baseline = BYTETracker(args, frame_rate=25)
            trackers = {str(weight): FusionMemoryTracker(FusionMemoryConfig(appearance_weight=weight,
                feature_source='rgb', quality_mode='uniform'), args, identity_start=0) for weight in [0., .05, .1]}
            capture = FusionFeatureCapture(None, feature_source='rgb', quality_policy='uniform')
            mot = {name: hashlib.sha256() for name in ['baseline', *trackers]}
            states = {name: hashlib.sha256() for name in ['baseline', *trackers]}
            updates = {name: 0 for name in trackers}
            packet_trace = hashlib.sha256()
            clip_detections = 0
            try:
                for frame, (index, original_frame) in enumerate(zip(clip['indices'], clip['frames']), 1):
                    assert metadata['frame_ids'][index] == original_frame
                    image = Path(plan['data_root'])/'train/BT-003/visible'/f'{original_frame:06d}.jpg'
                    recorded = content['BT-003', original_frame, 'rgb']
                    assert image.stat().st_size == recorded['bytes'] and sha(image) == recorded['sha256']
                    raw_sources.append(dict(frame=original_frame, bytes=recorded['bytes'], sha256=recorded['sha256']))
                    features = tuple(torch.from_numpy(np.array(arrays[level][index], copy=True))[None].float()
                                     for level in [3, 4, 5])
                    decoded = head(features)
                    standard = postprocess(decoded.clone(), 1, .1, .7)[0]
                    indexed = indexed_postprocess(decoded, 1, .1, .7)[0][0]
                    if standard is None:
                        assert indexed is None
                        detections = np.empty((0, 5), dtype=np.float32)
                    else:
                        assert torch.equal(standard, indexed)
                        data = indexed.cpu().numpy()
                        detections = np.c_[data[:, :4]/metadata['ratios'][index], data[:, 4]*data[:, 5]].astype(np.float32)
                    original = detections.copy()
                    capture.begin_frame(sequence, frame, tuple(metadata['image_shapes'][index]), (480, 864),
                                        metadata['ratios'][index])
                    capture.capture_rgb_only(sequence, frame, features[0])
                    packet = capture.take_packet(sequence, frame, detections)
                    assert packet['rgb'].shape == (len(detections), 320)
                    assert capture.current is None and capture.rgb is None
                    assert not packet['auxiliary_available'].any() and 'fused' not in packet and 'auxiliary' not in packet
                    assert np.isfinite(packet['rgb']).all()
                    packet_trace.update(packet['rgb'].tobytes())
                    captured_vectors += len(detections)
                    total_detections += len(detections)
                    clip_detections += len(detections)
                    reference = baseline.update(detections.copy(), (1, 1), (1, 1))
                    reference_state = state_digest(baseline)
                    append_tracks(mot['baseline'], frame, reference)
                    states['baseline'].update(reference_state)
                    global_counter = BaseTrack._count
                    protected_packet = dict(packet, auxiliary=NoFeatures(), fused=NoFeatures(),
                        quality=NoFeatures(), auxiliary_available=NoFeatures())
                    for name, tracker in trackers.items():
                        tracks = tracker.update(sequence, frame, detections, NoFeatures() if name == '0.0' else protected_packet)
                        state = state_digest(tracker.core)
                        states[name].update(state)
                        append_tracks(mot[name], frame, tracks)
                        if name == '0.0':
                            assert state == reference_state
                        for track in tracker.core.tracked_stracks+tracker.core.lost_stracks:
                            assert np.isfinite(track.mean).all() and np.isfinite(track.covariance).all()
                            if track.rgb_memory is not None:
                                assert track.rgb_memory.shape == (320,)
                                assert abs(float(np.linalg.norm(track.rgb_memory))-1.) < 1e-6
                                updates[name] += int(track.memory_last_frame['rgb'] == frame)
                            assert track.aux_memory is None and track.fused_memory is None
                        np.testing.assert_array_equal(detections, original)
                    assert BaseTrack._count == global_counter
                    seen.append(original_frame)
                    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= 2*2**30
                assert mot['baseline'].hexdigest() == mot['0.0'].hexdigest()
                assert states['baseline'].hexdigest() == states['0.0'].hexdigest()
                records.append(dict(clip=sequence, original_frames=clip['frames'], cache_indices=clip['indices'],
                    independent_local_frames=[1, 2, 3, 4], detections=clip_detections,
                    actual_RGB_vector_packet_sha256=packet_trace.hexdigest(),
                    zero_weight_all_KF_and_ID_states_exact=True, zero_weight_literal_MOT_exact=True,
                    trackers={name: dict(weight=tracker.config.appearance_weight,
                        actual_reliable_RGB_memory_updates=updates[name],
                        primary_cost_calls=tracker.statistics['primary_cost_calls'],
                        appearance_penalty_changed_edges=tracker.statistics['changed_edges'],
                        state_trace_sha256=states[name].hexdigest(), literal_MOT_sha256=mot[name].hexdigest())
                        for name, tracker in trackers.items()},
                    baseline_literal_MOT_sha256=mot['baseline'].hexdigest(),
                    ordinary_RGB_auxiliary_and_quality_fields_unreadable=True))
                print(json.dumps(dict(clip=sequence, frames=4, detections=clip_detections,
                    updates=updates, changed_edges={n: t.statistics['changed_edges'] for n, t in trackers.items()})), flush=True)
            finally:
                capture.close()
    assert len(seen) == len(set(seen)) == 20 and total_detections == captured_vectors
    assert initial == state_sha256(head.state_dict()) and all(p.grad is None for p in head.parameters())
    assert checkpoint_loads == [str(checkpoint)] and not any(guarded_attempts.values())
    assert not torch.cuda.is_initialized() and sources == {name: sha(ROOT/name) for name in sources}
    assert all(sha(ROOT/name) == value for r in registries for name, value in r['source_hashes'].items())
    require_storage(work)
    result = dict(status='COMPLETE_REAL_TRAIN_RGB_P3_NONZERO_MEMORY_CPU_ENGINEERING',
        specification_sha256=sha(SPEC), source_hashes=sources,
        source_feature_manifest_sha256=sha(manifest_path), RGB_FPN_sources=feature_sources,
        source_RGB_checkpoint_sha256=sha(checkpoint), actual_source_RGB_checkpoint_loads=1,
        source_image_content_reference_sha256=info['sha256'], RGB_source_images=raw_sources,
        clips=records, unique_real_Train_frames=20, actual_RGB_ROI_vectors=captured_vectors,
        guarded_attempt_counters=guarded_attempts, source_head_parameters_BN_and_gradients_unchanged=True,
        all9_prior_source_registrations_unchanged=True, actual_new_image_encoding=False,
        model_fit=False, actual_GT_read=False, actual_Dev_Test_read=False,
        actual_learned_fusion_features_transferred=False, ordinary_RGB_control_only=True,
        CPU_FP32_480x864_not_new_GPU_parity=True, new_official_accuracy=False,
        formal_F4_launch_ready=False, new_image_or_FPN_disk_cache=False, saved_weights=False,
        GPU_work=False, fixed_seed=0, additional_seed_work=False,
        seconds=time.monotonic()-started,
        peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
        scope='Five explicitly independent4-frameTrainclips on real learned RGB P3, original frozen '
            'CPUFP32YOLOXhead and exact originalNMS. Weight0 exact originalByteTrack; weights.05/.1 '
            'receive actualRGBROI vectors with q1. No learned fusion checkpoint/feature, GT,Dev/Test, '
            'new encoding, full sequence efficacy or innovation. Shortclipresets are engineering only.',
        supported_innovations=0, goal_met=False)
    assert result['peak_process_RSS_GiB'] <= 2
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], seconds=result['seconds'],
        vectors=captured_vectors, peak_process_RSS_GiB=result['peak_process_RSS_GiB'])), flush=True)


if __name__ == '__main__':
    main()
