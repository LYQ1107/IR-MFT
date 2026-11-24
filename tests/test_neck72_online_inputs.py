import copy
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from tools import neck72_runtime_study as runtime
from tools import launch_neck72_runtime as launch
from tools.benchmark_neck72_runtime import source_track_prefix
from tools.neck72_online_inputs import (needs_ir, rgb_masks, rgb_point_groups,
    rgb_frame_correspondences, detector_input, appearance_inputs, PrefixPackets,
    CurrentPointStream, RGB_MEAN, RGB_STD)
from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.data.data_augment_rgbt import RGBTValTransform
from yolox.tracker.foreground_transport import point_groups, frame_correspondences


def config(arm='rgb', cue='rgb', **extra):
    return dict(detection_arm=arm, backend='consensus_query', native_cue=cue,
                disable_native_increment=False, **extra)


def test_actual_ir_is_requested_only_by_consumed_pixels():
    assert not needs_ir(config())
    assert needs_ir(config(cue='ir'))
    missing = config(cue='ir'); missing['disable_native_increment'] = True
    assert not needs_ir(missing)
    assert needs_ir(dict(missing, detection_arm='new'))
    assert not needs_ir(dict(backend='byte', detection_arm='rgb'))
    assert needs_ir(dict(backend='byte', detection_arm='new'))


def test_rgb_descriptors_and_points_never_access_an_ir_object():
    image = np.random.default_rng(0).integers(0, 256, (48, 80, 3), np.uint8)
    boxes = np.array([[3., 4., 62., 41.]])
    arrays, views = appearance_inputs(image, object(), boxes, False)
    assert np.isfinite(arrays['quality'][:, 0]).all() and np.isnan(arrays['quality'][:, 1]).all()
    assert len(views) == 2 and views[0].shape == (3, 128, 256)
    groups = rgb_point_groups(image, object(), boxes, maximum=20)
    assert set(groups['groups']) == {'bbox', 'rgb'}


@pytest.mark.parametrize('variant', ['rgb', 'diffv2ir'])
def test_detector_preparation_equals_original_paired_val_transform(variant):
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (37, 71, 3), np.uint8)
    ir = rng.integers(0, 256, (37, 71), np.uint8)
    metadata = dict(variant=variant, input_size=[800, 1440],
                    ir_means=[.318765696840321]*3, ir_std=[.11454358863241268]*3)
    transform = RGBTValTransform(RGB_MEAN, RGB_STD, metadata['ir_means'], metadata['ir_std'])
    expected, _ = transform(rgb, np.repeat(ir[:, :, None], 3, axis=2), None, (800, 1440))
    actual, ratio = detector_input(rgb, ir if variant != 'rgb' else object(), metadata)
    np.testing.assert_array_equal(actual, expected if variant != 'rgb' else expected[:3])
    assert ratio == min(800/37, 1440/71)


def test_pruned_point_modes_match_original_groups_and_pooled_lk_votes():
    cv2.setNumThreads(1)
    rng = np.random.default_rng(0)
    old = rng.integers(0, 256, (72, 128, 3), np.uint8)
    current = np.roll(old, 1, axis=1); ir = rng.integers(0, 256, (72, 128), np.uint8)
    boxes = np.array([[5., 7., 61., 53.], [40., 12., 105., 65.]], np.float32)
    reference_old = point_groups(old, ir, boxes); reference_current = point_groups(current, ir, boxes+np.array([1, 0, 1, 0]))
    pruned_old = rgb_point_groups(old, None, boxes); pruned_current = rgb_point_groups(current, None, boxes+np.array([1, 0, 1, 0]))
    for mode in ['bbox', 'rgb']:
        for key in ['points', 'owners', 'weights', 'total', 'denominator']:
            np.testing.assert_array_equal(pruned_old['groups'][mode][key], reference_old['groups'][mode][key])
        for left, right in zip(pruned_old['masks'][mode], reference_old['masks'][mode]):
            np.testing.assert_array_equal(left, right)
    reference = frame_correspondences(old, current, {'x': reference_old}, {'x': reference_current})
    pruned = rgb_frame_correspondences(old, current, {'x': pruned_old}, {'x': pruned_current})
    for key, value in pruned.items():
        for a, b in zip(value, reference[key]):
            np.testing.assert_array_equal(a, b)


def test_sequential_packet_offsets_survive_an_empty_current_frame():
    rgb = np.random.default_rng(0).integers(0, 256, (32, 64, 3), np.uint8)
    packets, points = PrefixPackets(3), CurrentPointStream()
    for frame, detections in enumerate([np.array([[1., 2., 30., 24., .9]], np.float32),
                                       np.empty((0, 5), np.float32),
                                       np.array([[2., 3., 31., 25., .8]], np.float32)], 1):
        transport = points.step(frame, rgb, detections[:, :4])
        data = packets.append(frame, detections, {}, transport)
    np.testing.assert_array_equal(data['offsets'], [0, 1, 1, 2])
    np.testing.assert_array_equal(data['foreground_transport']['count_offsets_1'], [0, 0, 1, 1])
    np.testing.assert_array_equal(data['foreground_transport']['vote_offsets_2'], [0, 0, 0, 1])
    with pytest.raises(ValueError, match='sequential'):
        packets.append(3, np.empty((0, 5)), {}, {})
    with pytest.raises(ValueError, match='at most64'):
        PrefixPackets(65)


def test_runtime_numerical_limits_keep_shape_and_nonfinite_failures():
    assert runtime.compare_array(np.array([1.]), np.array([1.00001]), .0001)['passed']
    assert not runtime.compare_array(np.array([1.]), np.array([1.00001]), 0)['passed']
    assert not runtime.compare_array(np.array([np.nan]), np.array([1.]), 1)['passed']
    assert not runtime.compare_array(np.ones((2, 3)), np.ones((2, 4)), 1)['passed']
    records = [dict(frame=1, passed=True, fields={'vector': runtime.compare_array([1.], [1.], 0)}),
        dict(frame=2, passed=False, fields={'vector': runtime.compare_array([1.], [1.1], 0)})]
    result = runtime.comparison_summary(records)
    assert not result['all_passed'] and result['fields']['vector']['failed_frames'] == [2]
    assert result['fields']['vector']['maximum_absolute_difference'] > .09


def test_prefix_reference_keeps_original_ids_and_formatter_bytes(tmp_path):
    source = tmp_path / 'neck72_dev/case/track_results/BT-005.txt'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'1,972,1.000,2.000,3.000,4.000,0.900,-1,-1,-1\n33,973,1,2,3,4,.9,-1,-1,-1\n')
    output = tmp_path / 'prefix.txt'
    info = source_track_prefix(tmp_path, 'dev', 'case', 'BT-005', 32, output)
    assert output.read_bytes() == source.read_bytes().splitlines(keepends=True)[0]
    assert b',972,' in output.read_bytes() and info['prefix_sha256'] == sha256(output)


def test_process_release_distinguishes_live_pid_and_reused_identity():
    pid = os.getpid()
    ticks = int(Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19])
    assert not runtime.exited(pid, identity=dict(start_ticks=ticks))
    assert runtime.exited(pid, identity=dict(start_ticks=ticks+1))
    assert runtime.exited(pid, command=['a-different-original-worker'])


def test_complete_accuracy_state_does_not_replace_actual_exit_and_exports(tmp_path, monkeypatch):
    study = tmp_path / 'study.json'; dump(study, dict(seed=0))
    monkeypatch.setattr(runtime, 'STUDY', study)
    report = tmp_path / 'report.json'; monkeypatch.setattr(runtime, 'REPORT', report)
    dump(report, dict(status='COMPLETE_NECK72_SELECTED_ALL7TEST_REQUIRES_FULL_SYSTEM_COST_AUDIT', study_sha256=sha256(study)))
    pid = os.getpid(); ticks = int(Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19])
    state = dict(status=runtime.TERMINAL, study_sha256=sha256(study), controller_pid=pid,
                 phases={'test': {'case': dict(status='COMPLETE', exit_code=0, pid=pid, command=['python'])}})
    dump(tmp_path / 'neck72_study_state.json', state)
    dump(tmp_path / 'neck72_resource_guard_R2_controller_identity.json', dict(pid=pid, start_ticks=ticks))
    monkeypatch.setattr(runtime, 'cases', lambda work: {'cases': [dict(id='case', config={'seed': 0})]})
    assert not runtime.predecessor_ready(tmp_path)
    monkeypatch.setattr(runtime, 'exited', lambda *args, **kwargs: True)
    for split, frames in [('dev', 4834), ('test', 15985)]:
        dump(tmp_path / ('neck72_'+split) / 'case/inference_manifest.json',
             dict(status='INFERENCE_COMPLETE_BEFORE_GT_EVALUATION', config={'seed': 0}, frames=frames))
    assert runtime.predecessor_ready(tmp_path)
    state['phases']['test']['case']['exit_code'] = 1
    dump(tmp_path / 'neck72_study_state.json', state)
    assert not runtime.predecessor_ready(tmp_path)


def test_runtime_gpu_rejects_wrong_card_and_unfinished_predecessor_before_cuda(monkeypatch, tmp_path):
    assert not torch.cuda.is_initialized()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2')
    with pytest.raises(ValueError, match='physicalGPU4/5'):
        runtime.gpu_settings(tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4')
    monkeypatch.setattr(runtime, 'predecessor_ready', lambda work: False)
    with pytest.raises(RuntimeError, match='actual process exit'):
        runtime.gpu_settings(tmp_path)
    assert not torch.cuda.is_initialized()


def test_zero_worker_exit_without_matching_runtime_evidence_fails(monkeypatch, tmp_path):
    dump(tmp_path / runtime.REGISTRY, dict(source_hashes={}))
    monkeypatch.setattr(launch, 'require_storage', lambda *args: None)
    reservations = []
    monkeypatch.setattr(launch, 'require_host_memory',
        lambda incoming, maximum_workers: reservations.append((incoming, maximum_workers)))
    monkeypatch.setattr(launch.time, 'sleep', lambda seconds: None)
    class Done:
        pid = os.getpid()
        def poll(self):
            return 0
    monkeypatch.setattr(launch.subprocess, 'Popen', lambda *args, **kwargs: Done())
    state = dict(phases={})
    queue = launch.Queue(tmp_path, state, dict(memory={'maximum_case_growth_bytes': 1024}))
    with pytest.raises(FileNotFoundError):
        queue.jobs('dev', [dict(id='case', config={'native_seed': 0})])
    assert state['phases']['dev']['case']['status'] == 'FAILED_EVIDENCE'
    assert reservations == [(8*2**30, 2)]
