"""Artificial teacher/untrained fusers only; no checkpoints, datasets or HOTA."""
import hashlib

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from tools.fusion_memory_design import make_case
from tools.fusion_memory_online import OnlineFusionMemory, discard_consumed_observations, visual_routes
from tools.fusion_memory_streams import FusionMemoryStreamWriter
from tools.native_pyramid_stream_tracks import StreamTrackWriter
from tools.native_pyramid_train_data import preprocess_observation
from tools.probe_fusion_memory import state_digest
from tools.spatial_native_fusion import SpatialNativeFusion
from yolox.models.native_pyramid_fusion import NativePyramidFusion
from yolox.tracker.fusion_memory_tracker import FusionMemoryConfig, FusionMemoryTracker


CALIBRATION = {'statistics': {key: dict(mean=.5, std=.25) for key in ['gray', 'ir']}}


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.calls = 0

    def forward(self, pixels):
        self.calls += 1
        base = pixels.mean(1, keepdim=True)
        rows = []
        for channels, scale in [(8, 8), (16, 16), (32, 32)]:
            value = F.avg_pool2d(base, scale)
            rows.append(torch.cat([value*(1+i*.05)+i*.1 for i in range(channels)], 1))
        return tuple(rows)


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.calls, self.fail_at, self.empty = 0, None, False

    def forward(self, features):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('Intentional synthetic forward failure')
        count = sum(v.shape[-2]*v.shape[-1] for v in features)
        output = torch.zeros(1, count, 6, device=features[0].device)
        output[:, :, 0:4] = torch.tensor([30., 20., 24., 24.], device=output.device)
        output[:, :, 5] = 1.
        if not self.empty:
            output[:, 0, 4] = .95
            output[:, 1, 0] = 70.
            output[:, 1, 4] = .9
        return output


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, self.head = TinyBackbone(), TinyHead()
        self.eval().requires_grad_(False)


def visual(kind='quality', baseline=False):
    family = 'direct' if baseline else ('spatial' if kind == 'spatial' else 'native')
    identifier = 'artificial_'+kind+('_RGB' if baseline else '_IR')
    model = dict(id=identifier, cue='rgb' if baseline else 'ir', seed=0)
    if baseline:
        model['baseline'] = 'fresh_RGB_B1'
    else:
        model.update(mode='quality' if kind == 'spatial' else kind,
                     geometry_fixed=False, intervention='normal')
    return dict(id=identifier, family=family, cue=model['cue'], baseline=baseline, model=model)


def module(kind):
    torch.manual_seed(0)
    kwargs = dict(channels=(8, 16, 32))
    value = SpatialNativeFusion(**kwargs) if kind == 'spatial' else NativePyramidFusion(**kwargs, mode=kind)
    with torch.no_grad():
        value.output[0].bias.fill_(.125)  # Fixed engineering perturbation; no fitting.
    return value.float().eval().requires_grad_(False)


def inputs(frame=1):
    rgb = np.random.default_rng(frame).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    ir = np.arange(32*48, dtype=np.uint8).reshape(32, 48)
    return rgb, np.roll(ir, frame, 1)


def engine(cases, kind='quality'):
    routes, _ = visual_routes(cases)
    modules = {key: None if r['visual']['baseline'] else module(kind) for key, r in routes.items()}
    teacher = TinyTeacher()
    return OnlineFusionMemory(teacher, cases, modules, (64, 96), CALIBRATION), teacher, modules


def memory_digest(tracker):
    h = hashlib.sha256()
    for group in [tracker.core.tracked_stracks, tracker.core.lost_stracks, tracker.core.removed_stracks]:
        for track in group:
            for name in ['rgb_memory', 'aux_memory', 'fused_memory']:
                value = getattr(track, name)
                h.update(b'NONE' if value is None else value.tobytes())
            h.update(repr(track.memory_last_frame).encode())
            h.update(repr(track.memory_last_reliability).encode())
    return h.digest()


@pytest.mark.parametrize('kind', ['quality', 'temporal', 'spatial'])
def test_shared_real_native_forward_routes_capture_profiles_and_consecutive_state(kind):
    source = visual(kind)
    cases = [make_case(source, 'byte', 0.)]+[make_case(source, profile, weight)
        for profile in ['rgb_uniform', 'rgb_attention_proxy', 'fused_uniform', 'fused_attention_proxy'] for weight in [.05, .1]]
    value, teacher, modules = engine(cases, kind)
    try:
        for frame in range(1, 5):
            rows, packets, info = value.step(*inputs(frame), 'artificial_video', frame)
            assert set(rows) == {c['id'] for c in cases} and set(rows) == set(packets)
            assert teacher.backbone.calls == frame
            assert len(value.captures) == 4 and len(value.routes) == 1
            assert all(cap.rgb is None and cap.current is None for cap in value.captures.values())
            assert all(t.observation is None for tracker in value.trackers.values() for t in tracker.core.tracked_stracks)
            assert info['independent_input_batch_size'] == 1 and not info['zero_weight_feature_reads']
        if kind == 'temporal':
            state = next(iter(value.detector.states.values()))
            assert state.frame == 4 and state.sequence == 'artificial_video'
        rows, _, _ = value.step(*inputs(1), 'second_video', 1)
        assert all([row[2] for row in group] == [1, 2] for group in rows.values())
    finally:
        value.close()
    assert not teacher.backbone._forward_hooks
    assert all(not m._forward_hooks and not m._forward_pre_hooks and not m.encoder._forward_hooks for m in modules.values())


def test_zero_weight_registers_no_capture_or_backbone_hook():
    cases = [make_case(visual(baseline=True), 'byte', 0.)]
    value, teacher, _ = engine(cases)
    assert value.captures == {} and not teacher.backbone._forward_hooks
    value.step(inputs()[0], None, 'video', 1)
    value.close()


@pytest.mark.parametrize('floor', [.2, 0.])
def test_missing_ir_matches_rgb_same_weight_and_floor_for_all_profiles(floor):
    rgb, ir = visual(baseline=True), visual()
    cases = [make_case(rgb, 'rgb_uniform', w, floor=floor) for w in [.05, .1]]
    cases += [make_case(ir, p, w, floor=floor, intervention='IR_unavailable') for p in
        ['rgb_uniform', 'rgb_attention_proxy', 'fused_uniform', 'fused_attention_proxy'] for w in [.05, .1]]
    value, _, _ = engine(cases)
    assert not value.needs_actual_IR_image
    try:
        for frame in range(1, 5):
            rows, packets, _ = value.step(inputs(frame)[0], None, 'video', frame)
            for case in cases[2:]:
                control = next(c for c in cases[:2] if c['appearance_weight'] == case['appearance_weight'])
                a, b = value.trackers[case['id']], value.trackers[control['id']]
                np.testing.assert_array_equal(packets[case['id']], packets[control['id']])
                assert state_digest(a.core) == state_digest(b.core)
                assert memory_digest(a) == memory_digest(b) and a.statistics == b.statistics
    finally:
        value.close()


def test_ir_roll_enters_visual_encoder_as_actual_current_pixel_roll():
    source = visual()
    cases = [make_case(source, 'fused_uniform', .05, intervention=i) for i in ['normal', 'IR_image_spatial_roll']]
    value, _, modules = engine(cases)
    observed = {}
    handles = [m.encoder.register_forward_pre_hook(lambda m, args, key=key: observed.update({key: args[0].clone()}))
               for key, m in modules.items()]
    rgb, ir = inputs()
    try:
        value.step(rgb, ir, 'video', 1)
        for key, route in value.routes.items():
            image = np.roll(ir, (8, 12), (0, 1)) if route['intervention'] != 'normal' else ir
            expected = preprocess_observation(rgb, image, 'ir', (64, 96), CALIBRATION)[1]
            np.testing.assert_array_equal(observed[key].numpy()[0], expected)
    finally:
        for handle in handles:
            handle.remove()
        value.close()


def test_modules_cannot_alias_intervention_states():
    source = visual()
    cases = [make_case(source, 'fused_uniform', .05, intervention=i) for i in ['normal', 'IR_unavailable']]
    routes, _ = visual_routes(cases)
    shared = module('quality')
    with pytest.raises(ValueError, match='cannot alias'):
        OnlineFusionMemory(TinyTeacher(), cases, {key: shared for key in routes}, (64, 96), CALIBRATION)
    assert not shared._forward_hooks


def test_failed_forward_releases_capture_and_cannot_retry_partially_advanced_state():
    case = make_case(visual(), 'fused_uniform', .05)
    value, teacher, _ = engine([case])
    teacher.head.fail_at = 2
    with pytest.raises(RuntimeError, match='Intentional'):
        value.step(*inputs(), 'video', 1)
    assert value.failed and all(cap.current is None and cap.rgb is None for cap in value.captures.values())
    with pytest.raises(ValueError, match='failed online state'):
        value.step(*inputs(), 'video', 1)
    value.close()


def test_skipped_or_duplicate_frame_is_rejected_without_consuming_state():
    value, _, _ = engine([make_case(visual(baseline=True), 'rgb_uniform', .05)])
    try:
        value.step(inputs()[0], None, 'video', 1)
        for frame in [1, 3]:
            with pytest.raises(ValueError, match='consecutive'):
                value.step(inputs()[0], None, 'video', frame)
        value.step(inputs(2)[0], None, 'video', 2)
    finally:
        value.close()


def test_cleanup_keeps_exact_future_memory_association_through_lost_removed_rebirth():
    reference = FusionMemoryTracker(FusionMemoryConfig(feature_source='fused'))
    cleaned = FusionMemoryTracker(FusionMemoryConfig(feature_source='fused'))
    prefix = 0
    for frame in range(1, 45):
        count = 1 if frame in [1, 2, 6, 40, 41, 42, 43, 44] else 0
        values = np.array([[4., 4., 18., 18., .95]], np.float32)[:count]
        vector = np.array([[np.cos(frame*.01), np.sin(frame*.01)]], np.float32)[:count]
        packet = dict(rgb=vector, auxiliary=vector, fused=vector,
            auxiliary_available=np.ones(count, bool), quality=np.ones(count))
        reference.update('video', frame, values, packet)
        cleaned.update('video', frame, values, packet)
        prefix = discard_consumed_observations(cleaned, prefix)
        assert state_digest(reference.core) == state_digest(cleaned.core)
        assert memory_digest(reference) == memory_digest(cleaned)
        assert reference.statistics == cleaned.statistics
    assert reference.core.removed_stracks
    assert any(t.observation is not None for t in reference.core.removed_stracks)
    assert all(t.observation is None for t in cleaned.core.removed_stracks)


def test_stream_writer_zero_matches_original_native_writer_literal_bytes(tmp_path):
    case = make_case(visual(baseline=True), 'byte', 0.)
    value, teacher, _ = engine([case])
    name = case['id']
    writer = FusionMemoryStreamWriter(tmp_path/'memory', 'video', [name])
    reference = StreamTrackWriter(tmp_path/'original', 'video', [name])
    try:
        for frame in range(1, 5):
            teacher.head.empty = frame == 2
            rows, packets, _ = value.step(inputs(frame)[0], None, 'video', frame)
            writer.append(frame, rows, packets)
            reference.append(frame, packets)
        a, b = writer.close(4)[name], reference.close(4)[name]
        for key in ['track_sha256', 'detection_packet_sha256', 'detections', 'track_rows']:
            assert a[key] == b[key]
        assert (tmp_path/'memory'/name/'track_results/video.txt').read_bytes() == (
            tmp_path/'original'/name/'track_results/video.txt').read_bytes()
    finally:
        value.close()
        writer.abort()
        reference.abort()


def test_incomplete_writer_keeps_prefix_without_complete_receipt(tmp_path):
    writer = FusionMemoryStreamWriter(tmp_path, 'video', ['case'])
    writer.append(1, {'case': []}, {'case': np.empty((0, 5), np.float32)})
    with pytest.raises(ValueError, match='incomplete prefix'):
        writer.close(2)
    with pytest.raises(ValueError, match='Closed'):
        writer.close(1)
    assert writer.paths['case'].exists()
    with pytest.raises(ValueError, match='Preserve'):
        FusionMemoryStreamWriter(tmp_path, 'video', ['case'])


def test_writer_rejects_modified_emitted_file(tmp_path):
    writer = FusionMemoryStreamWriter(tmp_path, 'video', ['case'])
    writer.append(1, {'case': []}, {'case': np.empty((0, 5), np.float32)})
    with writer.paths['case'].open('a') as stream:
        stream.write('CHANGED\n')
    with pytest.raises(ValueError, match='Written MOT bytes differ'):
        writer.close(1)
