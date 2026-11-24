"""Synthetic P3/coordinate/lifecycle tests; no learned checkpoints or images."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from tools.fusion_memory_capture import FusionFeatureCapture
from tools.fusion_memory_features import pool_p3
from tools.spatial_native_fusion import SpatialNativeFusion
from yolox.models.native_pyramid_fusion import NativePyramidFusion
from yolox.tracker.fusion_memory_tracker import FusionMemoryConfig, FusionMemoryTracker


class Unreadable:
    def __getitem__(self, key):
        raise AssertionError('Unneeded auxiliary/fused/quality output was read')
    def __array__(self, *args, **kwargs):
        raise AssertionError('Unneeded auxiliary array was read')


def planes(channels=2):
    x = ((torch.arange(12, dtype=torch.float32)+.5)*8)[None, :].expand(8, 12)
    y = ((torch.arange(8, dtype=torch.float32)+.5)*8)[:, None].expand(8, 12)
    base = torch.stack([x, y]+[x+y for _ in range(channels-2)])[None]
    return tuple(torch.nn.functional.avg_pool2d(base, factor) for factor in [1, 2, 4])


class PlaneEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
    def forward(self, cue, valid):
        self.calls += 1
        raw = tuple(torch.stack([torch.ones_like(v[:, 0]), torch.ones_like(v[:, 0])*2], 1) for v in planes())
        return raw, tuple(torch.ones_like(v[:, :1], dtype=torch.bool) for v in raw), torch.zeros(1, 3)


class PlaneFusion(nn.Module):
    def __init__(self, filtered=False, gate=.25, bad_output=False):
        super().__init__()
        self.encoder = PlaneEncoder()
        self.filtered, self.gate, self.bad_output = filtered, gate, bad_output
    def forward(self, rgb, cue, scores, valid=None, available=True, sequence=None, frame=None):
        if self.bad_output:
            return Unreadable()
        if not available:
            return rgb, None, dict(missing_exact_RGB=True, levels=Unreadable())
        auxiliary, masks, _ = self.encoder(cue, valid)
        fused = tuple(v+torch.tensor([3., 4.])[None, :, None, None] for v in rgb)
        state = SimpleNamespace(sequence=sequence, frame=frame,
            latent=tuple(v+5 for v in auxiliary)) if self.filtered else None
        return fused, state, dict(missing_exact_RGB=False,
            levels=[dict(gate=torch.tensor([[self.gate]])) for _ in range(3)])


def arm(capture, frame=1, sequence='synthetic', original=(32, 48)):
    capture.begin_frame(sequence, frame, original, (64, 96), 2.)


def forward(model, frame=1, available=True, rgb=None):
    rgb = planes() if rgb is None else rgb
    return model(rgb, None, tuple(torch.ones_like(v[:, :1])*.7 for v in rgb),
        available=available, sequence='synthetic', frame=frame)


def detections():
    return np.array([[4., 2., 20., 10., .9]], dtype=np.float32)


def test_known_actual_triplet_roi_and_proxy_have_correct_source_coordinates():
    model = PlaneFusion()
    capture = FusionFeatureCapture(model, feature_source='fused', quality_policy='attention_proxy')
    arm(capture)
    forward(model)
    original = detections()
    before = original.copy()
    packet = capture.take_packet('synthetic', 1, original)
    np.testing.assert_allclose(packet['rgb'][0], np.array([24., 12.])/np.linalg.norm([24., 12.]), rtol=1e-6)
    np.testing.assert_allclose(packet['auxiliary'][0], np.array([1., 2.])/np.linalg.norm([1., 2.]), rtol=1e-6)
    np.testing.assert_allclose(packet['fused'][0], np.array([27., 16.])/np.linalg.norm([27., 16.]), rtol=1e-6)
    np.testing.assert_array_equal(packet['quality'], [.25])
    np.testing.assert_array_equal(original, before)
    assert packet['auxiliary_available'].tolist() == [True]
    assert packet['metadata']['quality_is_calibrated_reliability'] is False
    capture.close()


def test_temporal_uses_current_filtered_state_instead_of_raw_encoder_value():
    model = PlaneFusion(filtered=True)
    capture = FusionFeatureCapture(model, feature_source='fused', quality_policy='uniform')
    arm(capture)
    forward(model)
    packet = capture.take_packet('synthetic', 1, detections())
    np.testing.assert_allclose(packet['auxiliary'][0], np.array([6., 7.])/np.linalg.norm([6., 7.]), rtol=1e-6)
    assert packet['metadata']['auxiliary_source'] == 'current_filtered_native_state'
    capture.close()


def test_strong_rgb_uniform_never_reads_visual_outputs_or_auxiliary_diagnostics():
    model = PlaneFusion(bad_output=True)
    capture = FusionFeatureCapture(model, feature_source='rgb', quality_policy='uniform')
    arm(capture)
    forward(model)
    packet = capture.take_packet('synthetic', 1, detections())
    assert 'fused' not in packet and 'auxiliary' not in packet
    assert packet['metadata']['auxiliary_image_available'] is None
    assert packet['quality'].tolist() == [1.]
    capture.close()


def test_same_q_rgb_control_pools_no_auxiliary_or_fused_embedding():
    model = PlaneFusion()
    capture = FusionFeatureCapture(model, feature_source='rgb', quality_policy='attention_proxy')
    arm(capture)
    forward(model)
    assert capture.raw_auxiliary is None and capture.fused is None
    packet = capture.take_packet('synthetic', 1, detections())
    assert 'auxiliary' not in packet and 'fused' not in packet
    assert packet['auxiliary_available'].tolist() == [True] and packet['quality'].tolist() == [.25]
    capture.close()


def test_uniform_fused_route_does_not_require_quality_fields():
    model = PlaneFusion()
    capture = FusionFeatureCapture(model, feature_source='fused', quality_policy='uniform')
    arm(capture)
    rgb = planes()
    # The normal forward supplies a valid triplet; uniform policy never reads its gate.
    model.gate = float('nan')
    forward(model, rgb=rgb)
    packet = capture.take_packet('synthetic', 1, detections())
    assert packet['quality'].tolist() == [1.] and 'fused' in packet
    capture.close()


def test_zero_attention_is_valid_q_and_does_not_change_image_availability():
    model = PlaneFusion(gate=0.)
    capture = FusionFeatureCapture(model, feature_source='fused', quality_policy='attention_proxy')
    arm(capture)
    forward(model)
    packet = capture.take_packet('synthetic', 1, detections())
    assert packet['quality'].tolist() == [0.] and packet['auxiliary_available'].tolist() == [True]
    capture.close()


def test_missing_observation_clears_previous_ir_without_reading_missing_arrays():
    model = PlaneFusion()
    capture = FusionFeatureCapture(model, feature_source='fused', quality_policy='attention_proxy')
    arm(capture)
    forward(model)
    assert 'auxiliary' in capture.take_packet('synthetic', 1, detections())
    arm(capture, frame=2)
    forward(model, frame=2, available=False)
    packet = capture.take_packet('synthetic', 2, detections())
    assert not packet['auxiliary_available'].any() and 'auxiliary' not in packet and 'fused' not in packet
    assert packet['quality'].tolist() == [1.] and model.encoder.calls == 1
    capture.close()


def test_off_image_and_empty_candidates_are_preserved_without_descriptors():
    model = PlaneFusion()
    capture = FusionFeatureCapture(model, feature_source='fused', quality_policy='uniform')
    arm(capture)
    forward(model)
    off_image = np.array([[60., 40., 65., 45., .8]], np.float32)
    packet = capture.take_packet('synthetic', 1, off_image)
    assert len(packet['rgb']) == 1 and not packet['rgb'].any() and not packet['auxiliary_available'].any()
    arm(capture, frame=2)
    forward(model, frame=2)
    empty = capture.take_packet('synthetic', 2, np.empty((0, 5), np.float32))
    assert empty['rgb'].shape == (0, 2) and len(empty['quality']) == 0
    capture.close()


def test_lifecycle_rejects_stale_double_and_wrong_frame_capture():
    model = PlaneFusion()
    capture = FusionFeatureCapture(model, feature_source='fused', quality_policy='uniform')
    with pytest.raises(ValueError):
        forward(model)
    arm(capture)
    with pytest.raises(ValueError):
        forward(model, frame=2)
    forward(model)
    with pytest.raises(ValueError):
        forward(model)
    with pytest.raises(ValueError):
        capture.take_packet('synthetic', 2, detections())
    capture.take_packet('synthetic', 1, detections())
    with pytest.raises(ValueError):
        arm(capture, frame=3)
    arm(capture, frame=2)
    capture.abort_frame()
    arm(capture, frame=2)
    capture.abort_frame()
    arm(capture, frame=1, sequence='new_video')
    capture.abort_frame()
    capture.close()
    with pytest.raises(ValueError):
        arm(capture)


def test_packet_is_owned_detached_and_capture_releases_current_features():
    model = PlaneFusion()
    capture = FusionFeatureCapture(model, feature_source='fused', quality_policy='uniform')
    rgb = tuple(v.requires_grad_() for v in planes())
    arm(capture)
    forward(model, rgb=rgb)
    assert not capture.rgb.requires_grad and not capture.fused.requires_grad and not capture.auxiliary.requires_grad
    packet = capture.take_packet('synthetic', 1, detections())
    original = packet['rgb'].copy()
    with torch.no_grad():
        rgb[0].add_(500.)
    np.testing.assert_array_equal(packet['rgb'], original)
    assert capture.current is None and capture.rgb is None and capture.auxiliary is None
    capture.close()


@pytest.mark.parametrize('kind', ['quality', 'temporal', 'spatial'])
def test_actual_untrained_native_families_match_returned_p3_without_checkpoint(kind):
    torch.manual_seed(0)
    kwargs = dict(channels=(8, 16, 32))
    model = SpatialNativeFusion(**kwargs) if kind == 'spatial' else NativePyramidFusion(
        **kwargs, mode=kind, use_motion=False)
    model.eval().requires_grad_(False)
    with torch.no_grad():
        model.output[0].bias.fill_(.2)  # Deterministic engineering value, never fitted weights.
    rgb = tuple(torch.randn(1, c, 64//s, 96//s) for c, s in zip((8, 16, 32), (8, 16, 32)))
    cue = torch.randn(1, 3, 64, 96)
    scores = tuple(torch.ones_like(v[:, :1])*.2 for v in rgb)
    capture = FusionFeatureCapture(model, feature_source='fused', quality_policy='attention_proxy')
    arm(capture)
    result = model(rgb, cue, scores, sequence='synthetic', frame=1)
    packet = capture.take_packet('synthetic', 1, detections())
    expected, _ = pool_p3(result[0][0], detections()[:, :4], (32, 48), (64, 96), 2.)
    np.testing.assert_array_equal(packet['fused'], expected.numpy())
    assert not np.allclose(packet['rgb'], packet['fused'])
    if kind == 'temporal':
        assert packet['metadata']['auxiliary_source'] == 'current_filtered_native_state'
    capture.close()


def test_whole_missing_capture_to_tracker_matches_ordinary_rgb_memory():
    model = PlaneFusion()
    fused_capture = FusionFeatureCapture(model, feature_source='fused', quality_policy='attention_proxy')
    rgb_capture = FusionFeatureCapture(model, feature_source='rgb', quality_policy='uniform')
    fused = FusionMemoryTracker(FusionMemoryConfig(feature_source='fused', quality_mode='provided'))
    rgb = FusionMemoryTracker(FusionMemoryConfig(feature_source='rgb', quality_mode='uniform'))
    for frame in range(1, 5):
        arm(fused_capture, frame)
        arm(rgb_capture, frame)
        forward(model, frame=frame, available=False)
        boxes = detections()
        a = fused.update('synthetic', frame, boxes, fused_capture.take_packet('synthetic', frame, boxes))
        b = rgb.update('synthetic', frame, boxes, rgb_capture.take_packet('synthetic', frame, boxes))
        assert [t.track_id for t in a] == [t.track_id for t in b]
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x.mean, y.mean)
            np.testing.assert_array_equal(x.covariance, y.covariance)
            np.testing.assert_array_equal(x.rgb_memory, y.rgb_memory)
            assert x.aux_memory is None and x.fused_memory is None
    fused_capture.close()
    rgb_capture.close()


def test_direct_rgb_capture_and_invalid_geometry_are_explicit():
    capture = FusionFeatureCapture(None, feature_source='rgb', quality_policy='uniform')
    with pytest.raises(ValueError):
        capture.begin_frame('synthetic', 1, (32, 48), (64, 96), 1.)
    arm(capture)
    capture.capture_rgb_only('synthetic', 1, planes()[0])
    assert capture.take_packet('synthetic', 1, detections())['rgb'].shape == (1, 2)
    capture.close()
