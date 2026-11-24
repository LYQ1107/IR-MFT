"""True online source identity, missing-image I/O, state reset and interventions."""
import numpy as np
import pytest
import torch

from tools.native_pyramid_online_inference import OnlineNativeDetector
from yolox.models.native_pyramid_fusion import NativePyramidFusion
from yolox.models.yolo_head import YOLOXHead


class Source(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from test_native_pyramid_training_loop import TinyEncoder
        self.backbone = TinyEncoder()
        self.head = YOLOXHead(1, width=.125, in_channels=[64, 128, 256])


def setup(*, unavailable=False, nonzero=False):
    torch.manual_seed(0)
    source = Source().eval().requires_grad_(False)
    rows = [dict(id='RGB', cue='rgb', baseline=True)]
    modules = dict(RGB=None)
    for mode in ['quality', 'temporal']:
        for intervention in (['unavailable'] if unavailable else ['normal', 'IR_image_spatial_roll']):
            name = mode+'_'+intervention
            rows.append(dict(id=name, cue='ir', intervention=intervention, expect_RGB_exact=not nonzero))
            net = NativePyramidFusion(channels=(8, 16, 32), mode=mode).eval().requires_grad_(False)
            if nonzero:
                with torch.no_grad():
                    for layer in net.output: layer.weight.fill_(.01)
            modules[name] = net
    stats = {'statistics': {c: dict(mean=.4, std=.2) for c in ['gray', 'ir']}}
    return OnlineNativeDetector(source, rows, modules, (64, 64), stats)


def images():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (48, 64, 3), dtype=np.uint8), rng.integers(0, 256, (48, 64), dtype=np.uint8)


def test_zero_native_full_detector_nms_and_tracks_input_equal_fresh_rgb():
    engine = setup()
    rgb, ir = images()
    for frame in [1, 2]:
        values, info = engine.step(rgb, ir, 'A', frame)
        assert info['independent_input_batch_size'] == 1
        assert all(info['full_head_exact_to_fresh_RGB'].values())
        for value in values.values(): np.testing.assert_array_equal(value, values['RGB'])


def test_trained_unavailable_route_needs_no_ir_file_and_is_exact_rgb():
    engine = setup(unavailable=True, nonzero=True)
    assert not engine.needs_actual_IR_image
    rgb, _ = images()
    values, _ = engine.step(rgb, None, 'A', 1)
    for value in values.values(): np.testing.assert_array_equal(value, values['RGB'])
    assert all(state is None for state in engine.states.values())


def test_ir_roll_changes_current_cue_and_time_state_without_mutating_past_outputs():
    engine = setup(nonzero=True)
    rgb, ir = images()
    values, _ = engine.step(rgb, ir, 'A', 1)
    snapshot = {name: value.copy() for name, value in values.items()}
    normal = engine.states['temporal_normal']
    rolled = engine.states['temporal_IR_image_spatial_roll']
    assert any(not torch.equal(a, b) for a, b in zip(normal.latent, rolled.latent))
    past = [v.clone() for v in normal.latent]
    engine.step(rgb, 255-ir, 'A', 2)
    for name, value in values.items(): np.testing.assert_array_equal(value, snapshot[name])
    assert all(torch.equal(a, b) for a, b in zip(normal.latent, past))


def test_sequence_reset_matches_fresh_engine_and_rejects_skipped_frames():
    engine = setup(nonzero=True)
    fresh = setup(nonzero=True)
    rgb, ir = images()
    engine.step(rgb, ir, 'A', 1)
    engine.step(rgb, ir, 'A', 2)
    values, _ = engine.step(rgb, ir, 'B', 1)
    expected, _ = fresh.step(rgb, ir, 'B', 1)
    for name in values: np.testing.assert_array_equal(values[name], expected[name])
    with pytest.raises(ValueError, match='consecutive'):
        engine.step(rgb, ir, 'B', 3)
    with pytest.raises(ValueError, match='start at frame1'):
        engine.step(rgb, ir, 'C', 2)


def test_registered_actual_ir_cannot_silently_fall_back_when_file_is_missing():
    engine = setup()
    rgb, _ = images()
    with pytest.raises(ValueError, match='actual November IR'):
        engine.step(rgb, None, 'A', 1)


def test_flat_ir_observation_clears_mature_memory_and_emits_exact_rgb():
    engine = setup(nonzero=True)
    rgb, ir = images()
    engine.step(rgb, ir, 'A', 1)
    assert engine.states['temporal_normal'] is not None
    values, info = engine.step(rgb, np.zeros_like(ir), 'A', 2)
    for value in values.values(): np.testing.assert_array_equal(value, values['RGB'])
    assert all(info['full_head_exact_to_fresh_RGB'].values())
    assert all(state is None for state in engine.states.values())
    engine.step(rgb, ir, 'A', 3)
    assert engine.states['temporal_normal'].frame == 3
