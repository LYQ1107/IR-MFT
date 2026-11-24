"""Complete nonduplicated visits, causal memory and real pixel input contracts."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.native_pyramid_fit import learning_rate_factor, predict_clip
from tools.native_pyramid_train_data import (
    PixelClip, TrainClip, current_to_previous_flow, detach_state,
    epoch_clips, preprocess_observation, schedule_proof,
)
from yolox.models.frozen_head_repair import head_predictions
from yolox.models.native_pyramid_fusion import NativePyramidFusion
from yolox.models.yolo_head import YOLOXHead


def test_full_registered_train_has_exact_chronological_visits_and_short_tails():
    lengths = {'BT-001': 3000, 'BT-003': 243, 'MSK-002': 1754,
               'PF-001': 15000, 'SN-001': 684, 'SN-015': 6570}
    clips = list(epoch_clips(lengths, 1))
    assert len(clips) == 6814
    assert Counter(len(c.frames) for c in clips) == {4: 6811, 3: 1, 2: 2}
    for sequence, n in lengths.items():
        assert [f for c in clips if c.sequence == sequence for f in c.frames] == list(range(1, n + 1))
    assert sum(len(c.frames) for c in clips) == 27251
    assert list(epoch_clips(lengths, 1)) == clips
    assert schedule_proof(lengths, 2)['video_order'] != schedule_proof(lengths, 1)['video_order']
    assert schedule_proof(lengths, 1)['repeated_frames'] == 0
    with pytest.raises(ValueError):
        list(epoch_clips(lengths, 1, seed=1))


def test_rgb_gray_pixels_do_not_require_an_ir_image_and_flat_ir_is_missing():
    rgb = np.random.default_rng(0).integers(0, 256, (32, 64, 3), dtype=np.uint8)
    stats = {'statistics': {c: dict(mean=.4, std=.2) for c in ['gray', 'ir']}}
    for cue in ['rgb', 'gray']:
        value, auxiliary, valid, gray, _, available = preprocess_observation(rgb, None, cue, (64, 64), stats)
        assert available and value.shape == auxiliary.shape == (3, 64, 64)
        assert valid[:, :32].all() and not valid[:, 32:].any()
        assert gray.shape == (16, 16)
    result = preprocess_observation(rgb, np.zeros((32, 64), np.uint8), 'ir', (64, 64), stats)
    assert result[-1] is False
    with pytest.raises(ValueError):
        preprocess_observation(rgb, np.zeros((33, 64), np.uint8), 'ir', (64, 64), stats)


def test_farneback_uses_current_to_previous_direction():
    rng = np.random.default_rng(0)
    previous = rng.integers(0, 256, (96, 96), dtype=np.uint8)
    current = np.roll(previous, 3, axis=1)
    flow = current_to_previous_flow(current, previous)
    assert float(flow[0, 0, 16:-16, 16:-16].median()) == pytest.approx(-3, abs=.2)
    assert current_to_previous_flow(current, None) is None


def small_case(frames=(1, 2), sequence='A'):
    torch.manual_seed(0)
    rgb = tuple(torch.randn(len(frames), c, h, h) for c, h in [(8, 8), (16, 4), (32, 2)])
    head = YOLOXHead(1, width=.125, in_channels=[64, 128, 256]).eval().requires_grad_(False)
    batch = PixelClip(TrainClip(sequence, frames), torch.zeros(len(frames), 3, 64, 64),
        torch.randn(len(frames), 3, 64, 64), torch.zeros(len(frames), 500, 5),
        torch.ones(len(frames), 1, 64, 64, dtype=torch.bool), tuple(True for _ in frames),
        tuple(np.zeros((16, 16), np.uint8) for _ in frames))
    _, meta = head_predictions(head, rgb)
    module = NativePyramidFusion(channels=(8, 16, 32), mode='temporal')
    with torch.no_grad():
        for layer in module.output:
            layer.weight.fill_(.01)
    return module, head, rgb, meta, batch


def test_temporal_forward_is_identical_across_optimizer_clip_boundaries():
    module, head, rgb, meta, batch = small_case()
    whole, final_state, _ = predict_clip(module, head, rgb, meta, batch)
    pieces, state, gray = [], None, None
    for index, frame in enumerate([1, 2]):
        part = SimpleNamespace(address=TrainClip('A', (frame,)), cue=batch.cue[index:index+1],
            valid=batch.valid[index:index+1], available=(True,), flow_gray=(batch.flow_gray[index],))
        pred, state, gray = predict_clip(module, head, tuple(v[index:index+1] for v in rgb),
            dict(scores=tuple(v[index:index+1] for v in meta['scores'])), part, state, gray)
        pieces.append(pred)
    # Frozen head batch arithmetic may have small CPU convolution roundoff.
    torch.testing.assert_close(torch.cat(pieces), whole, atol=2e-6, rtol=2e-6)
    assert all(torch.equal(a, b) for a, b in zip(state.latent, final_state.latent))
    assert state.frame == 2 and all(not v.requires_grad for v in state.latent + state.covariance)


def test_online_backward_reaches_current_native_pixels_without_past_gradient():
    module, head, rgb, meta, batch = small_case(frames=(1,))
    _, state, gray = predict_clip(module, head, rgb, meta, batch)
    snapshot = [v.clone() for v in state.latent]
    current = batch.cue.clone().requires_grad_()
    second = SimpleNamespace(address=TrainClip('A', (2,)), cue=current,
                             valid=batch.valid, available=(True,), flow_gray=batch.flow_gray)
    predictions, next_state, _ = predict_clip(module, head, rgb, meta, second, state, gray)
    predictions.square().mean().backward()
    assert current.grad is not None and float(current.grad.abs().sum()) > 0
    assert all(torch.equal(a, b) for a, b in zip(state.latent, snapshot))
    assert next_state.frame == 2 and all(not v.requires_grad for v in next_state.latent)
    assert all(p.grad is None for p in head.parameters())


def test_new_video_resets_past_state_before_recurrence():
    module, head, rgb, meta, batch = small_case(frames=(1,))
    _, state, gray = predict_clip(module, head, rgb, meta, batch)
    batch.address = TrainClip('B', (1,))
    reset, state_b, _ = predict_clip(module, head, rgb, meta, batch, state, gray)
    clean, state_clean, _ = predict_clip(module, head, rgb, meta, batch)
    assert torch.equal(reset, clean) and state_b.sequence == state_clean.sequence == 'B'
    cpu = detach_state(state_b, 'cpu')
    assert cpu.frame == 1 and all(not x.requires_grad for x in cpu.latent)


def test_learning_rate_uses_fixed_horizon_across_dev_extensions():
    horizon = 6814 * 12
    assert learning_rate_factor(0, horizon) == .002
    assert learning_rate_factor(499, horizon) == 1
    assert learning_rate_factor(500, horizon) == 1
    assert .03 < learning_rate_factor(6814 * 3, horizon) < 1
    assert learning_rate_factor(horizon - 1, horizon) == pytest.approx(.03, abs=1e-8)
    with pytest.raises(ValueError):
        learning_rate_factor(horizon, horizon)
