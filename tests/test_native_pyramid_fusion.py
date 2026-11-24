"""Identity fallback, learnable native pixels, motion support and causal state."""
import pytest
import torch

from yolox.models.native_pyramid_fusion import NativePyramidFusion, warp_previous


def inputs(batch=1):
    torch.manual_seed(0)
    rgb = tuple(torch.randn(batch, c, n, n) for c, n in [(8, 8), (16, 4), (24, 2)])
    scores = tuple(torch.full((batch, 1, *v.shape[-2:]), .3) for v in rgb)
    return rgb, torch.randn(batch, 3, 64, 64), scores


def model(mode='quality', **kwargs):
    torch.manual_seed(0)
    return NativePyramidFusion(channels=(8, 16, 24), mode=mode, **kwargs)


def open_output(net):
    with torch.no_grad():
        for layer in net.output:
            layer.weight.fill_(.01)


@pytest.mark.parametrize('mode', ['quality', 'temporal'])
def test_zero_projection_is_exact_rgb(mode):
    rgb, cue, scores = inputs()
    net = model(mode)
    result, _, _ = net(rgb, cue, scores, sequence='A', frame=1)
    assert all(torch.equal(a, b) for a, b in zip(result, rgb))


def test_missing_cue_after_nonzero_learning_clears_state_and_is_exact_rgb():
    rgb, cue, scores = inputs()
    net = model('temporal')
    open_output(net)
    changed, state, _ = net(rgb, cue, scores, sequence='A', frame=1)
    assert any(not torch.equal(a, b) for a, b in zip(changed, rgb))
    result, state, info = net(rgb, None, scores, available=False, state=state, sequence='A', frame=2)
    assert state is None and info['state_cleared']
    assert all(torch.equal(a, b) for a, b in zip(result, rgb))


def test_native_encoder_receives_gradient_after_projection_warm_start():
    rgb, cue, scores = inputs()
    net = model()
    optimizer = torch.optim.SGD(net.parameters(), lr=.1)
    snapshot = [v.clone() for v in rgb]
    gradients = []
    for _ in range(2):
        result, _, _ = net(rgb, cue, scores)
        loss = sum((a - (b + .1)).square().mean() for a, b in zip(result, rgb))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients.append(float(net.encoder.blocks[0][0].weight.grad.abs().sum()))
        optimizer.step()
    assert gradients[0] == 0 and gradients[1] > 0
    assert all(torch.equal(a, b) for a, b in zip(rgb, snapshot))


def test_motion_direction_resolution_scaling_and_invalid_border():
    value = torch.arange(4).float().reshape(1, 1, 1, 4).expand(1, 1, 2, 4)
    flow = torch.zeros(1, 2, 4, 8)
    flow[:, 0] = 2  # Two pixels at twice the feature resolution is one feature cell.
    result, support = warp_previous(value, flow)
    torch.testing.assert_close(result[0, 0, 0], torch.tensor([1., 2., 3., 0.]))
    assert support[..., :3].all() and not support[..., 3:].any()


def test_temporal_state_requires_consecutive_frames_and_explicit_sequence_reset():
    rgb, cue, scores = inputs()
    net = model('temporal')
    _, state, _ = net(rgb, cue, scores, sequence='A', frame=1)
    for sequence, frame in [('B', 2), ('A', 3)]:
        with pytest.raises(ValueError, match='Reset temporal state'):
            net(rgb, cue, scores, state=state, sequence=sequence, frame=frame)
    with pytest.raises(ValueError, match='current-to-previous'):
        net(rgb, cue, scores, state=state, sequence='A', frame=2)


def test_future_cue_cannot_mutate_previous_output_or_state():
    rgb, cue, scores = inputs()
    net = model('temporal').eval()
    open_output(net)
    zero_flow = torch.zeros(1, 2, 64, 64)
    first, state1, _ = net(rgb, cue, scores, sequence='A', frame=1)
    first_copy = [v.detach().clone() for v in first]
    state_copy = [v.detach().clone() for v in state1.latent]
    second, state2, info = net(rgb, cue + .5, scores, state=state1, sequence='A', frame=2,
                               flow_to_previous=zero_flow)
    second_copy = [v.detach().clone() for v in second]
    for future_cue in [cue - 10, cue + 10]:
        net(rgb, future_cue, scores, state=state2, sequence='A', frame=3, flow_to_previous=zero_flow)
    assert all(torch.equal(a, b) for a, b in zip(first, first_copy))
    assert all(torch.equal(a, b) for a, b in zip(second, second_copy))
    assert all(torch.equal(a, b) for a, b in zip(state1.latent, state_copy))
    assert any(not torch.equal(a, b) for a, b in zip(second, rgb))
    for level in info['levels']:
        assert ((level['gain'] >= 0) & (level['gain'] <= 1)).all()
        assert torch.isfinite(level['covariance']).all() and (level['covariance'] >= 0).all()


def test_warp_without_past_support_falls_back_to_current_observation():
    rgb, cue, scores = inputs()
    temporal, current_only = model('temporal'), model('temporal', use_memory=False)
    open_output(temporal)
    current_only.load_state_dict(temporal.state_dict(), strict=True)
    _, state, _ = temporal(rgb, cue, scores, sequence='A', frame=1)
    invalid_flow = torch.full((1, 2, 64, 64), 1000.)
    actual, _, _ = temporal(rgb, cue + .5, scores, state=state, sequence='A', frame=2,
                            flow_to_previous=invalid_flow)
    expected, _, _ = current_only(rgb, cue + .5, scores, sequence='A', frame=2)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def test_no_valid_image_area_cannot_create_a_feature_correction():
    rgb, cue, scores = inputs()
    net = model()
    with torch.no_grad():
        for layer in net.output:
            layer.bias.fill_(1.)
    output, _, _ = net(rgb, cue, scores, valid=torch.zeros_like(cue[:, :1], dtype=torch.bool))
    assert all(torch.equal(a, b) for a, b in zip(output, rgb))
