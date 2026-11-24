import pytest
import torch

from tools.spatial_native_fusion import SpatialNativeFusion
from yolox.models.native_pyramid_fusion import NativePyramidFusion


def sample(batch=1):
    torch.manual_seed(12)
    rgb = tuple(torch.randn(batch, c, 64//s, 96//s) for c, s in zip((8, 16, 32), (8, 16, 32)))
    cue = torch.randn(batch, 3, 64, 96)
    scores = tuple(torch.full_like(v[:, :1], .3) for v in rgb)
    valid = torch.ones_like(cue[:, :1], dtype=torch.bool)
    return rgb, cue, scores, valid


def changed(module):
    with torch.no_grad():
        for layer in module.output:
            layer.weight.normal_(0, .05)
            layer.bias.normal_(0, .05)
        for layer in module.comparison:
            layer.spatial[-1].weight.normal_(0, .05)


def test_initial_full_feature_identity_and_fixed_parameter_budget():
    rgb, cue, scores, valid = sample()
    counts = []
    for quality, redundancy in [(False, False), (True, False), (True, True)]:
        torch.manual_seed(0)
        module = SpatialNativeFusion(channels=(8, 16, 32), use_quality=quality, use_redundancy=redundancy)
        values, state, _ = module(rgb, cue, scores, valid=valid)
        assert state is None and all(torch.equal(a, b) for a, b in zip(values, rgb))
        counts.append(sum(p.numel() for p in module.parameters()))
    assert len(set(counts)) == 1
    assert sum(p.numel() for p in SpatialNativeFusion().parameters()) == 556681


def test_zero_spatial_delta_matches_learned_quality_parent():
    rgb, cue, scores, valid = sample()
    parent = NativePyramidFusion(channels=(8, 16, 32), mode='quality')
    with torch.no_grad():
        for layer in parent.output:
            layer.weight.normal_(0, .05)
    child = SpatialNativeFusion(channels=(8, 16, 32))
    missing = child.load_state_dict(parent.state_dict(), strict=False)
    assert not missing.unexpected_keys and all(n.startswith('comparison.') for n in missing.missing_keys)
    original = parent(rgb, cue, scores, valid=valid)[0]
    new = child(rgb, cue, scores, valid=valid)[0]
    assert all(torch.equal(a, b) for a, b in zip(original, new))


def test_disabling_new_gates_preserves_loaded_parent_after_training():
    rgb, cue, scores, valid = sample()
    child = SpatialNativeFusion(channels=(8, 16, 32), use_spatial=False)
    changed(child)
    parent = NativePyramidFusion(channels=(8, 16, 32), mode='quality')
    parent.load_state_dict({n: v for n, v in child.state_dict().items() if not n.startswith('comparison.')})
    assert all(torch.equal(a, b) for a, b in zip(parent(rgb, cue, scores, valid=valid)[0],
        child(rgb, cue, scores, valid=valid)[0]))


def test_redundancy_starts_near_disabled_and_gate_is_bounded():
    rgb, cue, scores, valid = sample()
    module = SpatialNativeFusion(channels=(8, 16, 32), use_redundancy=True)
    _, _, diagnostics = module(rgb, cue, scores, valid=valid)
    for row in diagnostics['levels']:
        assert torch.allclose(row['redundancy'], torch.full_like(row['redundancy'], torch.sigmoid(torch.tensor(-6.))))
        assert bool((row['gate'] > 0).all() & (row['gate'] < .5).all())
        assert not bool(row['spatial_delta'].any())


def test_loss_reaches_encoder_spatial_and_redundancy_context():
    rgb, cue, scores, valid = sample()
    module = SpatialNativeFusion(channels=(8, 16, 32), use_redundancy=True)
    optimizer = torch.optim.AdamW(module.parameters(), lr=.001)
    gradients = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        values = module(rgb, cue, scores, valid=valid)[0]
        loss = sum(v.square().mean() for v in values)
        loss.backward()
        gradients.append({key: sum(float(p.grad.abs().sum()) for n, p in module.named_parameters()
            if key in n and p.grad is not None) for key in ['encoder.blocks.0.0', 'output.',
                'spatial.4', 'redundancy.2', 'rgb_projection', 'aux_projection']})
        optimizer.step()
    assert gradients[0]['output.'] > 0 and gradients[0]['encoder.blocks.0.0'] == 0
    assert all(gradients[1][key] > 0 for key in ['encoder.blocks.0.0', 'spatial.4', 'redundancy.2'])
    assert all(gradients[2][key] > 0 for key in ['rgb_projection', 'aux_projection'])


def test_updated_missing_route_never_reads_cue_and_is_exact():
    class Unreadable:
        def __getattribute__(self, name):
            raise AssertionError('Missing cue was read')
    rgb, _, scores, _ = sample()
    module = SpatialNativeFusion(channels=(8, 16, 32), use_redundancy=True)
    changed(module)
    values, state, diagnostics = module(rgb, Unreadable(), scores, available=False)
    assert state is None and diagnostics['missing_exact_RGB']
    assert all(a is b for a, b in zip(values, rgb))


def test_partial_availability_and_invalid_padding_do_not_change_rgb():
    rgb, cue, scores, valid = sample(batch=2)
    module = SpatialNativeFusion(channels=(8, 16, 32))
    changed(module)
    valid[0, :, :, 64:] = False
    result, _, diagnostics = module(rgb, cue, scores, valid=valid, available=[True, False])
    assert all(torch.equal(a[1], b[1]) for a, b in zip(result, rgb))
    assert any(not torch.equal(a[0], b[0]) for a, b in zip(result, rgb))
    for a, b, row in zip(result, rgb, diagnostics['levels']):
        padding = ~row['valid'].expand_as(a)
        assert torch.equal(a[padding], b[padding])


def test_spatial_gate_varies_by_location_and_residual_stays_bounded():
    rgb, cue, scores, valid = sample()
    module = SpatialNativeFusion(channels=(8, 16, 32))
    changed(module)
    result, _, diagnostics = module(rgb, cue, scores, valid=valid)
    for value, base, row in zip(result, rgb, diagnostics['levels']):
        assert float(row['gate'].std()) > 1e-6
        limit = .5*base.square().mean(1, keepdim=True).add(1e-6).sqrt()*.7*row['gate']
        assert bool(((value-base).abs() <= limit+2e-7).all())


def test_current_frame_has_no_history_and_does_not_rewrite_outputs():
    rgb, cue, scores, valid = sample()
    module = SpatialNativeFusion(channels=(8, 16, 32))
    changed(module)
    result = tuple(v.detach().clone() for v in module(rgb, cue, scores, valid=valid)[0])
    module(rgb, cue+1, scores, valid=valid)
    repeated = module(rgb, cue, scores, valid=valid)[0]
    assert all(torch.equal(a, b) for a, b in zip(result, repeated))
    with pytest.raises(ValueError):
        module(rgb, cue, scores, state=object())
    with pytest.raises(ValueError):
        module(rgb, cue, scores, flow_to_previous=torch.zeros(1, 2, 64, 96))


def test_rejects_inconsistent_geometry_and_invalid_configuration():
    rgb, cue, scores, valid = sample()
    module = SpatialNativeFusion(channels=(8, 16, 32))
    with pytest.raises(ValueError):
        module(rgb, cue[:, :, :-8], scores)
    with pytest.raises(ValueError):
        module(rgb, cue, scores, valid=valid.float())
    with pytest.raises(ValueError):
        SpatialNativeFusion(hidden=3)
    with pytest.raises(ValueError):
        SpatialNativeFusion(bound=float('nan'))
