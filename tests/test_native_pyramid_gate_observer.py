"""Observational instrumentation must preserve outputs and absence semantics."""
import copy

import torch

from tools.native_pyramid_gate_observer import NativeGateObserver
from yolox.models.native_pyramid_fusion import NativePyramidFusion


def setup(mode):
    torch.manual_seed(0)
    net = NativePyramidFusion(channels=(8, 16, 24), mode=mode).eval().requires_grad_(False)
    with torch.no_grad():
        for layer in net.output: layer.weight.fill_(.01)
    rgb = tuple(torch.randn(1, c, h, h) for c, h in [(8, 8), (16, 4), (24, 2)])
    scores = tuple(torch.full_like(v[:, :1], .2) for v in rgb)
    cue = torch.randn(1, 3, 64, 64)
    return net, rgb, cue, scores


def test_gate_collection_preserves_exact_outputs_and_measures_real_correction():
    net, rgb, cue, scores = setup('quality')
    reference = copy.deepcopy(net)
    observer = NativeGateObserver(); observer.attach('cue', net)
    values, _, _ = net(rgb=rgb, cue=cue, scores=scores, valid=torch.ones(1, 1, 64, 64, dtype=torch.bool))
    expected, _, _ = reference(rgb, cue, scores)
    assert all(torch.equal(a, b) for a, b in zip(values, expected))
    data = observer.summary('cue')
    assert data['frames'] == data['available_frames'] == 1 and data['missing_frames'] == 0
    for row, cells in zip(data['levels'], [64, 16, 4]):
        assert row['valid_cells'] == cells
        assert row['quality_gate'] == dict(count=1, mean=.5, std=0., min=.5, max=.5)
        assert row['relative_feature_correction_RMS'] > 0 and row['nonzero_feature_correction_fraction'] > 0
        assert row['internal_covariance']['count'] == 0
    observer.close(); net(rgb, cue, scores)
    assert observer.summary('cue')['frames'] == 1


def test_missing_mature_temporal_state_has_no_invented_gate_or_covariance_sample():
    net, rgb, cue, scores = setup('temporal')
    observer = NativeGateObserver(); observer.attach('cue', net)
    _, state, _ = net(rgb, cue, scores, sequence='A', frame=1)
    snapshot = [v.clone() for v in state.latent]
    values, cleared, _ = net(rgb, None, scores, state=state, available=False, sequence='A', frame=2)
    assert cleared is None and all(torch.equal(a, b) for a, b in zip(values, rgb))
    assert all(torch.equal(a, b) for a, b in zip(state.latent, snapshot))
    data = observer.summary('cue')
    assert data['frames'] == 2 and data['available_frames'] == data['missing_frames'] == 1
    for row in data['levels']:
        assert row['quality_gate']['count'] == 1
        assert row['internal_covariance']['min'] > 0
    observer.close()


def test_invalid_canvas_cannot_inflate_feature_change_statistics():
    net, rgb, cue, scores = setup('quality')
    observer = NativeGateObserver(); observer.attach('cue', net)
    values, _, _ = net(rgb, cue, scores, valid=torch.zeros(1, 1, 64, 64, dtype=torch.bool))
    assert all(torch.equal(a, b) for a, b in zip(values, rgb))
    for row in observer.summary('cue')['levels']:
        assert row['valid_cells'] == row['reliable_teacher_cells'] == row['reliable_changed_feature_cells'] == 0
        assert row['nonzero_feature_correction_fraction'] is None
        assert row['feature_correction_RMS'] is None and row['relative_feature_correction_RMS'] is None
        assert row['temporal_gain']['count'] == 0
    observer.close()
