import torch

from yolox.models.qra_fusion import (
    DynamicQualityFusion,
    QualitySpatialFusion,
    SpatialComplementarityFusion,
    TemporalComplementarityFusion,
)
from yolox.models.yolo_pafpn_rgbt import YOLOPAFPNRGBT


def _check_module(module):
    torch.manual_seed(7)
    rgb = torch.randn(2, 16, 8, 10, requires_grad=True)
    aux = torch.randn(2, 16, 8, 10, requires_grad=True)
    fused, info = module(rgb, aux, torch.tensor(-2.944439))
    assert fused.shape == rgb.shape
    for key in ("rgb_weight", "auxiliary_weight", "quality_stats", "redundancy", "effective_weight", "gate_range"):
        assert key in info
    assert torch.isfinite(fused).all()
    fused.mean().backward()
    assert rgb.grad is not None and aux.grad is not None


def test_qra_shapes_and_gradients():
    for module in (
        DynamicQualityFusion(16),
        SpatialComplementarityFusion(16),
        QualitySpatialFusion(16),
        QualitySpatialFusion(16, use_redundancy=True),
        TemporalComplementarityFusion(16),
    ):
        _check_module(module)


def test_zero_correction_matches_static_gate():
    torch.manual_seed(11)
    rgb = torch.randn(1, 16, 5, 7)
    aux = torch.randn(1, 16, 5, 7)
    static = torch.tensor(-2.944439)
    expected = torch.sigmoid(static)
    for module in (
        DynamicQualityFusion(16),
        SpatialComplementarityFusion(16),
        QualitySpatialFusion(16),
        TemporalComplementarityFusion(16),
    ):
        module.eval()
        fused, info = module(rgb, aux, static)
        assert torch.allclose(info["effective_weight"], expected, atol=1e-6)
        assert torch.allclose(fused, rgb + expected * (aux - rgb), atol=1e-6)


def test_temporal_state_is_causal_and_resettable():
    torch.manual_seed(13)
    a = TemporalComplementarityFusion(16)
    b = TemporalComplementarityFusion(16)
    b.load_state_dict(a.state_dict())
    a.set_stateful(True)
    b.set_stateful(True)
    rgb = torch.randn(1, 16, 4, 4)
    aux = torch.randn(1, 16, 4, 4)
    out_a, _ = a(rgb, aux, torch.tensor(-2.944439))
    out_b, _ = b(rgb, aux, torch.tensor(-2.944439))
    assert torch.allclose(out_a, out_b, atol=1e-6)
    a.reset_state()
    b.reset_state()
    changed = aux + 100.0
    prefix_a, _ = a(rgb, aux, torch.tensor(-2.944439))
    prefix_b, _ = b(rgb, changed, torch.tensor(-2.944439))
    # The module is causal in time, but a changed current observation is
    # allowed to change the current output.  After reset, the two streams
    # therefore differ only at this current frame, never through a future read.
    assert not torch.allclose(prefix_a, prefix_b)


def test_dynamic_initialization_is_v3_equivalent_on_fixed_input():
    torch.manual_seed(17)
    reference = YOLOPAFPNRGBT(depth=0.33, width=0.25, fusion_mode="gated_residual")
    candidate = YOLOPAFPNRGBT(depth=0.33, width=0.25, fusion_mode="dynamic_quality")
    target = candidate.state_dict()
    for key, value in reference.state_dict().items():
        if key in target and target[key].shape == value.shape:
            target[key] = value
    candidate.load_state_dict(target, strict=True)
    reference.eval()
    candidate.eval()
    input_tensor = torch.randn(1, 6, 128, 128)
    with torch.no_grad():
        reference_outputs = reference(input_tensor)
        candidate_outputs = candidate(input_tensor)
    for reference_output, candidate_output in zip(reference_outputs, candidate_outputs):
        assert torch.allclose(reference_output, candidate_output, atol=1e-6, rtol=1e-6)
