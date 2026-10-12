import torch

from cfp.recoverable_identity import RecoverableIdentityState, reliability_supervision_loss
from cfp.predictive import LatentIdentityPredictor


def test_semantic_trust_has_context_gradients_without_changing_feature_coordinates():
    torch.manual_seed(4)
    model = RecoverableIdentityState(semantic=True)
    features = torch.randn(2, 8, 512)
    times = torch.arange(1, 9)[None].repeat(2, 1)
    valid = torch.ones(2, 8, dtype=torch.bool)
    # Open the zero-initialized decision projection for the gradient invariant.
    torch.nn.init.normal_(model.token_head[-1].weight, std=.01)
    prediction, available, logits, confidence = model.details(features, times, valid, torch.tensor([9, 9]))
    assert prediction.shape == (2, 512) and available.all()
    loss = reliability_supervision_loss(logits, torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]]).repeat(2, 1), valid)
    loss.backward()
    assert model.feature_encoder[0].weight.grad.norm() > 0
    assert model.context_encoder.layers[0].self_attn.in_proj_weight.grad.norm() > 0
    assert torch.allclose(prediction.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_capacity_controls_are_within_three_percent_and_empty_context_stays_unavailable():
    model = RecoverableIdentityState(semantic=True).eval()
    size = sum(p.numel() for p in model.parameters())
    for architecture, hidden in [('gru', 52), ('transformer', 36)]:
        control = LatentIdentityPredictor(512, architecture, hidden=hidden)
        assert abs(sum(p.numel() for p in control.parameters()) / size - 1) < .03
    with torch.inference_mode():
        prediction, available = model(torch.zeros(2, 8, 512), torch.zeros(2, 8, dtype=torch.long), torch.zeros(2, 8, dtype=torch.bool), torch.tensor([9, 9]))
    assert not available.any() and torch.equal(prediction, torch.zeros_like(prediction))
