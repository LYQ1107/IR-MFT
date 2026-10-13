import torch

from cfp.candidate_relation_residual import bounded_candidate_delta, centered_prediction_loss, residual_scores


def test_bounded_delta_centers_real_candidates_and_preserves_outside_region():
    logits = torch.tensor([[10., -10., 1000.]])
    legal = torch.tensor([[True, True, False]])
    delta = bounded_candidate_delta(logits, legal)
    assert delta.abs().max() <= .05 and delta[0, 2] == 0
    prior = torch.tensor([[.2, .99, 1.]])
    eligible = torch.tensor([[True, False, False]])
    result = residual_scores(prior, delta, eligible, legal)
    assert torch.allclose(result[0, 1], -prior[0, 1] / .1)
    assert result[0, -1] == -.97 / .1


def test_prediction_centering_removes_shared_candidate_bias_and_stops_target():
    p = torch.randn(2, 3, 16, requires_grad=True); t = torch.randn(2, 3, 16, requires_grad=True)
    legal = torch.tensor([[True, True, False], [True, True, True]])
    valid = torch.tensor([True, True])
    a = centered_prediction_loss(p, t, legal, valid)
    b = centered_prediction_loss(p + torch.randn(2, 1, 16), t + torch.randn(2, 1, 16), legal, valid)
    assert torch.allclose(a, b, atol=1e-6)
    a.backward(); assert t.grad is None and torch.isfinite(p.grad).all()
