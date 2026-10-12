import torch

from cfp.recoverability_ablation import NoIdentityHistoryRisk


def test_no_identity_history_removes_derived_cost_margin_entropy_and_rival_leaks():
    model = NoIdentityHistoryRisk().eval()
    source = torch.randn(2, 3, 21)
    changed = source.clone()
    changed[..., :9] = 1000; changed[..., 19:] = 1000
    legal = torch.ones(2, 3, dtype=torch.bool)
    torch.testing.assert_close(model(source, legal), model(changed, legal))

