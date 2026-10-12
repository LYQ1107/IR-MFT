import torch

from cfp.candidate_jepa_controls import CandidateJEPAControl


def test_recomputed_evidence_cannot_recover_hidden_raw_or_cached_history():
    torch.manual_seed(11)
    model = CandidateJEPAControl(recoverability_evidence=True).eval()
    h = torch.randn(2, 8, 512); times = torch.zeros(2, 8, dtype=torch.long); times[:, :2] = torch.tensor([1, 2])
    valid = times > 0; frame = torch.tensor([4, 4]); candidates = torch.randn(2, 3, 512)
    legal = torch.ones(2, 3, dtype=torch.bool); evidence = torch.randn(2, 3, 21)
    a = model(h, times, valid, frame, candidates, legal, evidence)['scores']
    h[~valid] = 1000; evidence[..., :9] = -1000; evidence[..., 19:] = 1000
    b = model(h, times, valid, frame, candidates, legal, evidence)['scores']
    assert torch.allclose(a, b, atol=1e-6)
    v = valid.clone(); v[:] = False; times[:] = 0; legal[:] = False
    scores = model(h, times, v, frame, candidates, legal, evidence)['scores']
    assert torch.isfinite(scores[:, -1]).all()


def test_explicit_no_null_uses_same_candidate_head_with_disabled_null_score():
    torch.manual_seed(12)
    model = CandidateJEPAControl(use_null=False)
    h = torch.randn(1, 8, 512); t = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]])
    result = model(h, t, t > 0, torch.tensor([2]), torch.randn(1, 2, 512),
                   torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2, 21))
    assert result['scores'][0, -1] == -1e4 and torch.isfinite(result['scores'][0, :2]).all()
