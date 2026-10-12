import copy

import pytest
import torch

from cfp.candidate_jepa import CandidateJEPAStudent, CandidateEMATarget
from cfp.recoverability import FEATURE_NAMES


def inputs(m=3):
    return [torch.randn(2, 8, 16), torch.arange(1, 9)[None].repeat(2, 1),
        torch.ones(2, 8, dtype=torch.bool), torch.full((2,), 10),
        torch.randn(2, m, 16), torch.ones(2, m, dtype=torch.bool), torch.randn(2, m, len(FEATURE_NAMES))]


@pytest.mark.parametrize('architecture', ['cross_attention', 'gru', 'mlp'])
def test_candidate_conditioned_and_empty_null(architecture):
    model = CandidateJEPAStudent(16, 16, architecture).eval()
    for m in (0, 1, 3):
        result = model(*inputs(m))
        assert result['scores'].shape == (2, m + 1)
        assert torch.isfinite(result['scores']).all()


def test_candidate_order_and_masked_history_no_information_leak():
    model = CandidateJEPAStudent(16, 16).eval()
    x = inputs(); x[2][:, 4:] = False; x[1][:, 4:] = 0
    first = model(*x)
    altered = [v.clone() for v in x]; altered[0][:, 4:] = 999
    torch.testing.assert_close(model(*altered)['scores'], first['scores'])
    altered[6][..., :9] = 999
    altered[6][..., 19:] = 999
    torch.testing.assert_close(model(*altered)['scores'], first['scores'])
    permutation = torch.tensor([2, 0, 1])
    changed = [v.clone() for v in x]
    changed[4], changed[5], changed[6] = x[4][:, permutation], x[5][:, permutation], x[6][:, permutation]
    torch.testing.assert_close(model(*changed)['scores'][:, :-1], first['scores'][:, permutation])


def test_teacher_stopped_and_ema_only_successful_step():
    model = CandidateJEPAStudent(16, 16)
    target = CandidateEMATarget(model, .5)
    before = copy.deepcopy(target.state_dict())
    target.update(model, False)
    assert target.updates == 0
    with torch.no_grad():
        next(model.encoder.parameters()).add_(1)
    target.update(model, True)
    assert target.updates == 1
    assert not torch.equal(next(iter(before.values())), next(iter(target.state_dict().values())))
    assert not target(torch.randn(2, 16)).requires_grad
    assert all(p.grad is None and not p.requires_grad for p in target.parameters())


def test_future_rejected_and_empty_history_finite():
    model = CandidateJEPAStudent(16, 16)
    x = inputs()
    with pytest.raises(ValueError, match='strictly before'):
        model(*[x[0], x[1], x[2], torch.full((2,), 8), *x[4:]])
    x[2].fill_(False); x[1].zero_()
    result = model(*x)
    assert not result['history_available'].any() and torch.isfinite(result['scores']).all()

