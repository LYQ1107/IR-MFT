import pytest
import torch

from cfp.recoverable_identity import RecoverableIdentityState, reliability_supervision_loss


def inputs():
    return (torch.tensor([[[1., 0.], [0., 1.], [0., 1.]]]), torch.tensor([[1, 2, 3]]), torch.ones(1, 3, dtype=torch.bool), torch.tensor([4]))


def test_initial_soft_state_can_disagree_with_first_observation():
    model = RecoverableIdentityState()
    prototype, available = model(*inputs())
    assert available.item() and prototype[0, 1] > prototype[0, 0]
    assert torch.allclose(prototype.norm(dim=-1), torch.ones(1))


def test_current_future_token_rejected_and_empty_state_explicitly_unavailable():
    model = RecoverableIdentityState()
    feature, frames, valid, query = inputs()
    with pytest.raises(ValueError):
        model(feature, frames, valid, torch.tensor([3]))
    prototype, available = model(feature, torch.zeros_like(frames), torch.zeros_like(valid), query)
    assert not available.any() and torch.equal(prototype, torch.zeros_like(prototype))


def test_unknown_token_labels_receive_no_supervision_gradient():
    logits = torch.tensor([[.1, -.2, .4]], requires_grad=True)
    target = torch.tensor([[True, False, False]])
    known = torch.tensor([[True, True, False]])
    reliability_supervision_loss(logits, target, known).backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] > 0 and logits.grad[0, 2] == 0

