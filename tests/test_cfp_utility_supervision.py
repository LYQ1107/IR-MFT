import numpy as np
import pytest
import torch

from cfp.utility import counterfactual_utility
from cfp.utility_supervision import ranking_utility,ranking_sign_loss,decision_strata


def test_ranking_target_removes_null_penalty_but_keeps_identity_order_information():
    rgb=np.array([.1,.3]);ir=np.array([.3,.1])
    assert ranking_utility(rgb,ir,0)<0 and ranking_utility(rgb,ir,1)>0
    assert ranking_utility(rgb,ir,2)==0
    assert counterfactual_utility(rgb,ir,2)!=0  # Prior complete-assignment NULL term differs.
    assert ranking_utility([.1],[.3],0)==0
    assert ranking_utility(rgb,rgb,0)==0
    with pytest.raises(ValueError):ranking_utility(rgb,ir,0,temperature=float('nan'))


def test_signed_supervision_discriminates_help_and_harm_and_handles_all_zero_targets():
    truth=torch.tensor([.1,-.1]);right=torch.tensor([.1,-.1],requires_grad=True);wrong=-right.detach()
    assert ranking_sign_loss(right,truth)<ranking_sign_loss(wrong,truth)
    loss=ranking_sign_loss(right,truth);loss.backward();assert torch.isfinite(right.grad).all()
    output=torch.zeros(3,requires_grad=True);loss=ranking_sign_loss(output,torch.zeros(3));loss.backward()
    assert loss==0 and torch.equal(output.grad,torch.zeros(3))


def test_sampling_strata_use_only_physical_decision_type_not_utility_sign():
    rows=[dict(rgb=np.array([.1,.3]),target=2,utility=1.),dict(rgb=np.array([.1]),target=0,utility=-1.),dict(rgb=np.array([.1,.3]),target=0,utility=.2),dict(rgb=np.array([.1,.3]),target=0,utility=-.2)]
    groups=decision_strata(rows,[np.arange(4)])
    assert [a.tolist() for a in groups[0]]==[[0],[1],[2,3]]
