import pytest
import torch
from torch.nn import functional as F

from cfp.predictive import LatentIdentityPredictor,bounded_identity_prototype,latent_identity_loss


def inputs():
    features=F.normalize(torch.randn(2,4,4),dim=-1)
    frames=torch.tensor([[1,3,5,0],[0,0,0,0]])
    valid=frames>0
    return features,frames,valid,torch.tensor([6,6])


@pytest.mark.parametrize('architecture',['gru','transformer'])
def test_initial_predictor_is_last_feature_empty_history_unavailable_and_no_encoder_gradients(architecture):
    features,frames,valid,query=inputs();features.requires_grad_()
    model=LatentIdentityPredictor(4,architecture=architecture)
    predicted,available=model(features,frames,valid,query)
    assert available.tolist()==[True,False]
    assert torch.allclose(predicted[0],features[0,2]) and torch.equal(predicted[1],torch.zeros(4))
    reference,_=bounded_identity_prototype(features,frames,valid,query,'last')
    assert torch.equal(predicted,reference)
    predicted.sum().backward()
    assert features.grad is None and model.output.weight.grad is not None
    bad=frames.clone();bad[0,2]=6
    with pytest.raises(ValueError,match='strictly before'):model(features,bad,valid,query)
    bad=frames.clone();bad[0,1]=1
    with pytest.raises(ValueError,match='increase'):model(features,bad,valid,query)


def test_history_controls_use_identical_past_window_and_reject_noncontiguous_padding():
    features,frames,valid,query=inputs()
    for mode in ['last','ema','fifo']:
        result,available=bounded_identity_prototype(features,frames,valid,query,mode)
        assert torch.isfinite(result).all() and torch.equal(result[1],torch.zeros(4))
    valid[0]=torch.tensor([True,False,True,False])
    with pytest.raises(ValueError,match='contiguous'):bounded_identity_prototype(features,frames,valid,query)


def test_stopped_current_targets_with_hard_negative_penalize_constant_identity_predictions():
    predicted=F.normalize(torch.ones(2,4),dim=-1).requires_grad_()
    target=torch.eye(4)[:2].requires_grad_()
    candidates=torch.stack([torch.eye(4)[:2],torch.eye(4)[:2]]).requires_grad_()
    mask=torch.ones(2,2,dtype=torch.bool);positive=torch.tensor([0,1])
    constant_loss,_=latent_identity_loss(predicted,target,candidates,positive,mask)
    exact_loss,_=latent_identity_loss(target.detach(),target,candidates,positive,mask)
    assert constant_loss>exact_loss
    constant_loss.backward()
    assert predicted.grad is not None and target.grad is None and candidates.grad is None
    assert predicted.grad.norm()>0


def test_temporal_architectures_have_matched_parameter_budget():
    counts=[sum(p.numel() for p in LatentIdentityPredictor(512,a).parameters()) for a in ['gru','transformer']]
    assert max(counts)/min(counts)<1.05
