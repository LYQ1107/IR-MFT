import numpy as np
import torch
from cfp.centered_joint_identity import centered_joint_cost,CenteredJointIdentity
from cfp.centered_joint_training import conditional_identity_ranking_loss,balanced_identity_loss
from cfp.joint_identity_supervision import joint_identity_targets


def test_uniform_cost_edit_disappears_exactly_including_single_and_empty_columns():
    base=torch.rand(4,3);legal=torch.tensor([[1,0,1],[1,0,0],[1,0,0],[0,0,0]],dtype=torch.bool)
    for offset in [0.,.3,30.,-30.]:
        assert torch.equal(centered_joint_cost(base,torch.full_like(base,offset),legal),base)


def test_candidate_common_raw_offsets_do_not_change_ranking_edits():
    torch.manual_seed(5);base=torch.rand(5,3);residual=torch.randn(5,3,dtype=torch.float64);base=base.double();legal=torch.rand(5,3)>.2
    a=centered_joint_cost(base,residual,legal);b=centered_joint_cost(base,residual+torch.tensor([2.,-7.,9.]),legal)
    torch.testing.assert_close(a,b,atol=1e-15,rtol=1e-14)
    for j in range(3):
        if legal[:,j].any():assert abs(float((a-base)[legal[:,j],j].mean()))<1e-15
    assert (a-base).abs().max()<=.03
    assert torch.equal(a[~legal],base[~legal])


def test_centered_ranking_has_real_relative_gradient_and_no_uniform_bias_gradient():
    residual=torch.tensor([[.2,-.1],[-.4,.6],[.3,-.2]],requires_grad=True);base=torch.rand(3,2);legal=torch.ones_like(base,dtype=torch.bool)
    costs=centered_joint_cost(base,residual,legal);loss=costs[0,0]+costs[1,1];loss.backward()
    assert residual.grad.abs().sum()>0
    torch.testing.assert_close(residual.grad.sum(0),torch.zeros(2),atol=1e-7,rtol=0)


def test_ranking_ignores_null_and_unknown_without_fabricated_negatives():
    target=joint_identity_targets('video',torch.tensor([7,8,-1]),torch.tensor([7,9,-1]));cost=torch.tensor([[.3,.2,.1],[.6,.4,.5],[.2,.9,.8]],requires_grad=True);legal=torch.ones_like(cost,dtype=torch.bool)
    loss,parts=conditional_identity_ranking_loss(cost,target,legal);loss.backward()
    assert parts['existing']==1 and parts['null']==0 and parts['supervised_detections']==1
    assert torch.equal(cost.grad[2],torch.zeros(3))
    assert torch.equal(cost.grad[:,1:],torch.zeros(3,2))


def test_null_loss_keeps_correct_private_unknown_semantics_and_gradient():
    supervision=joint_identity_targets('video',torch.tensor([1,2]),torch.tensor([1,3,-1]));logits=torch.tensor([[.1,.4,-.2],[.6,.4,-.2],[.3,.2,.1]],requires_grad=True)
    loss,parts=balanced_identity_loss(logits,supervision,torch.ones(2,3,dtype=torch.bool));loss.backward()
    assert parts['existing']==1 and parts['null']==1
    assert logits.grad[1,-1]<0 and torch.equal(logits.grad[2],torch.zeros(3))


def test_empty_old_set_is_trainable_null_but_not_ranking():
    s=joint_identity_targets('video',torch.empty(0,dtype=torch.int64),torch.tensor([3,-1]));legal=torch.empty(0,2,dtype=torch.bool);logits=torch.zeros(2,1,requires_grad=True)
    loss,parts=balanced_identity_loss(logits,s,legal);assert torch.isfinite(loss) and parts['null']==1
    rank,p=conditional_identity_ranking_loss(torch.empty(0,2,requires_grad=True),s,legal);assert torch.isfinite(rank) and p['supervised_detections']==0


def test_registered_cross_transformer_is_explicitly_distinct_control():
    cross=CenteredJointIdentity(8,8,2,8,'cross_transformer');joint=CenteredJointIdentity(8,8,2,8,'joint');pair=CenteredJointIdentity(8,8,2,8,'pair')
    assert not cross.set_self_attention
    assert cross.track_competition and cross.detection_competition
    assert joint.track_competition and joint.detection_competition
    assert pair.architecture=='pair'
    assert sum(p.numel() for p in cross.parameters())==sum(p.numel() for p in joint.parameters())


def test_cross_parameter_matched_feedforward_does_not_mix_tokens_and_has_real_gradients():
    from cfp.centered_joint_identity import TokenFeedForward
    torch.manual_seed(9);block=TokenFeedForward(8);x=torch.randn(1,4,8,requires_grad=True)
    a=block(x,x,x)[0];changed=x.detach().clone();changed[:,1]+=100
    b=block(changed,changed,changed)[0];assert torch.equal(a[:,0],b[:,0])
    a.sum().backward();assert all(p.grad is not None and p.grad.abs().sum()>0 for p in block.parameters())
    attention=torch.nn.MultiheadAttention(8,2,batch_first=True)
    assert sum(p.numel() for p in block.parameters())==sum(p.numel() for p in attention.parameters())
