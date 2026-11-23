import torch

from yolox.models.yolo_head import YOLOXHead
from yolox.models.frozen_head_repair import head_predictions,original_supervised_loss,preserve_reference


def setup_head():
    torch.manual_seed(2)
    head=YOLOXHead(1,width=.25,in_channels=[32,64,128]).eval().requires_grad_(False)
    features=[torch.randn(2,c,h,w)for c,h,w in [(8,8,12),(16,4,6),(32,2,3)]]
    return head,features


def test_manual_outputs_match_original_frozen_head_evaluation():
    head,features=setup_head()
    expected=head(features)
    raw,metadata=head_predictions(head,features)
    actual=torch.cat([raw[:,:,:4],raw[:,:,4:].sigmoid()],-1)
    torch.testing.assert_close(actual[:,:,:4],expected[:,:,:4],rtol=0,atol=0)
    # Original eval applies sigmoid before concatenation; this training helper
    # preserves logits and applies it afterward. SIMD rounding can differ1ulp.
    torch.testing.assert_close(actual[:,:,4:],expected[:,:,4:],rtol=0,atol=1e-7)
    assert preserve_reference(raw,raw,metadata['strides']).item()==0


def test_original_supervision_backpropagates_through_frozen_weights():
    head,features=setup_head()
    features=[x.requires_grad_()for x in features]
    predictions,metadata=head_predictions(head,features)
    targets=torch.zeros(2,5,5);targets[:,0]=torch.tensor([0.,32.,24.,20.,15.])
    loss=original_supervised_loss(head,predictions,metadata,targets,(64,96))[0]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(x.grad is not None and torch.isfinite(x.grad).all()for x in features)
    assert all(parameter.grad is None for parameter in head.parameters())


def test_wrong_cue_probability_and_geometry_changes_receive_a_gradient():
    reference=torch.tensor([[[8.,8.,12.,10.,2.,2.],[24.,8.,12.,10.,-2.,1.]]])
    changed=reference.clone();changed[:,:,0]+=2;changed[:,:,4]+=.5;changed.requires_grad_()
    loss=preserve_reference(changed,reference,[torch.tensor([[8.,8.]])])
    loss.backward()
    assert loss.item()>0
    assert torch.isfinite(changed.grad).all()
    assert changed.grad[:,:,4].abs().sum()>0
