import numpy as np
import torch

from yolox.models.frozen_head_repair import preserve_reference
from yolox.models.recoverable_curriculum import (
    recovery_draw,weaken_recoverable_features,valid_counterfactual_locations,preserve_valid_reference,
)


def test_curriculum_is_shared_deterministic_and_cannot_degrade_ineligible_instances():
    targets=torch.tensor([[[0.,4.,4.,2.,2.],[0.,12.,12.,2.,2.]]]*4)
    eligible=np.array([[True,False]]*4)
    a=recovery_draw(targets,eligible,1,2,3,1.,1.)
    b=recovery_draw(targets,eligible,1,2,3,1.,1.)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a,eligible)
    feature=torch.zeros(4,2,16,16);feature[:,:,3:5,3:5]=8;original=feature.clone()
    output=weaken_recoverable_features([feature],targets,a,[16,16],strength=.75,kernel=9)[0]
    torch.testing.assert_close(feature,original,rtol=0,atol=0)
    assert bool((output[:,:,3:5,3:5]<feature[:,:,3:5,3:5]).all())
    outside=torch.ones_like(feature,dtype=torch.bool);outside[:,:,3:5,3:5]=False
    torch.testing.assert_close(output[outside],feature[outside],rtol=0,atol=0)
    assert weaken_recoverable_features([feature],targets,np.zeros_like(a),[16,16])[0]is feature


def test_counterfactual_excludes_shared_fish_class_but_keeps_nonoverlapping_false_evidence():
    metadata=dict(x_shifts=[torch.tensor([[0.,1.,2.]])],y_shifts=[torch.zeros(1,3)],strides=[torch.ones(1,3)*8])
    targets=torch.tensor([[[0.,8.,4.,16.,8.]],[[0.,16.,4.,16.,8.]]])
    valid=valid_counterfactual_locations(targets,torch.tensor([1,0]),metadata)
    torch.testing.assert_close(valid,torch.tensor([[True,False,True],[True,False,True]]))


def test_valid_preservation_is_original_loss_when_all_valid_and_zero_when_excluded():
    torch.manual_seed(19);reference=torch.randn(2,7,6);prediction=reference+torch.randn_like(reference)*.2
    strides=[torch.ones(1,7)*8];all_valid=torch.ones(2,7,dtype=torch.bool)
    ordinary=preserve_reference(prediction,reference,strides)
    torch.testing.assert_close(preserve_valid_reference(prediction,reference,strides,all_valid),ordinary)
    assert preserve_valid_reference(prediction,reference,strides,~all_valid).item()==0
    some=all_valid.clone();some[:,:4]=False
    assert preserve_valid_reference(prediction,reference,strides,some)<=ordinary
