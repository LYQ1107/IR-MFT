import numpy as np
import torch

from yolox.tracker.native_cue_corruption import permuted_current_packet,visible_mismatch,preserve_RGB_posterior
from yolox.tracker.native_query_features import identity_pair_features


def test_mismatch_recomputes_cross_modal_agreement_at_actual_RGB_locations():
    reference=np.eye(2,dtype=np.float32)
    original=identity_pair_features(reference,reference,reference,reference)[1]
    actual=permuted_current_packet(reference,reference,reference,reference,[1,0])
    np.testing.assert_array_equal(actual[...,4],np.zeros((2,2)))
    assert not np.array_equal(actual,np.roll(original,1,axis=1))
    np.testing.assert_array_equal(actual[...,1],original[...,1])


def test_deleted_long_candidate_cannot_survive_via_a_mismatched_cue():
    reference=torch.randn(2,3,9);first=torch.randn_like(reference);second=first.clone();second[0]=100000
    valid=torch.tensor([[True,False,True],[True,True,True]])
    dropped=torch.tensor([True,False]);available=torch.ones_like(valid)
    a,enabled=visible_mismatch(first,reference,available,valid,dropped)
    b,other=visible_mismatch(second,reference,available,valid,dropped)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert not enabled[0].any()
    torch.testing.assert_close(a[0],reference[0],rtol=0,atol=0)


def test_preservation_includes_the_null_and_has_no_RGB_teacher_gradient():
    real=torch.zeros(2,3,requires_grad=True);null=torch.zeros(2,requires_grad=True)
    valid=torch.tensor([[True,True,False],[True,False,False]])
    zero=torch.zeros_like(real,requires_grad=True)
    torch.testing.assert_close(preserve_RGB_posterior(real,null,zero,valid),torch.tensor(0.),atol=1e-7,rtol=0)
    delta=torch.full_like(real,2.,requires_grad=True)
    loss=preserve_RGB_posterior(real,null,delta,valid)
    assert loss>0
    loss.backward()
    assert delta.grad[valid].abs().sum()>0
    assert not delta.grad[~valid].any()
    assert real.grad is None and null.grad is None
