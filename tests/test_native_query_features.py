import numpy as np
import torch

from yolox.tracker.association_calibrator import AssociationCalibrator
from yolox.tracker.native_query_features import competition_margins,identity_pair_features,posterior_with_null


def test_identity_contrast_cancels_exactly_for_equal_RGB_and_cue_inputs():
    rng=np.random.default_rng(2);past=rng.normal(size=(3,8)).astype(np.float32);current=rng.normal(size=(4,8)).astype(np.float32)
    extra,cue,reference=identity_pair_features(past,current,past.copy(),current.copy())
    np.testing.assert_array_equal(cue,reference)
    base=torch.randn(12,26);quality=np.ones((3,4,2),np.float32)
    packet=torch.from_numpy(np.concatenate([cue,quality],axis=-1).reshape(12,9))
    model=AssociationCalibrator(base_dimensions=26,cue_dimensions=9)
    with torch.no_grad():model.net[-1].weight.normal_()
    assert torch.all(model.increment(base,packet,packet,torch.ones(12,dtype=torch.bool))==0)
    assert extra.shape==(3,4,3)


def test_competition_keeps_tied_identity_alternatives_ambiguous():
    row,column=competition_margins(np.array([[.8,.8],[.8,.2]],np.float32))
    assert row[0,0]==0 and row[0,1]==0 and column[0,0]==0 and column[1,0]==0


def test_null_is_a_real_normalized_alternative_and_handles_no_current_observation():
    real=np.array([[-10.,-11.],[3.,0.]],np.float32);null=np.array([0.,-1.],np.float32)
    probability,absence=posterior_with_null(real,null)
    np.testing.assert_allclose(probability.sum(1)+absence,1,rtol=0,atol=1e-7)
    assert absence[0]>.999 and probability[1,0]>.9
    no_null,p0=posterior_with_null(real,null,include_null=False)
    assert np.all(p0==0);np.testing.assert_allclose(no_null.sum(1),1,atol=1e-7)
    empty,p0=posterior_with_null(np.empty((2,0),np.float32),null)
    assert empty.shape==(2,0) and np.all(p0==1)
