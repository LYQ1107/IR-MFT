import numpy as np
import pytest
from tools.mft_identity_anchor import IdentityAnchors


def test_anchor_needs_three_reliable_past_views_and_excludes_current_frame():
    a=IdentityAnchors()
    for f in [1,2]:
        a.add(1,f,np.asarray([1.,0.]),np.asarray([0.,1.]),reliable=True)
        assert a.get(1,f+1)is None
    a.add(1,3,np.asarray([-1.,0.]),np.asarray([0.,-1.]),reliable=False)
    assert a.get(1,4)is None
    a.add(1,4,np.asarray([1.,0.]),np.asarray([0.,1.]),reliable=True)
    with pytest.raises(AssertionError):a.get(1,4)
    got=a.get(1,5);np.testing.assert_array_equal(got['RGB'],[1.,0.]);assert got['last_frame']==4


def test_later_template_drift_and_mutable_caller_arrays_cannot_change_anchor():
    a=IdentityAnchors();r=np.asarray([1.,0.]);ir=np.asarray([0.,1.])
    for f in range(1,4):a.add(1,f,r,ir,reliable=True)
    before=a.get(1,4);r[:]=-1;ir[:]=-1
    for f in range(4,20):a.add(1,f,r,ir,reliable=True)
    after=a.get(1,20);np.testing.assert_array_equal(before['RGB'],after['RGB']);np.testing.assert_array_equal(before['IR'],after['IR'])
    after['RGB'][:]=77;np.testing.assert_array_equal(a.get(1,21)['RGB'],before['RGB'])
    assert after['last_frame']==3


def test_missing_IR_does_not_contaminate_RGB_and_sequence_pool_pruning_clears_history():
    a=IdentityAnchors();rgb=np.asarray([1.,0.]);ir=np.asarray([0.,1.])
    for f in range(1,4):a.add(1,f,rgb,None if f==2 else ir,reliable=True)
    assert a.get(1,4)['IR']is None;np.testing.assert_array_equal(a.get(1,4)['RGB'],rgb)
    a.prune(set());assert not a.entries and a.get(1,5)is None
