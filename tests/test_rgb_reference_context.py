import numpy as np

from yolox.tracker.rgb_reference_context import extend_rgb_context


def test_current_candidate_competition_preserves_ties_and_has_no_future_group_effect():
    base=np.zeros((6,23),np.float32);p=np.array([.8,.8,.2,.6,.9,.1],np.float32);groups=np.array([0,0,0,1,2,2])
    result=extend_rgb_context(base,p,groups)
    assert result.shape==(6,26)
    assert result[0,-1]==0 and result[1,-1]==0
    assert result[3,-1]==p[3]
    np.testing.assert_allclose(result[4,-1],.8,atol=1e-6)
    p[4:]=0
    changed=extend_rgb_context(base,p,groups)
    np.testing.assert_array_equal(result[:4],changed[:4])
