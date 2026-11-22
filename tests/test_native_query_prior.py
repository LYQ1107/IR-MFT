import numpy as np

from yolox.tracker.native_query_prior import query_prior_objective


def test_real_null_prior_gradient_with_a_deleted_target_matches_finite_difference():
    group=dict(core_logits=np.array([1.,-1.,.5,-.5,2.,1.]),cosine=np.array([.4,.1,.2,.1,.5,.3]),query=np.array([0,0,1,1,1,2]),offsets=np.array([0,2,5,6]),
               valid=np.array([True,True,True,True,False,True]),label=np.array([1.,0.,0.,0.,0.,1.]),effective_counts=np.array([2,2,1]))
    theta=np.array([.7,2.,.5,-.2]);loss,gradient=query_prior_objective(theta,group)
    assert np.isfinite(loss)
    numerical=[]
    for i in range(4):
        delta=np.zeros(4);delta[i]=1e-5
        numerical.append((query_prior_objective(theta+delta,group)[0]-query_prior_objective(theta-delta,group)[0])/2e-5)
    np.testing.assert_allclose(gradient,numerical,rtol=1e-5,atol=1e-6)
