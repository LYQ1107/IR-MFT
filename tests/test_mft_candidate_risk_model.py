import numpy as np
from tools.mft_candidate_risk_features import FEATURE_NAMES,IR_FEATURES
from tools.mft_candidate_risk_model import RiskLogistic,NewtonPass,class_video_weights,sigmoid


def data():
    a=np.zeros((8,len(FEATURE_NAMES)),np.float64)
    a[:,0]=[-2,-1,-.5,-.25,.25,.5,1,2]
    a[:,1]=[1,-1,1,-1,1,-1,1,-1]
    return a,np.asarray([0,0,0,0,1,1,1,1],np.float64)


def test_streaming_gradient_hessian_and_loss_equal_materialized_objective():
    a,y=data();model=RiskLogistic(np.zeros(26),np.ones(26),'paired')
    model.coefficients[:3]=[.2,-.1,.05]
    weights=class_video_weights(y,4,4,1)
    streamed=NewtonPass(model);streamed.add(a[:3],y[:3],weights[:3]);streamed.add(a[3:],y[3:],weights[3:])
    dense=NewtonPass(model);dense.add(a,y,weights)
    for x,z in zip(streamed.objective(),dense.objective()):np.testing.assert_allclose(x,z,atol=1e-15)
    _,gradient,hessian=dense.objective();step=1e-5
    def objective(c):
        x=model.design(a);z=x@c
        return float(weights@(np.logaddexp(0,z)-y*z))+.0005*float(c[:-1]@c[:-1])
    for i in [0,1,26]:
        v=np.zeros(27);v[i]=step
        derivative=(objective(model.coefficients+v)-objective(model.coefficients-v))/(2*step)
        assert abs(derivative-gradient[i])<1e-8
    assert np.linalg.eigvalsh(hessian).min()>0


def test_six_fixed_newton_passes_improve_separable_supervision_and_roundtrip():
    a,y=data();model=RiskLogistic(np.zeros(26),np.ones(26),'paired');losses=[]
    for _ in range(6):
        update=NewtonPass(model);update.add(a,y,class_video_weights(y,4,4,1));losses.append(update.step()['weighted_objective_before_step'])
    assert all(b<a for a,b in zip(losses,losses[1:]))
    assert (model.probabilities(a[:4])<.5).all()and(model.probabilities(a[4:])>.5).all()
    np.testing.assert_array_equal(model.probabilities(a),RiskLogistic.from_dict(model.to_dict()).probabilities(a))


def test_rgb_fit_cannot_read_masked_ir_and_class_video_weights_sum_to_one():
    a,y=data();model=RiskLogistic(np.arange(26),np.ones(26),'RGB')
    baseline=model.design(a);a[:,IR_FEATURES]=777
    np.testing.assert_array_equal(baseline,model.design(a))
    weights=np.r_[class_video_weights(np.asarray([1,0,0]),1,2,2),class_video_weights(np.asarray([1,1]),2,0,2)]
    assert abs(weights.sum()-1)<1e-12 and abs(weights[:3].sum()-.5)<1e-12
    np.testing.assert_allclose(sigmoid(np.asarray([-1e4,0,1e4])),[0,.5,1])
