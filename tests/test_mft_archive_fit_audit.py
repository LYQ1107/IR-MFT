import numpy as np
from tools.audit_mft_archive_fit import batch_diagnostic
from tools.mft_archive_reasoning import ArchiveLogistic,IR_FEATURES


def test_independent_gradient_matches_finite_differences_and_never_updates_model():
    model=ArchiveLogistic(np.zeros(26),np.ones(26),'paired')
    model.coefficients[:]=np.linspace(-.2,.3,27)
    x=np.sin(np.arange(104).reshape(4,26));y=np.asarray([1,0,1,0]);w=np.asarray([.1,.2,.3,.4])
    before=model.coefficients.copy();loss,gradient,weight,p=batch_diagnostic(model,x,y,w)
    numerical=[]
    for i in range(27):
        model.coefficients[i]+=1e-6;a=batch_diagnostic(model,x,y,w)[0]
        model.coefficients[i]-=2e-6;b=batch_diagnostic(model,x,y,w)[0]
        model.coefficients[:]=before;numerical.append((a-b)/2e-6)
    np.testing.assert_allclose(gradient,numerical,rtol=2e-8,atol=2e-10)
    np.testing.assert_array_equal(model.coefficients,before)
    assert weight==1 and np.isfinite(loss)and((p>0)&(p<1)).all()


def test_independent_RGB_numerical_path_ignores_arbitrary_IR_features():
    model=ArchiveLogistic(np.zeros(26),np.ones(26),'RGB');model.coefficients[:]=.1
    x=np.cos(np.arange(78).reshape(3,26));other=x.copy();other[:,IR_FEATURES]=123
    y=np.asarray([0,1,0]);w=np.asarray([.25,.5,.25]);a=batch_diagnostic(model,x,y,w);b=batch_diagnostic(model,other,y,w)
    assert a[0]==b[0]and a[2]==b[2]
    np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[3],b[3])
    np.testing.assert_array_equal(a[1][list(IR_FEATURES)],0)
