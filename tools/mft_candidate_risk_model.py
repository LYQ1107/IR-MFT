"""Small deterministic Train-only logistic association model; no torch fit."""
import numpy as np
from tools.mft_candidate_risk_features import FEATURE_NAMES,IR_FEATURES,rgb_only_mask


def sigmoid(values):
    a=np.asarray(values,np.float64);out=np.empty_like(a)
    positive=a>=0
    out[positive]=1/(1+np.exp(-a[positive]))
    e=np.exp(a[~positive]);out[~positive]=e/(1+e)
    return out


class RiskLogistic:
    def __init__(self,mean,scale,kind):
        assert kind in ['paired','RGB']
        self.kind=kind
        self.mean=np.asarray(mean,np.float64).copy();self.scale=np.asarray(scale,np.float64).copy()
        assert self.mean.shape==self.scale.shape==(len(FEATURE_NAMES),)
        assert np.isfinite(self.mean).all()and np.isfinite(self.scale).all()and(self.scale>0).all()
        if kind=='RGB':self.mean[list(IR_FEATURES)]=0.;self.scale[list(IR_FEATURES)]=1.
        self.coefficients=np.zeros(len(FEATURE_NAMES)+1,np.float64)

    def design(self,values):
        values=np.asarray(values,np.float64)
        assert values.ndim==2 and values.shape[1]==len(FEATURE_NAMES)and np.isfinite(values).all()
        if self.kind=='RGB':values=rgb_only_mask(values)
        x=np.empty((len(values),len(FEATURE_NAMES)+1),np.float64)
        x[:,:-1]=np.clip((values-self.mean)/self.scale,-8,8);x[:,-1]=1
        return x

    def probabilities(self,values):
        return sigmoid(self.design(values)@self.coefficients)

    def to_dict(self):
        return dict(kind=self.kind,feature_names=list(FEATURE_NAMES),mean=self.mean.tolist(),
            scale=self.scale.tolist(),coefficients=self.coefficients.tolist(),
            coefficient_dtype='float64',standardized_feature_clip=8.,random_seed=0)

    @classmethod
    def from_dict(cls,value):
        assert value['feature_names']==list(FEATURE_NAMES)and value['standardized_feature_clip']==8.
        assert value['random_seed']==0
        result=cls(value['mean'],value['scale'],value['kind'])
        coefficients=np.asarray(value['coefficients'],np.float64)
        assert coefficients.shape==result.coefficients.shape and np.isfinite(coefficients).all()
        result.coefficients=coefficients.copy()
        return result


class NewtonPass:
    """Accumulate the exact weighted objective in streaming frame batches."""
    def __init__(self,model,regularization=1e-3):
        assert regularization>0
        self.model,self.regularization=model,float(regularization)
        size=len(model.coefficients)
        self.gradient=np.zeros(size,np.float64);self.hessian=np.zeros((size,size),np.float64)
        self.loss=0.;self.weight=0.;self.rows=0

    def add(self,values,targets,weights):
        x=self.model.design(values);y=np.asarray(targets,np.float64);w=np.asarray(weights,np.float64)
        assert y.shape==w.shape==(len(x),)and np.isfinite(w).all()and(w>=0).all()
        assert np.logical_or(y==0,y==1).all()
        z=x@self.model.coefficients;p=sigmoid(z)
        self.loss+=float(w@(np.logaddexp(0,z)-y*z))
        self.gradient+=x.T@(w*(p-y))
        self.hessian+=x.T@((w*p*(1-p))[:,None]*x)
        self.weight+=float(w.sum());self.rows+=len(x)

    def objective(self):
        coefficients=self.model.coefficients.copy();coefficients[-1]=0
        gradient=self.gradient+self.regularization*coefficients
        hessian=self.hessian+np.diag(np.r_[np.full(len(coefficients)-1,self.regularization),0.])
        loss=self.loss+.5*self.regularization*float(coefficients@coefficients)
        return loss,gradient,hessian

    def step(self):
        assert self.rows>0 and abs(self.weight-1.)<1e-7
        loss,gradient,hessian=self.objective()
        direction=np.linalg.solve(hessian+1e-8*np.eye(len(gradient)),gradient)
        # Fixed deterministic damping, with no data-dependent extra passes.
        norm=float(np.linalg.norm(direction))
        if norm>2.:direction*=2./norm
        self.model.coefficients-=direction
        assert np.isfinite(self.model.coefficients).all()
        return dict(weighted_objective_before_step=loss,gradient_norm=float(np.linalg.norm(gradient)),
            undamped_step_norm=norm,actual_step_norm=float(np.linalg.norm(direction)),
            rows=self.rows,weight=self.weight)


def class_video_weights(targets,positive,negative,videos):
    """Equal video weight, and equal available positive/negative class weight."""
    assert videos>0 and positive>=0 and negative>=0 and positive+negative>0
    y=np.asarray(targets,np.float64)
    assert np.logical_or(y==0,y==1).all()
    classes=int(positive>0)+int(negative>0)
    assert not(y==1).any()or positive>0
    assert not(y==0).any()or negative>0
    return np.where(y==1,1/max(positive,1),1/max(negative,1))/(videos*classes)
