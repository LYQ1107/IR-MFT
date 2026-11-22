"""A deterministic RGB real/null calibration with an auditable exact gradient."""
import numpy as np


def query_prior_objective(parameters,group):
    t,a,b,c=np.asarray(parameters,np.float64);z=group['core_logits'];cosine=group['cosine'];query=group['query'];offset=group['offsets'];mask=group['valid'];label=group['label'];counts=group['effective_counts']
    edge=np.where(mask,t*z+a*cosine,-1e9);null=b+c*np.log1p(counts)
    maximum=np.maximum(np.maximum.reduceat(edge,offset[:-1]),null)
    exponent=np.exp(edge-maximum[query]);null_exp=np.exp(null-maximum);denominator=np.add.reduceat(exponent,offset[:-1])+null_exp
    real_probability=exponent/denominator[query];null_probability=null_exp/denominator
    null_label=1-np.add.reduceat(label,offset[:-1]);number=len(offset)-1
    loss=maximum+np.log(denominator)-np.add.reduceat(edge*label,offset[:-1])-null*null_label
    error=real_probability-label;null_error=null_probability-null_label
    gradient=np.array([(error*z).sum()/number,(error*cosine).sum()/number,null_error.mean(),(null_error*np.log1p(counts)).mean()])
    return float(loss.mean()),gradient
