"""Current-frame RGB competition features for a frozen reference scorer."""
import numpy as np


def extend_rgb_context(base,probability,query):
    base=np.asarray(base,np.float32);probability=np.asarray(probability,np.float32).reshape(-1);query=np.asarray(query)
    if len(base)!=len(probability) or len(query)!=len(base):raise ValueError('RGB context indexing mismatch')
    if not len(base):return np.zeros((0,26),np.float32)
    if np.any(np.diff(query)<0):raise ValueError('Current candidate groups must be contiguous')
    _,start,count=np.unique(query,return_index=True,return_counts=True)
    maximum=np.maximum.reduceat(probability,start);maximum_per_edge=np.repeat(maximum,count)
    position=np.arange(len(probability));winner=np.minimum.reduceat(np.where(probability==maximum_per_edge,position,len(probability)),start)
    without=probability.copy();without[winner]=-np.inf
    second=np.maximum.reduceat(without,start);second[count==1]=0
    alternative=maximum_per_edge.copy();alternative[winner]=second
    margin=np.repeat(maximum-second,count)
    return np.c_[base,probability,probability-alternative,margin].astype(np.float32)
