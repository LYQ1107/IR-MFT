"""Targeted utility repair: separate genuine candidate ranking from NULL penalties."""
import numpy as np
import torch
from torch.nn import functional as F


def ranking_utility(rgb,ir,target,temperature=.1,ir_weight=.5):
    rgb=np.asarray(rgb,np.float64);ir=np.asarray(ir,np.float64)
    if rgb.ndim!=1 or ir.shape!=rgb.shape or not np.isfinite(rgb).all() or not np.isfinite(ir).all() or not np.isfinite([temperature,ir_weight,target]).all() or target!=int(target) or not 0<=target<=len(rgb) or temperature<=0 or not 0<=ir_weight<=1:raise ValueError('Valid fixed candidate ranking intervention required')
    if len(rgb)<2 or target==len(rgb) or np.array_equal(rgb,ir):return 0.
    def ce(cost):
        logits=-cost/temperature;peak=logits.max()
        return peak+np.log(np.exp(logits-peak).sum())-logits[int(target)]
    return float(ce(rgb)-ce((1-ir_weight)*rgb+ir_weight*ir))


def ranking_sign_loss(output,truth,delta=.1,sign_temperature=.05,sign_weight=.05,minimum_abs_utility=.005):
    if output.shape!=truth.shape or output.ndim!=1 or not torch.isfinite(output).all() or not torch.isfinite(truth).all() or not np.isfinite([delta,sign_temperature,sign_weight,minimum_abs_utility]).all() or delta<=0 or sign_temperature<=0 or sign_weight<0 or minimum_abs_utility<0:raise ValueError('Finite ranking utility supervision required')
    regression=F.huber_loss(output,truth,delta=delta)
    eligible=truth.abs()>minimum_abs_utility;positive=eligible & (truth>0);negative=eligible & (truth<0)
    parts=[]
    for mask in [positive,negative]:
        if mask.any():parts.append(F.binary_cross_entropy_with_logits(output[mask]/sign_temperature,(truth[mask]>0).float()))
    sign=torch.stack(parts).mean() if parts else output.sum()*0.
    return regression+sign_weight*sign


def decision_strata(rows,positions):
    result=[]
    for indices in positions:
        groups=[np.array([i for i in indices if (0 if rows[i]['target']==len(rows[i]['rgb']) else 1 if len(rows[i]['rgb'])==1 else 2)==kind],np.int64) for kind in range(3)]
        result.append([g for g in groups if len(g)])
    return result
