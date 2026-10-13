"""Training-only known-ID supervision for distinct ranking and NULL tasks."""
import torch
from torch.nn import functional as F
from .joint_identity_supervision import joint_identity_loss


def balanced_identity_loss(logits, supervision, legal, ranking_weight=.1, consistency_weight=.1):
    loss,parts=joint_identity_loss(logits,supervision,legal,ranking_weight=ranking_weight,consistency_weight=consistency_weight)
    target,allowed=supervision['target'],supervision['classes'].clone()
    allowed[:,:-1]&=legal.T
    rows=torch.arange(len(logits),device=logits.device)
    selected=(target>=0)&allowed[rows,target.clamp_min(0)]&torch.isfinite(logits[rows,target.clamp_min(0)])
    if not selected.any():
        return loss,dict(parts,balanced_classification=0.)
    active=logits[selected].masked_fill(~allowed[selected],-torch.inf);labels=target[selected]
    ce=F.cross_entropy(active,labels,reduction='none');existing=labels<logits.shape[1]-1
    groups=[ce[mask].mean() for mask in (existing,~existing) if mask.any()]
    balanced=torch.stack(groups).mean()
    return loss-ce.mean()+balanced,dict(parts,balanced_classification=float(balanced.detach()))


def conditional_identity_ranking_loss(costs, supervision, legal, temperature=.1, ranking_weight=.1, consistency_weight=.1):
    """Only known existing-ID competition; ignore NULL and unknown classes."""
    if not temperature>0:
        raise ValueError('Positive declared temperature required')
    target=supervision['target'].clone();target[target==len(costs)]=-1
    classes=supervision['classes'].clone();classes[:,-1]=False
    condition=dict(supervision,target=target,classes=classes)
    logits=torch.cat([-costs.T/temperature,costs.new_full((costs.shape[1],1),-torch.inf)],dim=1)
    return joint_identity_loss(logits,condition,legal,ranking_weight=ranking_weight,consistency_weight=consistency_weight)
