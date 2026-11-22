"""Causal candidate removal for the explicitly appearance-only Train regime."""
import torch


def masked_row_margin(similarity,valid):
    masked=similarity.masked_fill(~valid,-torch.inf)
    maximum,winner=masked.max(-1)
    without=masked.clone();without.scatter_(-1,winner[...,None],-torch.inf)
    second=without.max(-1).values
    count=valid.sum(-1);second=torch.where(count>1,second,torch.zeros_like(second))
    other=maximum[...,None].expand_as(similarity).clone();other.scatter_(-1,winner[...,None],second[...,None])
    margin=similarity-other
    return torch.where(valid,margin,torch.zeros_like(margin))


def remask_appearance_queries(base,cue,reference,valid):
    if torch.any(base[...,14:21][valid]!=0):raise ValueError('Local point evidence cannot be deleted after extraction')
    base=base.clone();cue=cue.clone();reference=reference.clone()
    base[...,22]=torch.log1p(valid.sum(-1).float())[...,None]/4
    RGB_margin=masked_row_margin(base[...,23],valid)
    base[...,24]=RGB_margin;reference[...,5]=RGB_margin
    cue[...,5]=masked_row_margin(cue[...,0],valid)
    return base,cue,reference
