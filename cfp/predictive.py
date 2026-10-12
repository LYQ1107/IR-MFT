"""Bounded causal latent identity predictors and stopped-target losses."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def checked_history(features, frames, valid, query_frame, time_encoding='linear_clipped'):
    if features.ndim!=3 or frames.shape!=features.shape[:2] or valid.shape!=frames.shape or query_frame.shape!=(len(features),):
        raise ValueError('Aligned batched accepted histories required')
    if not 1<=features.shape[1]<=16 or not torch.isfinite(features).all() or not torch.isfinite(frames).all() or not torch.isfinite(query_frame).all():
        raise ValueError('Finite bounded history required')
    valid=valid.bool();lengths=valid.sum(1);available=lengths>0
    prefix=torch.arange(features.shape[1],device=features.device)[None,:]<lengths[:,None]
    if not torch.equal(valid,prefix):raise ValueError('History must be ordered, right padded and contiguous')
    if torch.any(valid & ((frames<=0)|(frames>=query_frame[:,None]))):
        raise ValueError('History may contain only positive timestamps strictly before the current query')
    if features.shape[1]>1 and torch.any(valid[:,1:] & (frames[:,1:]<=frames[:,:-1])):
        raise ValueError('Accepted history timestamps must increase strictly')
    source=F.normalize(features.detach().float(),dim=-1)*valid[...,None]
    if time_encoding not in ('linear_clipped','log1p_15000'):raise ValueError('Unknown declared temporal encoding')
    raw_age=(query_frame[:,None]-frames).float()
    age=(raw_age/37).clamp(0,2)*valid if time_encoding=='linear_clipped' else torch.log1p(raw_age.clamp_min(0))/math.log(15001)*valid
    gap=torch.zeros_like(age)
    if features.shape[1]>1:
        raw_gap=(frames[:,1:]-frames[:,:-1]).float()
        gap[:,1:]=(raw_gap/37).clamp(0,2)*valid[:,1:] if time_encoding=='linear_clipped' else torch.log1p(raw_gap.clamp_min(0))/math.log(15001)*valid[:,1:]
    times=torch.stack([age,gap],-1)
    last=source[torch.arange(len(source),device=source.device),(lengths-1).clamp_min(0)]*available[:,None]
    return source,times,lengths,available,last


class LatentIdentityPredictor(nn.Module):
    def __init__(self,dimension=512,architecture='gru',hidden=None,time_encoding='linear_clipped'):
        super().__init__()
        if architecture not in ('gru','transformer'):raise ValueError('Unknown temporal predictor')
        self.architecture=architecture
        if time_encoding not in ('linear_clipped','log1p_15000'):raise ValueError('Unknown declared temporal encoding')
        self.time_encoding=time_encoding
        hidden=hidden or (128 if architecture=='gru' else 96)
        self.input=nn.Linear(dimension+2,hidden)
        if architecture=='gru':
            self.temporal=nn.GRU(hidden,hidden,batch_first=True)
        else:
            layer=nn.TransformerEncoderLayer(hidden,4,dim_feedforward=144,dropout=0.,batch_first=True)
            self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.output=nn.Linear(hidden,dimension)
        # A causal last-observation initialization avoids an initial average-fish prototype.
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,features,frames,valid,query_frame):
        source,times,lengths,available,last=checked_history(features,frames,valid,query_frame,self.time_encoding)
        tokens=F.silu(self.input(torch.cat([source,times],-1)))
        if self.architecture=='gru':
            packed=nn.utils.rnn.pack_padded_sequence(tokens,lengths.clamp_min(1).cpu(),batch_first=True,enforce_sorted=False)
            _,state=self.temporal(packed);state=state[-1]
        else:
            safe_valid=valid.bool().clone();safe_valid[~available,0]=True
            future=torch.ones(tokens.shape[1],tokens.shape[1],dtype=torch.bool,device=tokens.device).triu(1)
            causal=torch.zeros_like(future,dtype=tokens.dtype).masked_fill(future,-torch.inf)
            padding=torch.zeros_like(safe_valid,dtype=tokens.dtype).masked_fill(~safe_valid,-torch.inf)
            output=self.temporal(tokens,mask=causal,src_key_padding_mask=padding)
            state=output[torch.arange(len(output),device=output.device),(lengths-1).clamp_min(0)]
        predicted=F.normalize(last+self.output(state),dim=-1)
        predicted=torch.where(available[:,None],predicted,torch.zeros_like(predicted))
        if not torch.isfinite(predicted).all():raise ValueError('Nonfinite latent identity prediction')
        return predicted,available


def bounded_identity_prototype(features,frames,valid,query_frame,mode='last',ema=.9):
    source,_,lengths,available,last=checked_history(features,frames,valid,query_frame)
    if mode=='last':return F.normalize(last,dim=-1),available
    if mode=='fifo':prototype=source.sum(1)/lengths.clamp_min(1)[:,None]
    elif mode=='ema':
        if not 0<=ema<=1:raise ValueError('Invalid bounded EMA decay')
        prototype=source[:,0].clone()
        for step in range(1,source.shape[1]):
            update=F.normalize(ema*prototype+(1-ema)*source[:,step],dim=-1)
            prototype=torch.where(valid[:,step,None],update,prototype)
    else:raise ValueError('Unknown bounded history baseline')
    return F.normalize(prototype,dim=-1)*available[:,None],available


def latent_identity_loss(predicted,target,candidates,positive_index,candidate_valid,
                         prediction_weight=1.,temperature=.1):
    if predicted.shape!=target.shape or candidates.ndim!=3 or candidates.shape[0]!=len(predicted) or candidates.shape[2]!=predicted.shape[1] or candidate_valid.shape!=candidates.shape[:2] or positive_index.shape!=(len(predicted),):
        raise ValueError('Aligned stopped targets and same-frame candidate identities required')
    if temperature<=0 or prediction_weight<0 or not all(torch.isfinite(x).all() for x in (predicted,target,candidates)):
        raise ValueError('Invalid latent prediction loss inputs')
    if torch.any(positive_index<0) or torch.any(positive_index>=candidates.shape[1]) or not candidate_valid[torch.arange(len(predicted),device=predicted.device),positive_index].all():
        raise ValueError('The true physical identity must be a valid same-frame training candidate')
    predicted=F.normalize(predicted,dim=-1);target=F.normalize(target.detach(),dim=-1)
    candidates=F.normalize(candidates.detach(),dim=-1)
    prediction=(1-(predicted*target).sum(-1)).mean()
    logits=torch.einsum('bd,bkd->bk',predicted,candidates)/temperature
    association=F.cross_entropy(logits.masked_fill(~candidate_valid.bool(),-torch.inf),positive_index)
    return prediction_weight*prediction+association,dict(prediction=prediction.detach(),association=association.detach())
