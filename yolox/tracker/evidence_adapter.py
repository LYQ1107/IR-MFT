"""Bounded, identity-initialized metric adaptation; an ordinary learned control.

The input cue is a declared RGB or new-pseudo-IR figure-ground view. Capacity,
loss, training identities and update budgets must be matched across cues.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def pack_evidence(data, cue):
    raw=np.asarray(data['embedding'],np.float32)
    if cue=='ir':
        masked=np.asarray(data['ir_mask_embedding'],np.float32)
        shape=np.asarray(data['ir_hog'],np.float32)
        quality=np.asarray(data['quality'],np.float32)
    elif cue=='rgb':
        masked=np.asarray(data['rgb_mask_embedding'],np.float32)
        shape=np.asarray(data['rgb_hog'],np.float32)
        # The RGB-only control must not consume even the auxiliary quality.
        quality=np.repeat(np.asarray(data['quality'],np.float32)[:,:1],2,axis=1)
    else:raise ValueError('Unknown declared input cue')
    return raw,np.concatenate([raw,masked-raw,shape,quality],axis=1)


class EvidenceAdapter(nn.Module):
    def __init__(self, hidden=128, amplitude=.35):
        super().__init__()
        self.amplitude=float(amplitude)
        self.register_buffer('mean',torch.zeros(1170))
        self.register_buffer('std',torch.ones(1170))
        self.net=nn.Sequential(nn.Linear(1170,hidden),nn.LayerNorm(hidden),nn.GELU(),
                               nn.Linear(hidden,512,bias=False))
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, raw, evidence):
        raw=raw.float()
        x=((evidence.float()-self.mean)/self.std).clamp(-8,8)
        correction=self.net(x)
        correction=correction/(1+correction.norm(dim=-1,keepdim=True))
        scale=raw.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        return F.normalize(raw+self.amplitude*scale*correction,dim=-1)


def apply_adapter(data, checkpoint_path):
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    model=EvidenceAdapter(**checkpoint['architecture'])
    model.load_state_dict(checkpoint['model'],strict=True);model.eval()
    raw,evidence=pack_evidence(data,checkpoint['cue'])
    projected=[]
    with torch.inference_mode():
        for lo in range(0,len(raw),512):
            projected.append(model(torch.from_numpy(raw[lo:lo+512]),torch.from_numpy(evidence[lo:lo+512])).numpy())
    embedding=np.concatenate(projected,axis=0) if projected else np.empty((0,512),np.float32)
    result=dict(data,embedding=embedding,ir_mask_embedding=embedding,rgb_mask_embedding=embedding)
    return result,dict(cue=checkpoint['cue'],seed=checkpoint['seed'],epoch=checkpoint['epoch'],
                       fit_sequences=checkpoint['fit_sequences'],train_bank_sha256=checkpoint['train_bank_sha256'])
