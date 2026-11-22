#!/usr/bin/env python3
"""Matched RGB/new-IR metric control, fitted only to the six Train videos."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from yolox.tracker.evidence_adapter import EvidenceAdapter,pack_evidence
from tools.run_pseudo_ir_comparison import dump,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--cue',choices=['rgb','ir'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--epochs',type=int,default=18)
    p.add_argument('--wait-for-bank',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in {'2','3'}:raise RuntimeError('Only physical GPU 2 or 3')
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    config_path=ROOT/'configs/perceptual_metric_20261122.json';config=json.loads(config_path.read_text())
    bank=Path(plan['work_root'])/'train_bank'
    while not (bank/'complete.json').exists():
        if not a.wait_for_bank:raise RuntimeError('Incomplete fit bank')
        time.sleep(5)
    metadata=json.loads((bank/'complete.json').read_text());path=bank/'train.npz'
    if metadata['status']!='COMPLETE_TRAIN_ONLY' or metadata['Dev_rows_used'] or metadata['Test_rows_used']:raise ValueError('Illegal fit partition')
    if sha256(path)!=metadata['feature_bank_sha256']:raise ValueError('Fit bank changed')
    output=Path(plan['work_root'])/'metric'/f'{a.cue}_s{a.seed}';output.mkdir(parents=True,exist_ok=True)
    if (output/'complete.json').exists() and not a.resume:raise ValueError('Completed fitting already exists')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25,0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with np.load(path) as z:data={k:z[k] for k in z.files}
    raw,evidence=pack_evidence(data,a.cue);labels=data['label'];sequence=data['sequence'];frames=data['frame']
    if len(np.unique(labels))!=98 or len(np.unique(sequence))!=6:raise ValueError('Unexpected fitting identities/videos')
    x=torch.from_numpy(raw).cuda();e=torch.from_numpy(evidence).cuda();y=torch.from_numpy(labels).long().cuda();f=torch.from_numpy(frames).cuda()
    model=EvidenceAdapter(**config['architecture']).cuda()
    model.mean.copy_(e.mean(dim=0));model.std.copy_(e.std(dim=0).clamp_min(.05))
    prototypes=F.normalize(torch.stack([x[y==i].mean(0) for i in range(98)]),dim=-1)
    class_sequence=torch.tensor([int(sequence[labels==i][0]) for i in range(98)],device='cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['optimizer']['lr'],weight_decay=config['optimizer']['weight_decay'])
    sampler=np.random.default_rng(a.seed);history=[];start_epoch=0
    by_identity={int(i):np.flatnonzero(labels==i) for i in np.unique(labels)}
    by_sequence={int(s):np.unique(labels[sequence==s]) for s in np.unique(sequence)}
    if a.resume:
        c=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
        if c['train_bank_sha256']!=metadata['feature_bank_sha256'] or c['cue']!=a.cue or c['seed']!=a.seed:raise ValueError('Resume provenance mismatch')
        model.load_state_dict(c['model']);optimizer.load_state_dict(c['optimizer']);sampler.bit_generator.state=c['sampler_state']
        torch.set_rng_state(c['torch_rng']);torch.cuda.set_rng_state_all(c['cuda_rng']);history=c['history'];start_epoch=c['epoch']
    if a.epochs>config['maximum_epochs'] or a.epochs<start_epoch:raise ValueError('Invalid registered epoch boundary')
    steps=math.ceil(len(labels)/64);started=time.perf_counter()
    for epoch in range(start_epoch,a.epochs):
        lr=.001*(.1+.9*(1+math.cos(math.pi*epoch/36))/2)
        for group in optimizer.param_groups:group['lr']=lr
        model.train();totals=np.zeros(4)
        for _ in range(steps):
            seq=int(sampler.choice(list(by_sequence)));ids=by_sequence[seq]
            chosen=sampler.choice(ids,8,replace=len(ids)<8);indices=[]
            for identity in chosen:
                available=by_identity[int(identity)];strata=np.array_split(available,8)
                for subset in strata:indices.append(int(sampler.choice(subset if len(subset) else available)))
            idx=torch.tensor(indices,device='cuda');base=x[idx];obs=e[idx];target=y[idx]
            z=model(base,obs+torch.randn_like(obs)*config['noise_std'])
            logits=(z@prototypes.T)/.1;logits=logits.masked_fill(class_sequence[None]!=seq,-10000)
            ce=F.cross_entropy(logits,target)
            distance=torch.cdist(z,z);same=target[:,None]==target[None]
            positive=same&((f[idx,None]-f[idx][None]).abs()>=25);negative=~same
            valid=positive.any(1)&negative.any(1)
            if valid.any():
                hp=distance.masked_fill(~positive,-torch.inf).amax(1);hn=distance.masked_fill(~negative,torch.inf).amin(1)
                triplet=F.relu(hp[valid]-hn[valid]+.2).mean()
            else:triplet=z.sum()*0
            anchor=(z-F.normalize(base,dim=-1)).square().sum(1).mean()
            loss=ce+.5*triplet+.05*anchor
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite fit loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            totals+=np.array([loss.item(),ce.item(),triplet.item(),anchor.item()])
        record=dict(epoch=epoch+1,lr=lr,steps=steps,observations=steps*64,losses=dict(zip(['total','prototype_CE','triplet','anchor'],(totals/steps).tolist())))
        history.append(record);print(json.dumps(record),flush=True)
        common=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},architecture=config['architecture'],cue=a.cue,seed=a.seed,epoch=epoch+1,fit_sequences=metadata['sequences'],train_bank_sha256=metadata['feature_bank_sha256'],config_sha256=sha256(config_path))
        latest=dict(common,optimizer=optimizer.state_dict(),sampler_state=sampler.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),history=history)
        torch.save(latest,output/'latest.tmp.pt');os.replace(output/'latest.tmp.pt',output/'latest.pt')
        dump(output/'history.json',history)
        if (epoch+1)%6==0:
            torch.save(common,output/f'epoch{epoch+1:02d}.pt')
    dump(output/'complete.json',dict(status='COMPLETE_FIT_STAGE',cue=a.cue,seed=a.seed,epochs=a.epochs,physical_gpu=physical,seconds=time.perf_counter()-started,train_bank_sha256=metadata['feature_bank_sha256'],Dev_fitting_rows=0,Test_fitting_rows=0,source_sha256=sha256(__file__),history=history))


if __name__=='__main__':main()
