#!/usr/bin/env python3
"""Matched RGB/new-IR signed correspondence calibration, six Train videos only."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from yolox.tracker.association_calibrator import AssociationCalibrator,prepare_calibration_cue
from tools.run_pseudo_ir_comparison import dump,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--cue',choices=['rgb','ir'],required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--epochs',type=int,default=18);p.add_argument('--resume',action='store_true');p.add_argument('--wait-bank',action='store_true');a=p.parse_args()
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in {'2','3'}:raise RuntimeError('Only physical GPU2 or3')
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());protocol=json.loads((ROOT/plan['protocol']).read_text())
    config_path=ROOT/'configs/perceptual_pair_calibration_20261123.json';config=json.loads(config_path.read_text());w=Path(plan['work_root'])
    prior_path=ROOT/'configs/perceptual_missing_evidence_prior_20261123.json'
    prior_sha256=sha256(prior_path)
    capacity_sha256=sha256(ROOT/'configs/perceptual_calibration_capacity_20261123.json')
    while True:
        ready=all((w/'association_train'/s/'complete.json').exists() for s in protocol['train'])
        if ready:break
        if not a.wait_bank:raise RuntimeError('Full Train bank required')
        state=w/'association_train_state.json'
        if state.exists() and json.loads(state.read_text())['status']=='FAILED':raise RuntimeError('Train bank dependency failed')
        time.sleep(5)
    torch.set_num_threads(2);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.cuda.set_per_process_memory_fraction(.25,0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pieces={k:[] for k in ['base','reference','cue','label','available','video']};sources={}
    for video,sequence in enumerate(protocol['train']):
        directory=w/'association_train'/sequence;metadata=json.loads((directory/'complete.json').read_text());path=directory/'pairs.npz'
        if metadata['status']!='COMPLETE_TRAIN_ONLY' or metadata['Dev_rows'] or metadata['Test_rows']:raise ValueError('Illegal fit partition')
        if metadata['sequence']!=sequence or metadata['config_sha256']!=sha256(config_path) or metadata['bank_sha256']!=sha256(path):raise ValueError('Train bank provenance changed')
        sources[sequence]=metadata['bank_sha256']
        with np.load(path) as z:
            for key in ['base','reference','label']:pieces[key].append(z[key].copy())
            pieces['cue'].append(z[a.cue].copy())
            pieces['available'].append(z['IR_available'].copy() if a.cue=='ir' else np.ones(len(z['label']),np.uint8))
            pieces['video'].append(np.full(len(z['label']),video,np.uint8))
    data={k:np.concatenate(v) for k,v in pieces.items()};del pieces
    data['cue']=prepare_calibration_cue(data['cue'],a.cue)
    if not len(data['label']) or len(np.unique(data['video']))!=6:raise ValueError('Incomplete pair bank')
    output=w/'association_metric'/f'{a.cue}_s{a.seed}';output.mkdir(parents=True,exist_ok=True)
    if (output/'complete.json').exists() and not a.resume:raise ValueError('Completed fit cannot be silently overwritten')
    base=torch.from_numpy(data['base'].astype(np.float32)).cuda();cue=torch.from_numpy(data['cue'].astype(np.float32)).cuda();reference=torch.from_numpy(data['reference'].astype(np.float32)).cuda()
    target=torch.from_numpy(data['label'].astype(np.float32)).cuda();available=torch.from_numpy(data['available'].astype(bool)).cuda()
    by_video={i:np.flatnonzero(data['video']==i) for i in range(6)};positives=int(data['label'].sum());rows=len(target);del data
    model=AssociationCalibrator(**config['architecture']).cuda()
    # One shared normalization for cue and its RGB reference maintains exact
    # cancellation when the two views coincide. All statistics use Train only.
    model.mean[:23].copy_(base.mean(0));model.std[:23].copy_(base.std(0).clamp_min(.05))
    joined=torch.cat([cue,reference],dim=0);model.mean[23:].copy_(joined.mean(0));model.std[23:].copy_(joined.std(0).clamp_min(.05));del joined
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    sampler=np.random.default_rng(a.seed);history=[];start_epoch=0
    if a.resume:
        c=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
        if c['sources']!=sources or c['config_sha256']!=sha256(config_path) or c['cue']!=a.cue or c['seed']!=a.seed:raise ValueError('Resume provenance mismatch')
        if c['prior_plan_sha256']!=prior_sha256:raise ValueError('RGB missing-evidence policy changed')
        if c['capacity_plan_sha256']!=capacity_sha256:raise ValueError('RGB capacity control changed')
        model.load_state_dict(c['model']);optimizer.load_state_dict(c['optimizer']);sampler.bit_generator.state=c['sampler_state']
        torch.set_rng_state(c['torch_rng']);torch.cuda.set_rng_state_all(c['cuda_rng']);history=c['history'];start_epoch=c['epoch']
    if a.epochs>36 or a.epochs<start_epoch:raise ValueError('Invalid registered epoch boundary')
    steps=math.ceil(rows/config['batch_size']);started=time.perf_counter()
    for epoch in range(start_epoch,a.epochs):
        lr=.001*(.1+.9*(1+math.cos(math.pi*epoch/36))/2)
        for group in optimizer.param_groups:group['lr']=lr
        model.train();sums=np.zeros(5)
        for _ in range(steps):
            video=int(sampler.integers(6));index=sampler.choice(by_video[video],config['batch_size'],replace=True);index=torch.from_numpy(index).cuda()
            b,c,r,y,valid=base[index],cue[index],reference[index],target[index],available[index]
            delta=model.increment(b,c,r,valid);logits=model.RGB_offset(b)+delta
            ce=F.binary_cross_entropy_with_logits(logits,y);regularizer=.001*delta.square().mean();loss=ce+regularizer
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite association fit')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            probability=logits.detach().sigmoid()
            sums+=np.array([loss.item(),ce.item(),regularizer.item(),delta.detach().abs().mean().item(),((probability>.5)==y.bool()).float().mean().item()])
        record=dict(epoch=epoch+1,lr=lr,steps=steps,drawn_pair_rows=steps*config['batch_size'],losses=dict(zip(['total','BCE','increment_regularizer','absolute_increment','classification_accuracy'],(sums/steps).tolist())))
        history.append(record);print(json.dumps(record),flush=True)
        common=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},architecture=config['architecture'],cue=a.cue,seed=a.seed,epoch=epoch+1,sources=sources,fit_sequences=protocol['train'],Dev_fit_rows=0,Test_fit_rows=0,config_sha256=sha256(config_path),prior_plan_sha256=prior_sha256,capacity_plan_sha256=capacity_sha256)
        latest=dict(common,optimizer=optimizer.state_dict(),sampler_state=sampler.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),history=history)
        torch.save(latest,output/'latest.tmp.pt');os.replace(output/'latest.tmp.pt',output/'latest.pt');dump(output/'history.json',history)
        if (epoch+1)%6==0:torch.save(common,output/f'epoch{epoch+1:02d}.pt')
    dump(output/'complete.json',dict(status='COMPLETE_FIT_STAGE',cue=a.cue,seed=a.seed,epochs=a.epochs,pair_rows=rows,positive_rows=positives,physical_gpu=physical,Dev_fit_rows=0,Test_fit_rows=0,sources=sources,prior_plan_sha256=prior_sha256,capacity_plan_sha256=capacity_sha256,seconds=time.perf_counter()-started,source_sha256=sha256(__file__),model_source_sha256=sha256(ROOT/'yolox/tracker/association_calibrator.py'),history=history))


if __name__=='__main__':main()
