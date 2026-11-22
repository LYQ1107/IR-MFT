#!/usr/bin/env python3
"""Error-focused cue residuals around an exactly frozen fitted RGB scorer."""
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
from yolox.tracker.rgb_reference_context import extend_rgb_context
from tools.run_pseudo_ir_comparison import dump,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--cue',choices=['rgb','ir'],required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--epochs',type=int,default=18);p.add_argument('--resume',action='store_true');a=p.parse_args()
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in {'2','3'}:raise RuntimeError('Only physical GPU2 or3')
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root'])
    config_path=ROOT/'configs/perceptual_strong_rgb_residual_v2_20261123.json';config=json.loads(config_path.read_text())
    selection_path=w/'strong_rgb_base_selection.json';selection=json.loads(selection_path.read_text())
    if sha256(selection_path)!=config['frozen_base_selection_sha256'] or selection['Test_selection']:raise ValueError('Frozen RGB selection changed')
    front=selection['selected_frontend'];base_info=next(x for x in selection['frontends'][front]['selected'] if x['seed']==a.seed)
    if sha256(base_info['path'])!=base_info['sha256']:raise ValueError('Frozen RGB weights changed')
    base_checkpoint=torch.load(base_info['path'],map_location='cpu',weights_only=False)
    if base_checkpoint['cue']!='rgb' or base_checkpoint['fit_sequences']!=protocol['train'] or base_checkpoint['Dev_fit_rows'] or base_checkpoint['Test_fit_rows']:raise ValueError('Invalid fitted RGB base')
    long_directory=w/config['long_bank_output'];long_manifest=json.loads((long_directory/'complete.json').read_text())
    if long_manifest['status']!='COMPLETE_TRAIN_ONLY_APPEARANCE_SETS' or long_manifest['Dev_rows'] or long_manifest['Test_rows'] or long_manifest['plan_sha256']!=sha256(config_path):raise ValueError('Long Train provenance')
    torch.set_num_threads(2);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.cuda.set_per_process_memory_fraction(.25,0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    frozen=AssociationCalibrator(**base_checkpoint['architecture']);frozen.load_state_dict(base_checkpoint['model'],strict=True);frozen=frozen.cuda().eval()
    for parameter in frozen.parameters():parameter.requires_grad_(False)
    groups=[];sources={};local_rows=0;stats_mean=[];stats_second=[]
    for regime in ['local','long']:
        for sequence in protocol['train']:
            if regime=='local':
                directory=w/'association_train'/sequence;meta=json.loads((directory/'complete.json').read_text());path=directory/'pairs.npz'
                if meta['status']!='COMPLETE_TRAIN_ONLY' or meta['Dev_rows'] or meta['Test_rows']:raise ValueError('Local Train provenance')
                expected=meta['bank_sha256']
            else:path=long_directory/(sequence+'.npz');expected=long_manifest['outputs'][sequence]['sha256']
            if sha256(path)!=expected:raise ValueError('Immutable Train bank changed')
            sources[regime+'/'+sequence]=expected
            with np.load(path) as z:
                b=z['base'].astype(np.float32);r=z['reference'].astype(np.float32);rgb=prepare_calibration_cue(z['rgb'],'rgb')
                cue=prepare_calibration_cue(z[a.cue],a.cue);labels=z['label'].astype(np.float32);query=z['query'].copy()
                valid=z['IR_available'].astype(bool) if a.cue=='ir' else np.ones(len(labels),bool)
            if regime=='local':local_rows+=len(labels)
            logits=np.zeros(len(labels),np.float32)
            with torch.inference_mode():
                for lo in range(0,len(b),4096):
                    bs=torch.from_numpy(b[lo:lo+4096]).cuda();cs=torch.from_numpy(rgb[lo:lo+4096]).cuda();rs=torch.from_numpy(r[lo:lo+4096]).cuda()
                    available=torch.ones(len(bs),dtype=torch.bool,device='cuda')
                    logits[lo:lo+len(bs)]=frozen(bs,cs,rs,available).cpu().numpy()
            probability=1/(1+np.exp(-logits));extended=extend_rgb_context(b,probability,query)
            data=dict(base=torch.from_numpy(extended).cuda(),cue=torch.from_numpy(cue).cuda(),reference=torch.from_numpy(r).cuda(),target=torch.from_numpy(labels).cuda(),available=torch.from_numpy(valid).cuda(),RGB_logits=torch.from_numpy(logits).cuda())
            groups.append(data)
            # Equal video/regime normalization follows the actual batch draw
            # distribution. Cue and its reference share the same statistics.
            both=torch.cat([data['cue'],data['reference']],dim=0)
            stats_mean.append(torch.cat([data['base'].mean(0),both.mean(0)]))
            stats_second.append(torch.cat([data['base'].square().mean(0),both.square().mean(0)]))
            print(json.dumps(dict(stage='TRAIN_FEATURES_PREPARED',regime=regime,sequence=sequence,rows=len(labels),positive_rows=int(labels.sum()))),flush=True)
    del frozen
    if len(groups)!=12:raise ValueError('Both regimes need all six Train videos')
    model=AssociationCalibrator(**config['architecture']).cuda();mean=torch.stack(stats_mean).mean(0);variance=torch.stack(stats_second).mean(0)-mean.square()
    model.mean.copy_(mean);model.std.copy_(variance.clamp_min(.05**2).sqrt())
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);sampler=np.random.default_rng(a.seed)
    output=w/'strong_rgb_residual'/f'{a.cue}_s{a.seed}';output.mkdir(parents=True,exist_ok=True)
    if (output/'complete.json').exists() and not a.resume:raise ValueError('Completed residual fitting exists')
    history=[];start_epoch=0
    if a.resume:
        checkpoint=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
        if checkpoint['config_sha256']!=sha256(config_path) or checkpoint['sources']!=sources or checkpoint['base_metadata']['sha256']!=base_info['sha256'] or checkpoint['cue']!=a.cue or checkpoint['seed']!=a.seed:raise ValueError('Residual resume provenance mismatch')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer']);sampler.bit_generator.state=checkpoint['sampler_state']
        torch.set_rng_state(checkpoint['torch_rng']);torch.cuda.set_rng_state_all(checkpoint['cuda_rng']);history=checkpoint['history'];start_epoch=checkpoint['epoch']
    if a.epochs>36 or a.epochs<start_epoch:raise ValueError('Invalid common epoch boundary')
    steps=math.ceil(local_rows/config['batch_size']);started=time.perf_counter()
    for epoch in range(start_epoch,a.epochs):
        lr=.001*(.1+.9*(1+math.cos(math.pi*epoch/36))/2)
        for group in optimizer.param_groups:group['lr']=lr
        totals=np.zeros(7);regime_batches=[0,0];model.train()
        for _ in range(steps):
            regime=int(sampler.integers(2));video=int(sampler.integers(6));data=groups[regime*6+video];regime_batches[regime]+=1
            index=torch.from_numpy(sampler.integers(len(data['target']),size=config['batch_size'])).cuda()
            b,c,r,y,available,z=[data[k][index] for k in ['base','cue','reference','target','available','RGB_logits']]
            delta=model.increment(b,c,r,available);prediction=z+delta
            error=(z.sigmoid()-y).abs().detach();weight=.05+error.square()
            raw=F.binary_cross_entropy_with_logits(prediction,y,reduction='none');ce=(raw*weight).sum()/weight.sum()
            preserve=.02*((1-error).square()*delta.square()).mean();regularizer=.001*delta.square().mean();loss=ce+preserve+regularizer
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite strong-RGB residual loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            totals+=np.array([loss.item(),ce.item(),raw.mean().item(),preserve.item(),regularizer.item(),delta.detach().abs().mean().item(),error.mean().item()])
        record=dict(epoch=epoch+1,lr=lr,steps=steps,regime_batches=regime_batches,losses=dict(zip(['total','error_weighted_BCE','raw_BCE','RGB_preservation','increment_regularizer','absolute_increment','frozen_RGB_error'],(totals/steps).tolist())))
        history.append(record);print(json.dumps(record),flush=True)
        common=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},architecture=config['architecture'],base_model=base_checkpoint['model'],base_architecture=base_checkpoint['architecture'],base_metadata=base_info,base_frontend=front,cue=a.cue,seed=a.seed,epoch=epoch+1,sources=sources,fit_sequences=protocol['train'],Dev_fit_rows=0,Test_fit_rows=0,config_sha256=sha256(config_path),prior_plan_sha256=base_checkpoint['prior_plan_sha256'],capacity_plan_sha256=sha256(config_path),base_selection_sha256=sha256(selection_path))
        latest=dict(common,optimizer=optimizer.state_dict(),sampler_state=sampler.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),history=history)
        torch.save(latest,output/'latest.tmp.pt');os.replace(output/'latest.tmp.pt',output/'latest.pt');dump(output/'history.json',history)
        if (epoch+1)%6==0:
            temporary=output/f'epoch{epoch+1:02d}.tmp.pt';torch.save(common,temporary);os.replace(temporary,output/f'epoch{epoch+1:02d}.pt')
    dump(output/'complete.json',dict(status='COMPLETE_FIT_STAGE',cue=a.cue,seed=a.seed,epochs=a.epochs,physical_gpu=physical,config_sha256=sha256(config_path),base_metadata=base_info,base_frontend=front,sources=sources,Dev_fit_rows=0,Test_fit_rows=0,seconds=time.perf_counter()-started,source_sha256=sha256(__file__),model_source_sha256=sha256(ROOT/'yolox/tracker/association_calibrator.py'),history=history))


if __name__=='__main__':main()
