#!/usr/bin/env python3
"""Matched native-cue corrections around an exactly frozen RGB/null posterior."""
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
from tools.native_query_data import frozen_RGB_model
from tools.native_robust_query_data import load_robust_query_group as load_query_group
from yolox.tracker.native_cue_corruption import visible_mismatch,preserve_RGB_posterior
from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.tracker.association_calibrator import AssociationCalibrator
from yolox.tracker.native_query_mask import remask_appearance_queries


def main():
    p=argparse.ArgumentParser();p.add_argument('--frontend',choices=['new','rgb'],required=True);p.add_argument('--cue',choices=['rgb','ir'],required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--epochs',type=int,default=18);p.add_argument('--resume',action='store_true');a=p.parse_args()
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in {'2','3'}:raise ValueError('Only GPUs2/3')
    config_path=ROOT/'configs/perceptual_native_robust_query_20261123.json';config=json.loads(config_path.read_text())
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root'])
    if a.epochs>36 or a.epochs<18:raise ValueError('Registered common18..36 epochs only')
    prior_path=w/'native_RGB_prior'/f'{a.frontend}_s{a.seed}'/'complete.json';prior=json.loads(prior_path.read_text())
    if prior['status']!='COMPLETE_TRAIN_ONLY_CONVEX_RGB_PRIOR' or prior['config_sha256']!=config['reference_prior_plan_sha256'] or prior['Dev_fit_rows'] or prior['Test_fit_rows']:raise ValueError('RGB/null prior provenance')
    torch.set_num_threads(2);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.cuda.set_per_process_memory_fraction(.25,0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    frozen,core=frozen_RGB_model(a.frontend,a.seed);frozen=frozen.cuda().eval()
    if core['checkpoint_sha256']!=prior['core']['checkpoint_sha256']:raise ValueError('Frozen RGB reference mismatch')
    coefficients=prior['parameters'];t,weight,bias,density=[float(coefficients[key]) for key in ['t','a','b','c']]
    groups=[];sources={};means=[];seconds=[];total_queries=0
    for regime in ['local','long']:
        for sequence in protocol['train']:
            data=load_query_group(w,sequence,regime,a.cue,a.seed);name=sequence+'_'+regime;cache=prior['cache'][name];path=Path(cache['path'])
            if sha256(path)!=cache['sha256']:raise ValueError('Frozen RGB cached logits changed')
            with np.load(path) as z:core_logits=z['core_logits'].copy()
            q=torch.from_numpy(data['query']).cuda();position=torch.from_numpy(data['position']).cuda();Q=len(data['counts']);M=int(data['counts'].max());total_queries+=Q
            def pad(value,fill=0):
                source=torch.from_numpy(value).cuda();result=torch.full((Q,M)+value.shape[1:],fill,dtype=source.dtype,device='cuda');result[q,position]=source;return result
            group=dict(base=pad(data['base']),cue=pad(data['cue']),reference=pad(data['reference']),available=pad(data['available']),
                       valid=pad(np.ones(len(data['query']),bool)),mismatch_cue=pad(data['mismatch_cue']),mismatch_available=pad(data['mismatch_available']),core_logits=pad(core_logits),regime=regime,sequence=sequence,
                       target=torch.from_numpy(np.where(data['target']>=0,data['target'],M)).cuda())
            if regime=='long':group.update(RGB_extra=pad(data['RGB_extra']),point_reference=pad(data['point_reference']))
            groups.append(group);sources[name]=dict(query=data['sources'],prior_cache=cache['sha256'])
            # Query-uniform, then video/regime-uniform statistics match draws.
            row_weight=torch.from_numpy(1/data['counts'][data['query']]).cuda().float()
            base_tensor=torch.from_numpy(data['base']).cuda();cue_tensor=torch.from_numpy(data['cue']).cuda();ref_tensor=torch.from_numpy(data['reference']).cuda()
            mean_base=(base_tensor*row_weight[:,None]).sum(0)/Q;mean_cue=((cue_tensor+ref_tensor)*row_weight[:,None]).sum(0)/(2*Q)
            second_base=(base_tensor.square()*row_weight[:,None]).sum(0)/Q;second_cue=((cue_tensor.square()+ref_tensor.square())*row_weight[:,None]).sum(0)/(2*Q)
            means.append(torch.cat([mean_base,mean_cue]));seconds.append(torch.cat([second_base,second_cue]))
            del base_tensor,cue_tensor,ref_tensor,row_weight,data
            print(json.dumps(dict(status='QUERY_GROUP_PREPARED',frontend=a.frontend,cue=a.cue,seed=a.seed,group=name,queries=Q,maximum_candidates=M)),flush=True)
    if len(groups)!=12:raise ValueError('All video/regime groups required')
    model=AssociationCalibrator(**config['architecture']).cuda();mean=torch.stack(means).mean(0);variance=torch.stack(seconds).mean(0)-mean.square()
    model.mean.copy_(mean);model.std.copy_(variance.clamp_min(.05**2).sqrt())
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);sampler=np.random.default_rng(a.seed)
    output=w/'native_robust_query'/f'{a.frontend}_{a.cue}_s{a.seed}';output.mkdir(parents=True,exist_ok=True)
    if (output/'latest.pt').exists() and not a.resume:raise ValueError('Explicit resume required for an existing native query fit')
    history=[];start_epoch=0
    if a.resume:
        checkpoint=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
        if checkpoint['config_sha256']!=sha256(config_path) or checkpoint['sources']!=sources or checkpoint['frontend']!=a.frontend or checkpoint['cue']!=a.cue or checkpoint['seed']!=a.seed:raise ValueError('Native query resume mismatch')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer']);sampler.bit_generator.state=checkpoint['sampler_state']
        torch.set_rng_state(checkpoint['torch_rng']);torch.cuda.set_rng_state_all(checkpoint['cuda_rng']);history=checkpoint['history'];start_epoch=checkpoint['epoch']
    steps=math.ceil(total_queries/config['batch_queries']);started=time.perf_counter()
    for epoch in range(start_epoch,a.epochs):
        lr=.001*(.1+.9*(1+math.cos(math.pi*epoch/36))/2)
        for param in optimizer.param_groups:param['lr']=lr
        totals=np.zeros(8);regimes=[0,0];sample_digest=[];model.train()
        for step in range(steps):
            regime=int(sampler.integers(2));video=int(sampler.integers(6));group=groups[regime*6+video];regimes[regime]+=1
            index_np=sampler.integers(len(group['target']),size=config['batch_queries']);index=torch.from_numpy(index_np).cuda()
            base,cue,reference,available,valid,target,z=[group[k][index] for k in ['base','cue','reference','available','valid','target','core_logits']]
            M=valid.shape[-1]
            dropped=torch.zeros(len(index),dtype=torch.bool,device='cuda')
            if regime==1:
                dropped=(target<M)&torch.from_numpy(sampler.random(len(index))<.15).cuda();rows=torch.where(dropped)[0]
                valid=valid.clone();valid[rows,target[rows]]=False;target=target.clone();target[dropped]=M
                base,cue,reference=remask_appearance_queries(base,cue,reference,valid)
                with torch.no_grad():z=frozen(base[...,:23],group['RGB_extra'][index],group['point_reference'][index],torch.ones_like(valid))
            with torch.no_grad():
                real=(t*z+weight*base[...,23]).masked_fill(~valid,-1e9)
                null=bias+density*torch.log1p(valid.sum(-1).float())
                RGB_logits=torch.cat([real,null[:,None]],dim=-1);RGB_probability=RGB_logits.softmax(-1)
                correct=RGB_probability.gather(1,target[:,None]).squeeze(1);error=1-correct;importance=.05+error.square()
            delta=model.increment(base,cue,reference,available&valid)
            logits=torch.cat([(real+delta).masked_fill(~valid,-1e9),null[:,None]],dim=-1)
            raw=F.cross_entropy(logits,target,reduction='none');CE=(raw*importance).sum()/importance.sum()
            size=valid.sum(-1).clamp_min(1);square=(delta.square()*valid).sum(-1)/size
            preserve=.02*(correct.square()*square).mean();regularizer=.001*square.mean()
            mismatch,mismatch_available=visible_mismatch(group['mismatch_cue'][index],reference,group['mismatch_available'][index],valid,dropped)
            mismatch_delta=model.increment(base,mismatch,reference,mismatch_available)
            mismatch_KL=preserve_RGB_posterior(real,null,mismatch_delta,valid)
            loss=CE+preserve+regularizer+config['mismatch_KL_weight']*mismatch_KL
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native query loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            absolute=(delta.detach().abs()*valid).sum(-1)/size
            totals+=np.array([loss.item(),CE.item(),raw.mean().item(),preserve.item(),regularizer.item(),absolute.mean().item(),error.mean().item(),mismatch_KL.item()])
        record=dict(epoch=epoch+1,lr=lr,steps=steps,regime_batches=regimes,losses=dict(zip(['total','error_weighted_CE','raw_query_CE','RGB_preservation','increment_regularizer','absolute_increment','RGB_query_error','mismatch_preservation_KL'],(totals/steps).tolist())))
        history.append(record);print(json.dumps(record),flush=True)
        common=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},architecture=config['architecture'],feature_schema=config['feature_schema'],frontend=a.frontend,cue=a.cue,seed=a.seed,epoch=epoch+1,
                    config_sha256=sha256(config_path),feature_plan_sha256=config['source_feature_plan_sha256'],sources=sources,core=core,prior_path=str(prior_path),prior_sha256=sha256(prior_path),prior_parameters=coefficients,
                    fit_sequences=protocol['train'],Dev_fit_rows=0,Test_fit_rows=0,alignment_sha256=sha256(w/'native_alignment/complete.json'))
        latest=dict(common,optimizer=optimizer.state_dict(),sampler_state=sampler.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),history=history)
        temporary=output/'latest.tmp.pt';torch.save(latest,temporary);os.replace(temporary,output/'latest.pt');dump(output/'history.json',history)
        if (epoch+1)%6==0:
            temporary=output/f'epoch{epoch+1:02d}.tmp.pt';torch.save(common,temporary);os.replace(temporary,output/f'epoch{epoch+1:02d}.pt')
    dump(output/'complete.json',dict(status='COMPLETE_NATIVE_ROBUST_QUERY_FIT_STAGE',frontend=a.frontend,cue=a.cue,seed=a.seed,epochs=a.epochs,queries=total_queries,physical_gpu=physical,config_sha256=sha256(config_path),prior_sha256=sha256(prior_path),sources=sources,history=history,fit_sequences=protocol['train'],Dev_fit_rows=0,Test_fit_rows=0,source_sha256=sha256(__file__),model_source_sha256=sha256(ROOT/'yolox/tracker/association_calibrator.py'),mask_source_sha256=sha256(ROOT/'yolox/tracker/native_query_mask.py'),corruption_source_sha256=sha256(ROOT/'yolox/tracker/native_cue_corruption.py'),seconds=time.perf_counter()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30))


if __name__=='__main__':main()
