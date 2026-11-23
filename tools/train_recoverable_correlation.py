#!/usr/bin/env python3
"""Conditional recoverability curriculum; unchanged original encoder/head."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch

from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.models.yolo_head import YOLOXHead
from yolox.models.local_correlation_repair import CorrelationRepair
from yolox.models.frozen_head_repair import head_predictions,original_supervised_loss,preserve_reference
from yolox.models.recoverable_curriculum import recovery_draw,weaken_recoverable_features,valid_counterfactual_locations,preserve_valid_reference


def load_banks(work,protocol,config_path,cue,smoke):
    sequences=protocol['train'][:2]if smoke else protocol['train']
    banks={};sources={}
    for sequence in sequences:
        directory=work/'correlation_cache'/sequence/('feature_smoke_2'if smoke else'features')
        meta=json.loads((directory/'complete.json').read_text())
        if meta['config_sha256']!=sha256(config_path)or meta['Dev_rows']or meta['Test_rows']:
            raise ValueError('Frozen Train-only feature source mismatch')
        expected='SMOKE_COMPLETE'if smoke else'COMPLETE_TRAIN_FROZEN_FPN'
        if meta['status']!=expected:raise ValueError('Incomplete frozen feature producer')
        maps={}
        for mode in set(['rgb',cue]):
            for level in [3,4,5]:
                key=f'{mode}_P{level}';info=meta['outputs'][key];path=Path(info['path'])
                if sha256(path)!=info['sha256']:raise ValueError('Frozen feature array changed')
                maps[key]=np.load(path,mmap_mode='r')
        target=Path(meta['targets']['path'])
        if sha256(target)!=meta['targets']['sha256']:raise ValueError('Train target cache changed')
        maps['targets']=np.load(target,mmap_mode='r');banks[sequence]=maps
        sources[sequence]=dict(manifest=str(directory/'complete.json'),sha256=sha256(directory/'complete.json'),frames=meta['frames'])
    return sequences,banks,sources


def batches(sequences,banks,seed,epoch):
    rng=np.random.default_rng(20261123+1000*seed+epoch)
    order={seq:rng.permutation(len(banks[seq]['targets']))for seq in sequences}
    sizes={len(x)for x in order.values()}
    if len(sizes)!=1 or next(iter(sizes))%2 or len(sequences)%2:
        raise ValueError('Matched two-video/two-frame batches require equal even counts')
    for chunk in range(next(iter(sizes))//2):
        videos=rng.permutation(sequences)
        for first in range(0,len(videos),2):
            yield [(str(seq),int(index))for seq in videos[first:first+2]for index in order[str(seq)][chunk*2:chunk*2+2]]


def tensors(banks,draw,cue):
    rgb=[];auxiliary=[]
    for level in [3,4,5]:
        value=np.stack([banks[seq][f'rgb_P{level}'][index]for seq,index in draw])
        rgb.append(torch.from_numpy(value).cuda().float())
        if cue=='rgb':auxiliary.append(rgb[-1])
        else:
            value=np.stack([banks[seq][f'{cue}_P{level}'][index]for seq,index in draw])
            auxiliary.append(torch.from_numpy(value).cuda().float())
    target=np.stack([banks[seq]['targets'][index]for seq,index in draw])
    return rgb,auxiliary,torch.from_numpy(target).cuda()


def main():
    p=argparse.ArgumentParser();p.add_argument('--cue',choices=['rgb','gray','ir'],required=True)
    p.add_argument('--variant',choices=['full','aligned','no_gate','no_counterfactual','no_curriculum','blanket_counterfactual'],default='full')
    p.add_argument('--seed',type=int,choices=[0,1,2],required=True);p.add_argument('--smoke-steps',type=int,default=0)
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in {'2','3'}:raise ValueError('Only physical GPU2/3')
    torch.set_num_threads(2);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.45,0)
    main_plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(main_plan['work_root'])
    config_path=ROOT/'configs/perceptual_correlation_recoverability_20261123.json';repair=json.loads(config_path.read_text())
    parent_path=ROOT/repair['parent_architecture_and_cache_plan'];config=json.loads(parent_path.read_text())
    if sha256(parent_path)!=repair['parent_sha256']:raise ValueError('Frozen parent architecture changed')
    activation_path=work/'correlation_recoverability_activation.json'
    parent_state=work/'correlation_pilot_policies_dev_state.json'
    if not activation_path.exists()or json.loads(parent_state.read_text())['status']!='COMPLETE':
        raise ValueError('Complete all parent Dev results and freeze conditional activation before fitting')
    activation=json.loads(activation_path.read_text())
    if activation['config_sha256']!=sha256(config_path)or activation['status']!='ACTIVATED_AFTER_COMPLETE_PARENT_DEV':
        raise ValueError('Conditional repair was not activated')
    protocol=json.loads((ROOT/main_plan['protocol']).read_text())
    out=work/'correlation_recoverability_fit'/f'{args.cue}_{args.variant}_s{args.seed}'
    if args.smoke_steps:out=out/f'smoke_{args.smoke_steps}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise ValueError('Completed correlation fit already exists')
    sequences,banks,sources=load_banks(work,protocol,parent_path,args.cue,False)
    eligibility_sources={}
    for sequence in sequences:
        directory=work/'correlation_recoverability'/sequence;eligibility=json.loads((directory/'complete.json').read_text())
        if eligibility['config_sha256']!=sha256(config_path)or eligibility['source_feature_manifest_sha256']!=sources[sequence]['sha256']:
            raise ValueError('Shared Train eligibility provenance mismatch')
        if sha256(eligibility['path'])!=eligibility['sha256']:raise ValueError('Train eligibility changed')
        with np.load(eligibility['path'])as data:
            banks[sequence]['eligible']=data['eligible'].copy()
        eligibility_sources[sequence]=dict(manifest=str(directory/'complete.json'),sha256=sha256(directory/'complete.json'))
    checkpoint=Path(config['source_RGB_checkpoint'])
    if sha256(checkpoint)!=config['source_RGB_sha256']:raise ValueError('Frozen RGB source changed')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    head=YOLOXHead(1,width=1.25)
    head.load_state_dict({key.removeprefix('head.'):value for key,value in saved['model'].items()if key.startswith('head.')},strict=True)
    del saved
    head=head.cuda().float().eval().requires_grad_(False)
    architecture=config['architecture']
    adapter=CorrelationRepair(latent=architecture['latent_channels'],radius=0 if args.variant=='aligned'else architecture['local_radius'],
        temperature=architecture['temperature'],risk_gate=args.variant!='no_gate',bound=architecture['maximum_residual_RMS_fraction']).cuda()
    fit=config['fitting'];optimizer=torch.optim.AdamW(adapter.parameters(),lr=fit['learning_rate'],weight_decay=fit['weight_decay'])
    history=[];start_epoch=0
    if args.resume:
        latest=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        if latest['config_sha256']!=sha256(config_path):raise ValueError('Resume plan changed')
        adapter.load_state_dict(latest['adapter']);optimizer.load_state_dict(latest['optimizer'])
        for state in optimizer.state.values():
            for key,value in state.items():
                if torch.is_tensor(value):state[key]=value.cpu()if key=='step'else value.cuda()
        start_epoch=latest['completed_epochs'];history=latest['history']
    state=dict(status='FITTING_ADAPTER_ONLY',cue=args.cue,variant=args.variant,seed=args.seed,physical_gpu=physical,
        config_sha256=sha256(config_path),source_RGB_sha256=config['source_RGB_sha256'],Train_sources=sources,
        frozen_original_head=True,frozen_original_encoder=True,trainable_parameters=sum(x.numel()for x in adapter.parameters()),
        Train_videos=sequences,Dev_fit_rows=0,Test_read=False,
        source_sha256=sha256(__file__),adapter_source_sha256=sha256(ROOT/'yolox/models/local_correlation_repair.py'),
        loss_source_sha256=sha256(ROOT/'yolox/models/frozen_head_repair.py'),
        curriculum_source_sha256=sha256(ROOT/'yolox/models/recoverable_curriculum.py'),
        parent_sha256=sha256(parent_path),activation_sha256=sha256(activation_path),eligibility_sources=eligibility_sources,
        control_disclosure='All cues share actual-IR-conditioned Train eligibility; RGB/gray use privileged Train selection but RGB-only inference.')
    dump(out/'fitting_manifest.json',state)
    epochs=1 if args.smoke_steps else fit['epochs'];steps_per_epoch=sum(len(banks[seq]['targets'])for seq in sequences)//4
    started=time.perf_counter();successful=0
    for epoch in range(start_epoch+1,epochs+1):
        values=[];draw_digest=hashlib.sha256();curriculum_digest=hashlib.sha256();selected_objects=0;adapter.train()
        draws=batches(sequences,banks,args.seed,epoch)
        if args.smoke_steps:
            draws=list(draws);draws=(draws*math.ceil(args.smoke_steps/len(draws)))[:args.smoke_steps]
        for step,draw in enumerate(draws):
            if args.smoke_steps and step>=args.smoke_steps:break
            draw_digest.update(json.dumps(draw,separators=(',',':')).encode())
            position=(epoch-1)*steps_per_epoch+step
            if position<steps_per_epoch:factor=(position+1)/steps_per_epoch
            else:factor=.03+.97*.5*(1+math.cos(math.pi*(position-steps_per_epoch)/max(1,(fit['epochs']-1)*steps_per_epoch)))
            for group in optimizer.param_groups:group['lr']=fit['learning_rate']*factor
            rgb,auxiliary,target=tensors(banks,draw,args.cue)
            eligible=np.stack([banks[seq]['eligible'][index]for seq,index in draw])
            curriculum=repair['curriculum']
            selected=recovery_draw(target,eligible,args.seed,epoch,step,curriculum['sample_probability'],curriculum['eligible_instance_probability'])
            if args.variant=='no_curriculum':selected[:]=False
            curriculum_digest.update(selected.tobytes());selected_objects+=int(selected.sum())
            rgb=weaken_recoverable_features(rgb,target,selected,config['training_input_hw'],
                strength=curriculum['contrast_reduction'],kernel=curriculum['background_pool_kernel'])
            with torch.no_grad():reference,reference_meta=head_predictions(head,rgb)
            repaired=adapter(rgb,auxiliary,reference_meta['scores'])
            prediction,meta=head_predictions(head,repaired)
            supervised=original_supervised_loss(head,prediction,meta,target,config['training_input_hw'])
            reliable=preserve_reference(prediction,reference,meta['strides'],reliable_only=True)
            wrong=prediction.new_zeros(())
            if args.variant!='no_counterfactual':
                permutation=torch.tensor([1,0,3,2],device='cuda')
                shifted=[x[permutation]for x in auxiliary]
                counterfactual=adapter(rgb,shifted,reference_meta['scores'])
                wrong_prediction,_=head_predictions(head,counterfactual)
                if args.variant=='blanket_counterfactual':wrong=preserve_reference(wrong_prediction,reference,meta['strides'])
                else:
                    validity=valid_counterfactual_locations(target,permutation,reference_meta)
                    wrong=preserve_valid_reference(wrong_prediction,reference,meta['strides'],validity)
                loss=supervised[0]+fit['lambda_reliable_RGB']*reliable+fit['lambda_wrong_auxiliary']*wrong
            else:loss=supervised[0]
            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite fusion loss; no unfair partial fit')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            gradient=torch.nn.utils.clip_grad_norm_(adapter.parameters(),fit['gradient_clip_norm'],error_if_nonfinite=True)
            optimizer.step();successful+=1
            values.append([float(loss.detach()),float(supervised[0].detach()),float(reliable.detach()),float(wrong.detach()),float(gradient)])
            if (step+1)%32==0 or args.smoke_steps:
                print(json.dumps(dict(epoch=epoch,step=step+1,steps=steps_per_epoch,loss=np.mean(values,axis=0).tolist(),
                    physical_gpu=physical,seconds=time.perf_counter()-started,peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30)),flush=True)
        row=dict(epoch=epoch,steps=len(values),loss=np.mean(values,axis=0).tolist(),sample_digest=draw_digest.hexdigest(),curriculum_digest=curriculum_digest.hexdigest(),selected_objects=selected_objects)
        history.append(row)
        exported={key:value.detach().cpu()for key,value in adapter.state_dict().items()}
        payload=dict(adapter=exported,completed_epochs=epoch,history=history,config_sha256=sha256(config_path),
                     metadata=state,optimizer=optimizer.state_dict())
        temporary=out/'latest.tmp.pt';torch.save(payload,temporary);os.replace(temporary,out/'latest.pt')
        if epoch in fit['snapshots']or args.smoke_steps:
            compact=dict(adapter=exported,completed_epochs=epoch,history=history,config_sha256=sha256(config_path),metadata=state)
            temporary=out/f'epoch{epoch:02d}.tmp.pt';torch.save(compact,temporary);os.replace(temporary,out/f'epoch{epoch:02d}.pt')
        dump(out/'history.json',history);print(json.dumps(row),flush=True)
    state.update(status='SMOKE_COMPLETE'if args.smoke_steps else'COMPLETE_TRAIN_ONLY_ADAPTER',epochs=epochs,
                 successful_steps_this_invocation=successful,seconds=time.perf_counter()-started,history=history,
                 peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30)
    dump(out/'complete.json',state)


if __name__=='__main__':main()
