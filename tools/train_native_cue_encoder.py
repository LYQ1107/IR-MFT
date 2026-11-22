#!/usr/bin/env python3
"""Equal-budget native RGB/new-IR encoders with a frozen paired RGB teacher."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from tools.extract_perceptual_mft_features import AppearanceEncoder
from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.tracker.native_cue import native_input,native_forward,same_video_triplet


def main():
    p=argparse.ArgumentParser();p.add_argument('--cue',choices=['rgb','ir'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in {'2','3'}:raise ValueError('Only physical GPUs2/3')
    config_path=ROOT/'configs/perceptual_native_cue_20261123.json';config=json.loads(config_path.read_text())
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root'])
    initial=Path(config['initial_checkpoint'])
    if sha256(initial)!=config['initial_sha256']:raise ValueError('Preserved initialization changed')
    free=int(subprocess.check_output(['nvidia-smi','-i',physical,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<12*1024:raise RuntimeError('Encoder worker requires12GiB free')
    torch.set_num_threads(2);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.cuda.set_per_process_memory_fraction(.35,0);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=w/'train_bank/train.npz';metadata=json.loads((source.parent/'complete.json').read_text())
    if metadata['status']!='COMPLETE_TRAIN_ONLY' or metadata['sequences']!=protocol['train'] or metadata['Dev_rows_used'] or metadata['Test_rows_used'] or sha256(source)!=metadata['feature_bank_sha256']:raise ValueError('Train teacher provenance')
    with np.load(source) as bank:
        labels=bank['label'].copy();frames=bank['frame'].copy();videos=bank['sequence'].copy();teacher=bank['embedding'].astype(np.float32)
    count=len(labels);channels=3 if a.cue=='rgb' else 1
    pixels=np.zeros((count,128,256,channels),np.uint8);rectangles=np.zeros((count,4),np.int16);filled=np.zeros(count,bool);sources={}
    for sequence in protocol['train']:
        directory=w/'native_crops'/sequence;meta=json.loads((directory/'complete.json').read_text());path=directory/'crops.npz'
        if meta['status']!='COMPLETE_TRAIN_ONLY' or meta['Dev_rows'] or meta['Test_rows'] or meta['source_bank_sha256']!=sha256(source) or meta['config_sha256']!=sha256(config_path) or sha256(path)!=meta['crop_sha256']:raise ValueError('Native crop provenance')
        with np.load(path) as bank:
            indices=bank['indices']
            if np.any(filled[indices]) or not np.array_equal(labels[indices],bank['label']) or not np.array_equal(frames[indices],bank['frame']):raise ValueError('Native Train crop indexing')
            pixels[indices]=bank[a.cue];rectangles[indices]=bank['rectangles'];filled[indices]=True
        sources[sequence]=meta['crop_sha256']
    if not filled.all() or len(np.unique(labels))!=98:raise ValueError('Incomplete native Train observations')
    pixels=torch.from_numpy(pixels);rectangles=torch.from_numpy(rectangles)
    teacher=F.normalize(torch.from_numpy(teacher).cuda(),dim=-1)
    label_tensor=torch.from_numpy(labels).cuda();frame_tensor=torch.from_numpy(frames).cuda();video_tensor=torch.from_numpy(videos).cuda()
    checkpoint=torch.load(initial,map_location='cpu',weights_only=False);model=AppearanceEncoder();model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
    model=model.cuda();backbone=list(model.backbone.parameters());head=list(model.projection.parameters())+list(model.neck.parameters())+list(model.classifier.parameters())
    optimizer=torch.optim.AdamW([dict(params=backbone,lr=3e-5),dict(params=head,lr=3e-4)],weight_decay=.0001)
    scaler=torch.amp.GradScaler('cuda');sampler=np.random.default_rng(a.seed)
    by_identity={identity:np.flatnonzero(labels==identity) for identity in np.unique(labels)}
    strata={}
    for identity,indices in by_identity.items():
        edges=np.linspace(frames[indices].min(),frames[indices].max()+1,9)
        bins=[]
        for low,high in zip(edges[:-1],edges[1:]):
            eligible=indices[(frames[indices]>=low)&(frames[indices]<high)]
            if not len(eligible):eligible=indices[np.argsort(abs(frames[indices]-(low+high)/2))[:1]]
            bins.append(eligible)
        strata[identity]=bins
    output=w/'native_cue'/f'{a.cue}_s{a.seed}';output.mkdir(parents=True,exist_ok=True)
    if (output/'latest.pt').exists() and not a.resume:raise ValueError('Existing native fit; explicit resume required')
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        value=native_input(pixels[:8],rectangles[:8],'cuda');identity,logits=native_forward(model,value)
        if identity.shape!=(8,512) or logits.shape!=(8,98) or not torch.isfinite(identity).all() or not torch.isfinite(logits).all():raise ValueError('Native forward preflight failed')
    print(json.dumps(dict(status='NATIVE_PREFLIGHT_PASSED',physical_gpu=physical,cue=a.cue,seed=a.seed,Train_observations=count,Dev_rows=0,Test_rows=0)),flush=True)
    history=[];start_epoch=0
    if a.resume:
        checkpoint=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
        if checkpoint['config_sha256']!=sha256(config_path) or checkpoint['crop_sources']!=sources or checkpoint['cue']!=a.cue or checkpoint['seed']!=a.seed:raise ValueError('Native resume provenance')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer']);scaler.load_state_dict(checkpoint['scaler'])
        sampler.bit_generator.state=checkpoint['sampler_state'];torch.set_rng_state(checkpoint['torch_rng']);torch.cuda.set_rng_state_all(checkpoint['cuda_rng']);history=checkpoint['history'];start_epoch=checkpoint['epoch']
    steps=math.ceil(count/64);started=time.perf_counter()
    for epoch in range(start_epoch,config['epochs']):
        factor=.1+.9*(1+math.cos(math.pi*epoch/config['epochs']))/2
        optimizer.param_groups[0]['lr']=3e-5*factor;optimizer.param_groups[1]['lr']=3e-4*factor
        model.train()
        for parameter in backbone:parameter.requires_grad_(epoch>0)
        if epoch==0:model.backbone.eval()
        totals=np.zeros(6);sample_digest=hashlib.sha256();epoch_started=time.perf_counter();skipped_updates=0
        for _ in range(steps):
            identities=sampler.choice(np.array(list(strata)),size=8,replace=False)
            index=np.array([int(sampler.choice(eligible)) for identity in identities for eligible in strata[identity]],np.int64)
            sample_digest.update(index.astype('<i8').tobytes());cpu_index=torch.from_numpy(index);index=cpu_index.cuda()
            value=native_input(pixels[cpu_index],rectangles[cpu_index],'cuda')
            flip=torch.rand(64,device='cuda')<.5;value=torch.where(flip[:,None,None,None],value.flip(-1),value)
            for i in range(64):
                if sampler.random()>=.2:continue
                area=float(sampler.uniform(.05,.2));aspect=float(sampler.uniform(.5,2));rh=min(128,max(1,round(math.sqrt(area*128*256/aspect))));rw=min(256,max(1,round(math.sqrt(area*128*256*aspect))))
                y=int(sampler.integers(129-rh));x=int(sampler.integers(257-rw));value[i,:,y:y+rh,x:x+rw]=0
            with torch.autocast('cuda',dtype=torch.float16):identity,logits=native_forward(model,value)
            ce=F.cross_entropy(logits.float(),label_tensor[index],label_smoothing=.1)
            triplet,valid_fraction=same_video_triplet(identity,label_tensor[index],video_tensor[index],frame_tensor[index])
            distill=(1-(identity*teacher[index]).sum(-1)).mean();loss=ce+triplet+.5*distill
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native cue loss')
            optimizer.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            previous_scale=scaler.get_scale();scaler.step(optimizer);scaler.update();skipped_updates+=int(scaler.get_scale()<previous_scale)
            accuracy=(logits.argmax(1)==label_tensor[index]).float().mean()
            totals+=np.array([loss.item(),ce.item(),triplet.item(),distill.item(),valid_fraction,accuracy.item()])
        record=dict(epoch=epoch+1,steps=steps,optimizer_updates=steps-skipped_updates,AMP_skipped_updates=skipped_updates,AMP_scale=scaler.get_scale(),seconds=time.perf_counter()-epoch_started,lr_factor=factor,sample_sha256=sample_digest.hexdigest(),losses=dict(zip(['total','CE','triplet','paired_RGB_cosine','valid_triplet_fraction','accuracy'],(totals/steps).tolist())))
        history.append(record);print(json.dumps(record),flush=True)
        common=dict(architecture='Preserved RGB ResNet50/512/BNneck/98ID',cue=a.cue,seed=a.seed,epoch=epoch+1,config_sha256=sha256(config_path),initial_sha256=sha256(initial),crop_sources=sources,teacher_source_sha256=sha256(source),fit_sequences=protocol['train'],Dev_fit_rows=0,Test_fit_rows=0)
        latest=dict(common,model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),sampler_state=sampler.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),history=history)
        temporary=output/'latest.tmp.pt';torch.save(latest,temporary);os.replace(temporary,output/'latest.pt');dump(output/'history.json',history)
        if (epoch+1)%6==0:
            candidate=dict(common,model={k:v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu() for k,v in model.state_dict().items()},weight_precision='FP16 inference candidate; resumable latest retains FP32 training weights')
            temporary=output/f'epoch{epoch+1:02d}.tmp.pt';torch.save(candidate,temporary);os.replace(temporary,output/f'epoch{epoch+1:02d}.pt')
    dump(output/'complete.json',dict(status='COMPLETE_FIXED36_NATIVE_FIT',cue=a.cue,seed=a.seed,epochs=config['epochs'],physical_gpu=physical,config_sha256=sha256(config_path),crop_sources=sources,teacher_source_sha256=sha256(source),source_sha256=sha256(__file__),pixel_loss_source_sha256=sha256(ROOT/'yolox/tracker/native_cue.py'),fit_sequences=protocol['train'],Dev_fit_rows=0,Test_fit_rows=0,history=history,seconds=time.perf_counter()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,selected_native_epoch=36,selection='Fixed registered native representation budget; downstream association selection remains Dev-only.'))


if __name__=='__main__':main()
