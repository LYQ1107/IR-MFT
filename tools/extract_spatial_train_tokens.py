#!/usr/bin/env python3
"""Compact spatial RGB identity evidence from the preserved Train observations."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from tools.extract_perceptual_mft_features import AppearanceEncoder
from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.tracker.perceptual_features import clipped_crop,letterbox_tensor_array
from yolox.tracker.spatial_identity_tokens import spatial_weights,pool_identity_tokens


def spatial_forward(model,batch):
    backbone=model.backbone
    value=backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(batch))))
    return backbone.layer4(backbone.layer3(backbone.layer2(backbone.layer1(value))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--sequence',required=True);p.add_argument('--max-observations',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=64);a=p.parse_args()
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in {'2','3'}:raise ValueError('Only physical GPU2/3')
    plan_path=ROOT/'configs/perceptual_spatial_binding_20261123.json';config=json.loads(plan_path.read_text())
    main_plan=ROOT/'configs/perceptual_mft_20261122.json';plan=json.loads(main_plan.read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root'])
    if a.sequence not in protocol['train']:raise ValueError('Train partition only')
    source=w/'train_bank/train.npz';metadata=json.loads((source.parent/'complete.json').read_text())
    if metadata['status']!='COMPLETE_TRAIN_ONLY' or metadata['Dev_rows_used'] or metadata['Test_rows_used'] or metadata['sequences']!=protocol['train'] or sha256(source)!=metadata['feature_bank_sha256']:raise ValueError('Invalid Train source')
    checkpoint_path=Path('/data3/liuyeqiang/IR-MFT_assets/cfp/phase1/reid_rgb_seed0/epoch12.pt')
    if sha256(checkpoint_path)!=metadata['checkpoint_sha256']:raise ValueError('Preserved encoder changed')
    free=int(subprocess.check_output(['nvidia-smi','-i',physical,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<12*1024:raise RuntimeError('Less than12GiB free for a bounded encoder worker')
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.cuda.set_per_process_memory_fraction(.2,0)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    directory=w/'spatial_tokens/train'/a.sequence
    if a.max_observations:directory=directory/f'smoke_{a.max_observations}'
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():raise ValueError('Completed spatial tokens already exist')
    if shutil.disk_usage(directory).free<40*2**30:raise RuntimeError('Keep the work filesystem reserve')
    with np.load(source) as bank:
        indices=np.flatnonzero(bank['sequence']==protocol['train'].index(a.sequence))
        if a.max_observations:indices=indices[:a.max_observations]
        frames=bank['frame'][indices].copy();boxes=bank['boxes'][indices].astype(np.float32);boxes[:,2:]+=boxes[:,:2]
        labels=bank['label'][indices].copy();original=bank['embedding'][indices].astype(np.float32)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    model=AppearanceEncoder();model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
    model=model.cuda().half().eval()
    count=len(indices);arrays=dict(indices=indices,frame=frames,boxes=boxes,label=labels)
    for mode in ['box','rgb','ir']:
        arrays[mode+'_tokens']=np.zeros((count,3,512),np.float16);arrays[mode+'_quality']=np.zeros((count,9),np.float16)
    started=time.perf_counter();processed=0
    unique_frames=np.unique(frames)
    with torch.inference_mode():
        for frame in unique_frames:
            frame=int(frame);pick=np.flatnonzero(frames==frame)
            rgb_path=Path(plan['data_root'])/'train'/a.sequence/'visible'/f'{frame:06d}.jpg'
            ir_path=Path(plan['generated_ir_root'])/'train'/a.sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
            rgb=cv2.imread(str(rgb_path));ir=cv2.imread(str(ir_path),cv2.IMREAD_UNCHANGED)
            if rgb is None or ir is None or ir.ndim!=2 or ir.dtype!=np.uint8 or rgb.shape[:2]!=ir.shape:raise ValueError('Missing actual Train RGB/new-IR pair')
            for lo in range(0,len(pick),a.batch_size):
                part=pick[lo:lo+a.batch_size];images=[];weights={mode:[] for mode in ['box','rgb','ir']}
                for index in part:
                    box=boxes[index];patch=clipped_crop(rgb,box,.05);ir_patch=clipped_crop(ir,box,.05)
                    spatial=spatial_weights(patch,ir_patch)
                    images.append(letterbox_tensor_array(rgb,box))
                    for mode in weights:
                        weights[mode].append(spatial[mode]['weights']);arrays[mode+'_quality'][index]=spatial[mode]['quality']
                batch=torch.from_numpy(np.stack(images)).cuda().half();feature=spatial_forward(model,batch)
                for mode in weights:
                    ownership=torch.from_numpy(np.stack(weights[mode])).cuda()
                    arrays[mode+'_tokens'][part]=pool_identity_tokens(feature,ownership,model.projection).cpu().numpy().astype(np.float16)
            processed+=len(pick)
            if processed%100< len(pick) or frame==int(unique_frames[-1]):
                progress=dict(sequence=a.sequence,processed=processed,observations=count,frames_processed=int(np.searchsorted(unique_frames,frame)+1),frames=len(unique_frames),seconds=time.perf_counter()-started,physical_gpu=physical,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
                dump(directory/'progress.json',progress);print(json.dumps(progress),flush=True)
    computed=arrays['box_tokens'][:,0].astype(np.float32)
    similarity=(computed*original).sum(1)/np.maximum(np.linalg.norm(computed,axis=1)*np.linalg.norm(original,axis=1),1e-8)
    if float(np.median(similarity))<.9999 or float(similarity.min())<.995:raise ValueError('Uniform pooled RGB drifted from the preserved encoder')
    path=directory/'tokens.npz';temporary=directory/'tokens.tmp.npz';np.savez_compressed(temporary,**arrays);os.replace(temporary,path)
    dump(directory/'complete.json',dict(status='SMOKE_COMPLETE' if a.max_observations else 'COMPLETE_TRAIN_ONLY',sequence=a.sequence,observations=count,frames=len(unique_frames),physical_gpu=physical,source_bank_sha256=sha256(source),plan_sha256=sha256(plan_path),main_plan_sha256=sha256(main_plan),checkpoint_sha256=sha256(checkpoint_path),token_sha256=sha256(path),bytes=path.stat().st_size,source_sha256=sha256(__file__),token_source_sha256=sha256(ROOT/'yolox/tracker/spatial_identity_tokens.py'),encoder_source_sha256=sha256(ROOT/'tools/extract_perceptual_mft_features.py'),RGB_uniform_cosine=dict(minimum=float(similarity.min()),median=float(np.median(similarity))),seconds=time.perf_counter()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,Dev_rows=0,Test_rows=0,labels_used_in_forward=False,scope='Exactly the preserved labeled Train observation index; labels copied only for later Train fitting/audit. Actual current RGB/new-IR pixels and boxes alone enter token extraction. No future-dependent operations.'))
    print('SPATIAL_TOKENS_COMPLETE',a.sequence,count,flush=True)


if __name__=='__main__':main()
