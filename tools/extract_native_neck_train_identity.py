#!/usr/bin/env python3
"""Additional pre/BN-neck controls, preserving the original RGB feature bank."""
import argparse
import json
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
from yolox.tracker.native_cue import native_input


def main():
    p=argparse.ArgumentParser();p.add_argument('--cue',choices=['reference','rgb','ir'],required=True);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in {'2','3'}:raise ValueError('Only GPUs2/3')
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root'])
    space_path=ROOT/'configs/perceptual_native_neck_feature_space_20261123.json';config_path=ROOT/'configs/perceptual_native_neck_metric_20261123.json';config=json.loads(config_path.read_text())
    source=w/'train_bank/train.npz';metadata=json.loads((source.parent/'complete.json').read_text())
    if metadata['Dev_rows_used'] or metadata['Test_rows_used'] or sha256(source)!=metadata['feature_bank_sha256']:raise ValueError('Train source changed')
    if a.cue=='reference':
        if a.seed!=0:raise ValueError('One frozen reference only')
        checkpoint_path=Path(config['initial_checkpoint'])
        if sha256(checkpoint_path)!=config['initial_sha256']:raise ValueError('Reference changed')
    else:
        folder=w/'native_neck_metric'/f'{a.cue}_s{a.seed}';completion=json.loads((folder/'complete.json').read_text())
        if completion['status']!='COMPLETE_FIXED36_NATIVE_NECK_METRIC_FIT' or completion['epochs']!=36 or completion['Dev_fit_rows'] or completion['Test_fit_rows']:raise ValueError('Native fitting incomplete')
        checkpoint_path=folder/'epoch36.pt'
        if not checkpoint_path.exists():raise ValueError('Wait for completed exact FP32 neck endpoint')
    free=int(subprocess.check_output(['nvidia-smi','-i',physical,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<12*1024:raise RuntimeError('Bounded feature worker requires12GiB free')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2,0);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    model=AppearanceEncoder();model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
    model=model.cuda().float().eval()
    directory=w/'native_neck_train_features'/f'{a.cue}_s{a.seed}';directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():raise ValueError('Completed native Train features exist')
    with np.load(source) as bank:arrays={key:bank[key].copy() for key in ['label','frame','sequence']}
    count=len(arrays['label']);arrays.update(pre_neck=np.zeros((count,512),np.float32),post_neck=np.zeros((count,512),np.float32),projection_norm=np.zeros(count,np.float32));filled=np.zeros(count,bool)
    started=time.perf_counter();sources={};view='rgb' if a.cue=='reference' else a.cue
    with torch.inference_mode():
        for sequence in protocol['train']:
            path=w/'native_crops'/sequence/'crops.npz';meta=json.loads((path.parent/'complete.json').read_text())
            if meta['status']!='COMPLETE_TRAIN_ONLY' or meta['Dev_rows'] or meta['Test_rows'] or sha256(path)!=meta['crop_sha256']:raise ValueError('Train crop provenance')
            with np.load(path) as bank:indices=bank['indices'].copy();pixels=torch.from_numpy(bank[view]);rectangles=torch.from_numpy(bank['rectangles'])
            if filled[indices].any():raise ValueError('Duplicate crop indices')
            for lo in range(0,len(indices),64):
                selected=indices[lo:lo+64];value=native_input(pixels[lo:lo+64],rectangles[lo:lo+64],'cuda')
                raw=model.projection(model.backbone(value));pre=F.normalize(raw,dim=-1);post=F.normalize(model.neck(raw),dim=-1)
                arrays['pre_neck'][selected]=pre.cpu().numpy();arrays['post_neck'][selected]=post.cpu().numpy();arrays['projection_norm'][selected]=raw.norm(dim=-1).cpu().numpy()
            filled[indices]=True;sources[sequence]=meta['crop_sha256'];print(json.dumps(dict(cue=a.cue,seed=a.seed,sequence=sequence,observations=len(indices))),flush=True)
    if not filled.all() or not np.isfinite(arrays['post_neck']).all():raise ValueError('Missing/invalid native identity rows')
    path=directory/'features.npz';temporary=directory/'features.tmp.npz';np.savez_compressed(temporary,**arrays);os.replace(temporary,path)
    dump(directory/'complete.json',dict(status='COMPLETE_TRAIN_ONLY_NATIVE_FEATURES',cue=a.cue,seed=a.seed,observations=count,fit_sequences=protocol['train'],Dev_fit_rows=0,Test_fit_rows=0,config_sha256=sha256(space_path),native_plan_sha256=sha256(config_path),source_bank_sha256=sha256(source),checkpoint=str(checkpoint_path),checkpoint_sha256=sha256(checkpoint_path),crop_sources=sources,feature_sha256=sha256(path),bytes=path.stat().st_size,physical_gpu=physical,precision='FP32 parameters, BN buffers and feature inference; TF32 disabled',pre_neck_mean_norm=float(np.linalg.norm(arrays['pre_neck'].mean(0))),post_neck_mean_norm=float(np.linalg.norm(arrays['post_neck'].mean(0))),projection_norm_median=float(np.median(arrays['projection_norm'])),source_sha256=sha256(__file__),seconds=time.perf_counter()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,labels_used_in_forward=False))


if __name__=='__main__':main()
