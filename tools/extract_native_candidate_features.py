#!/usr/bin/env python3
"""Fixed native encoders on exact Train proposal crops, including backgrounds."""
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
from tools.native_cue_io import load_native_encoder
from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.tracker.native_cue import native_input


def main():
    p=argparse.ArgumentParser();p.add_argument('--sequence',required=True);p.add_argument('--cue',choices=['rgb','ir'],required=True);a=p.parse_args()
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in {'2','3'}:raise ValueError('Only GPUs2/3')
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root'])
    if a.sequence not in protocol['train']:raise ValueError('Train-only candidate features')
    path=w/'native_candidate_crops'/a.sequence/'crops.npz';meta=json.loads((path.parent/'complete.json').read_text())
    if meta['status']!='COMPLETE_EXACT_TRAIN_CANDIDATE_CROPS' or meta['Dev_rows'] or meta['Test_rows'] or sha256(path)!=meta['crop_sha256']:raise ValueError('Candidate crop provenance')
    free=int(subprocess.check_output(['nvidia-smi','-i',physical,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<12*1024:raise RuntimeError('Keep inference memory headroom')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25,0);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    models={};checkpoints={}
    for seed in [0,1,2]:
        key=f'native_s{seed}';model,metadata=load_native_encoder(a.cue,seed,w,protocol);models[key]=model.cuda().eval();checkpoints[key]=metadata
    if a.cue=='rgb':
        model,metadata=load_native_encoder('reference',0,w,protocol);models['reference_post']=model.cuda().eval();checkpoints['reference_post']=metadata
    with np.load(path) as bank:pixels=torch.from_numpy(bank[a.cue]);rectangles=torch.from_numpy(bank['rectangles'])
    count=len(pixels)
    if count!=meta['observations']:raise ValueError('Candidate crop count changed')
    directory=w/'native_candidate_features'/a.sequence/a.cue;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():raise ValueError('Completed native candidate features exist')
    arrays={key:np.zeros((count,512),np.float32) for key in models};started=time.perf_counter()
    with torch.inference_mode():
        for lo in range(0,count,64):
            value=native_input(pixels[lo:lo+64],rectangles[lo:lo+64],'cuda')
            for key,model in models.items():
                raw=model.projection(model.backbone(value));arrays[key][lo:lo+64]=F.normalize(model.neck(raw),dim=-1).cpu().numpy()
            if (lo//64+1)%32==0 or lo+64>=count:
                progress=dict(sequence=a.sequence,cue=a.cue,observations=min(lo+64,count),total=count,physical_gpu=physical,seconds=time.perf_counter()-started)
                dump(directory/'progress.json',progress);print(json.dumps(progress),flush=True)
    for key,value in arrays.items():
        if not np.isfinite(value).all() or np.max(abs(np.linalg.norm(value,axis=1)-1),initial=0)>1e-4:raise ValueError('Invalid native candidate identity features')
    output=directory/'features.npz';temporary=directory/'features.tmp.npz';np.savez_compressed(temporary,**arrays);os.replace(temporary,output)
    dump(directory/'complete.json',dict(status='COMPLETE_EXACT_TRAIN_NATIVE_FEATURES',sequence=a.sequence,cue=a.cue,observations=count,feature_sha256=sha256(output),bytes=output.stat().st_size,crop_sha256=sha256(path),index_sha256=meta['index_sha256'],checkpoints=checkpoints,config_sha256=sha256(ROOT/'configs/perceptual_native_candidates_20261123.json'),source_sha256=sha256(__file__),pixel_source_sha256=sha256(ROOT/'yolox/tracker/native_cue.py'),model_io_source_sha256=sha256(ROOT/'tools/native_cue_io.py'),physical_gpu=physical,Dev_rows=0,Test_rows=0,labels_used_in_forward=False,precision='FP32 parameters, BN buffers and inference; no TF32',seconds=time.perf_counter()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30))


if __name__=='__main__':main()
