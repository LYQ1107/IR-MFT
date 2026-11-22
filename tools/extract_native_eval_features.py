#!/usr/bin/env python3
"""Current-frame FP32 BN-neck features; RGB controls never open IR images."""
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
from torch.nn import functional as F
from tools.native_cue_io import load_native_encoder
from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.tracker.native_cue import native_crop,native_input


def main():
    p=argparse.ArgumentParser();p.add_argument('--cue',choices=['rgb','ir'],required=True);p.add_argument('--split',choices=['dev','test'],required=True);p.add_argument('--sequence',required=True);p.add_argument('--max-frames',type=int,default=0);a=p.parse_args()
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in {'2','3'}:raise ValueError('Only GPUs2/3')
    plan_path=ROOT/'configs/perceptual_mft_20261122.json';plan=json.loads(plan_path.read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root'])
    if a.sequence not in protocol['dev' if a.split=='dev' else 'historical_test']:raise ValueError('Sequence outside frozen partition')
    if json.loads((w/'native_train_features_state.json').read_text())['status']!='COMPLETE':raise ValueError('All native36 models and FP32 states required')
    free=int(subprocess.check_output(['nvidia-smi','-i',physical,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<12*1024:raise RuntimeError('Keep inference memory headroom')
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.cuda.set_per_process_memory_fraction(.25,0);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    models={};checkpoints={}
    for seed in [0,1,2]:
        key=f'native_s{seed}';model,metadata=load_native_encoder(a.cue,seed,w,protocol);models[key]=model.cuda().eval();checkpoints[key]=metadata
    if a.cue=='rgb':
        model,metadata=load_native_encoder('reference',0,w,protocol);models['reference_post']=model.cuda().eval();checkpoints['reference_post']=metadata
    directory=w/'native_eval_features'/a.split/a.sequence/a.cue
    if a.max_frames:directory=directory/f'smoke_{a.max_frames}'
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():raise ValueError('Completed native features exist')
    if shutil.disk_usage(directory).free<40*2**30:raise RuntimeError('Work filesystem reserve')
    sources={};arrays={};expected=None
    for arm,prefix in [('new','new_diffv2ir'),('rgb','rgb')]:
        path=Path(plan['frozen_detection_sources'][prefix+'_'+a.split])/'detections'/(a.sequence+'.npz')
        with np.load(path) as bank:data={key:bank[key].copy() for key in ['frame_ids','offsets','boxes']}
        if not np.array_equal(data['frame_ids'],np.arange(1,len(data['frame_ids'])+1)):raise ValueError('Missing source frames')
        if expected is not None:np.testing.assert_array_equal(data['frame_ids'],expected)
        expected=data['frame_ids'];count=len(data['boxes'])
        data.update({key:np.zeros((count,512),np.float32) for key in models});arrays[arm]=data;sources[arm]=dict(path=str(path),sha256=sha256(path))
    limit=min(len(expected),a.max_frames or len(expected));started=time.perf_counter()
    with torch.inference_mode():
        for frame in range(1,limit+1):
            if a.cue=='rgb':image_path=Path(plan['data_root'])/('train' if a.split=='dev' else 'test')/a.sequence/'visible'/f'{frame:06d}.jpg';image=cv2.imread(str(image_path))
            else:image_path=Path(plan['generated_ir_root'])/a.split/a.sequence/'infrared_diffv2ir'/f'{frame:06d}.png';image=cv2.imread(str(image_path),cv2.IMREAD_UNCHANGED)
            if image is None or (a.cue=='ir' and (image.ndim!=2 or image.dtype!=np.uint8)):raise ValueError('Actual current cue image missing')
            pixels=[];rectangles=[];places=[]
            for arm,data in arrays.items():
                lo,hi=map(int,data['offsets'][frame-1:frame+1])
                for index in range(lo,hi):
                    value,rectangle=native_crop(image,data['boxes'][index]);pixels.append(value);rectangles.append(rectangle);places.append((arm,index))
            for lo in range(0,len(pixels),64):
                value=native_input(torch.from_numpy(np.stack(pixels[lo:lo+64])),torch.from_numpy(np.stack(rectangles[lo:lo+64])),'cuda')
                for key,model in models.items():
                    raw=model.projection(model.backbone(value));features=F.normalize(model.neck(raw),dim=-1).cpu().numpy()
                    for feature,(arm,index) in zip(features,places[lo:lo+64]):arrays[arm][key][index]=feature
            if frame%100==0 or frame==limit:
                progress=dict(frame=frame,frames=limit,cue=a.cue,physical_gpu=physical,seconds=time.perf_counter()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
                dump(directory/'progress.json',progress);print(json.dumps(progress),flush=True)
    outputs={}
    for arm,data in arrays.items():
        if a.max_frames:
            count=int(data['offsets'][limit]);data={key:(value[:limit] if key=='frame_ids' else value[:limit+1] if key=='offsets' else value[:count]) for key,value in data.items()}
        for key in models:
            if not np.isfinite(data[key]).all() or np.max(abs(np.linalg.norm(data[key],axis=1)-1),initial=0)>1e-4:raise ValueError('Invalid native identity feature')
        path=directory/(arm+'.npz');temporary=directory/(arm+'.tmp.npz');np.savez_compressed(temporary,**data);os.replace(temporary,path)
        outputs[arm]=dict(path=str(path),sha256=sha256(path),frames=len(data['frame_ids']),detections=len(data['boxes']),bytes=path.stat().st_size)
    dump(directory/'complete.json',dict(status='SMOKE_COMPLETE' if a.max_frames else 'COMPLETE_NATIVE_CURRENT_FEATURES',split=a.split,sequence=a.sequence,cue=a.cue,physical_gpu=physical,frames=limit,outputs=outputs,proposal_sources=sources,checkpoints=checkpoints,plan_sha256=sha256(plan_path),feature_space_plan_sha256=sha256(ROOT/'configs/perceptual_native_feature_space_20261123.json'),source_sha256=sha256(__file__),model_io_source_sha256=sha256(ROOT/'tools/native_cue_io.py'),pixel_source_sha256=sha256(ROOT/'yolox/tracker/native_cue.py'),GT_read=False,future_frame_input=False,precision='FP32 including BN state, TF32 disabled',RGB_control_reads_IR_images=False,seconds=time.perf_counter()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30))


if __name__=='__main__':main()
