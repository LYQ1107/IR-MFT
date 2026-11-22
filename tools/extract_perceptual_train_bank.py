#!/usr/bin/env python3
"""Prepare train-only paired identity evidence under the exact current protocol.

Reuses the frozen historical crop coordinates, explicitly excludes every Dev
record, and recomputes RGB/new-IR views with the current inference encoder.
Only compact feature arrays are saved, never a new whole-image dataset.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from tools.extract_perceptual_mft_features import AppearanceEncoder
from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.tracker.perceptual_features import clipped_crop,color_histogram,foreground,hog,letterbox_tensor_array


def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-for')
    p.add_argument('--index',default='/data3/liuyeqiang/IR-MFT_assets/cfp/phase1/fish_paired_causal_seed0/index.json')
    p.add_argument('--checkpoint',default='/data3/liuyeqiang/IR-MFT_assets/cfp/phase1/reid_rgb_seed0/epoch12.pt')
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in {'2','3'}:raise RuntimeError('Only physical GPU 2 or 3')
    if a.wait_for:
        while not Path(a.wait_for).exists():time.sleep(5)
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    protocol_path=ROOT/plan['protocol'];protocol=json.loads(protocol_path.read_text())
    original=json.loads(Path(a.index).read_text())
    if original['protocol_sha256']!=sha256(protocol_path):raise ValueError('Crop protocol mismatch')
    records=[r for r in original['records'] if r['partition']=='train']
    if set(r['sequence'] for r in records)!=set(protocol['train']):raise ValueError('Incorrect train-only source')
    if any(r['train_label']<0 for r in records):raise ValueError('Unlabelled or Dev row in fit bank')
    output=Path(plan['work_root'])/'train_bank';output.mkdir(exist_ok=True)
    if (output/'complete.json').exists():raise ValueError('Completed train evidence already exists')
    if shutil.disk_usage(output).free<40*2**30:raise RuntimeError('Work disk reserve')
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.55,0)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=AppearanceEncoder();model.load_state_dict(checkpoint['model'],strict=True)
    model=model.cuda().half().eval();del checkpoint
    dimensions=dict(embedding=512,ir_mask_embedding=512,rgb_mask_embedding=512,ir_hog=144,rgb_hog=144,color=128,quality=2)
    data={k:np.zeros((len(records),v),np.float16) for k,v in dimensions.items()}
    data.update(label=np.array([r['train_label'] for r in records]),frame=np.array([r['frame_id'] for r in records]),
                sequence=np.array([protocol['train'].index(r['sequence']) for r in records]),
                boxes=np.array([r['bbox_xywh'] for r in records],np.float32))
    grouped={}
    for i,r in enumerate(records):grouped.setdefault((r['sequence'],r['frame_id']),[]).append(i)
    start=time.perf_counter()
    for position,((sequence,frame),indices) in enumerate(grouped.items()):
        rgb=cv2.imread(str(Path(plan['data_root'])/'train'/sequence/'visible'/f'{frame:06d}.jpg'))
        ir=cv2.imread(str(Path(plan['generated_ir_root'])/'train'/sequence/'infrared_diffv2ir'/f'{frame:06d}.png'),cv2.IMREAD_UNCHANGED)
        if rgb is None or ir is None or ir.ndim!=2 or rgb.shape[:2]!=ir.shape:raise ValueError('Invalid train pair')
        views,placements=[],[]
        for index in indices:
            box=data['boxes'][index].copy();box[2:]+=box[:2]
            r=clipped_crop(rgb,box,.05);g=cv2.cvtColor(r,cv2.COLOR_BGR2GRAY);t=clipped_crop(ir,box,.05)
            mask_ir,q_ir=foreground(t);mask_rgb,q_rgb=foreground(g)
            data['ir_hog'][index]=hog(t);data['rgb_hog'][index]=hog(g)
            data['color'][index]=color_histogram(r);data['quality'][index]=[q_rgb,q_ir]
            for name,mask in [('embedding',None),('ir_mask_embedding',mask_ir),('rgb_mask_embedding',mask_rgb)]:
                views.append(letterbox_tensor_array(rgb,box,mask));placements.append((name,index))
        with torch.inference_mode():
            for lo in range(0,len(views),64):
                batch=torch.from_numpy(np.stack(views[lo:lo+64])).cuda().half()
                features=model(batch).cpu().numpy()
                for value,(field,index) in zip(features,placements[lo:lo+64]):data[field][index]=value
        if (position+1)%100==0 or position+1==len(grouped):
            progress=dict(completed_frames=position+1,frames=len(grouped),seconds=time.perf_counter()-start,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'])
            dump(output/'progress.json',progress);print(json.dumps(progress),flush=True)
    path=output/'train.npz';tmp=output/'train.tmp.npz';np.savez_compressed(tmp,**data);os.replace(tmp,path)
    report=dict(status='COMPLETE_TRAIN_ONLY',observations=len(records),identities=len(set(data['label'])),frames=len(grouped),
                sequences=protocol['train'],Dev_rows_used=0,Test_rows_used=0,index_sha256=sha256(a.index),protocol_sha256=sha256(protocol_path),
                checkpoint_sha256=sha256(a.checkpoint),feature_bank_sha256=sha256(path),feature_bank_bytes=path.stat().st_size,
                physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],seconds=time.perf_counter()-start,source_sha256=sha256(__file__))
    dump(output/'complete.json',report);print('TRAIN_BANK_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
