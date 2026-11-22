#!/usr/bin/env python3
"""Unlabeled, causal foreground-point correspondences for fixed proposal streams."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from yolox.tracker.foreground_transport import MODES,point_groups,frame_correspondences
from tools.run_pseudo_ir_comparison import dump,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','test'],required=True)
    p.add_argument('--sequence',required=True);p.add_argument('--max-frames',type=int,default=0)
    a=p.parse_args();cv2.setNumThreads(1)
    plan_path=ROOT/'configs/perceptual_mft_20261122.json';plan=json.loads(plan_path.read_text())
    transport_path=ROOT/'configs/perceptual_transport_20261122.json';cfg=json.loads(transport_path.read_text())
    source_hashes=dict(extractor=sha256(__file__),algorithm=sha256(ROOT/'yolox/tracker/foreground_transport.py'),config=sha256(transport_path))
    protocol=json.loads((ROOT/plan['protocol']).read_text())
    if a.sequence not in protocol['historical_test' if a.split=='test' else 'dev']:raise ValueError('Unknown frozen sequence')
    out=Path(plan['work_root'])/'transport'/a.split/a.sequence
    if a.max_frames:out=out/f'smoke_{a.max_frames}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise ValueError('Completed producer cannot be overwritten')
    if shutil.disk_usage(out).free<40*2**30:raise RuntimeError('Work disk reserve')
    streams,outputs,sources={},{},{}
    for arm,prefix in [('new','new_diffv2ir'),('rgb','rgb')]:
        source=Path(plan['frozen_detection_sources'][prefix+'_'+a.split])/'detections'/(a.sequence+'.npz')
        with np.load(source) as z:data={k:z[k].copy() for k in ['frame_ids','offsets','boxes','scores']}
        if not np.array_equal(data['frame_ids'],np.arange(1,len(data['frame_ids'])+1)):raise ValueError('Frame coverage')
        if a.max_frames:
            frames=min(a.max_frames,len(data['frame_ids']));n=int(data['offsets'][frames])
            data=dict(frame_ids=data['frame_ids'][:frames],offsets=data['offsets'][:frames+1],boxes=data['boxes'][:n],scores=data['scores'][:n])
        streams[arm]=data;sources[arm]=dict(path=str(source),sha256=sha256(source))
        counts=np.diff(data['offsets']);total=len(data['boxes'])
        output=dict(frame_ids=data['frame_ids'],detection_offsets=data['offsets'],quality=np.zeros((total,2),np.float16))
        for lag in cfg['lags']:
            previous=np.r_[np.zeros(lag,dtype=int),counts[:-lag]] if len(counts)>lag else np.zeros(len(counts),int)
            elements=counts*previous
            output[f'vote_offsets_{lag}']=np.r_[0,np.cumsum(elements)]
            output[f'count_offsets_{lag}']=np.r_[0,np.cumsum(previous)]
            for mode in MODES:
                output[f'good_{lag}_{mode}']=np.zeros(int(previous.sum()),np.uint8)
                output[f'total_{lag}_{mode}']=np.zeros(int(previous.sum()),np.uint8)
                for target in MODES:output[f'vote_{lag}_{mode}_{target}']=np.zeros(int(elements.sum()),np.float16)
        outputs[arm]=output
    if not np.array_equal(streams['new']['frame_ids'],streams['rgb']['frame_ids']):raise ValueError('Control stream mismatch')
    history={};started=time.perf_counter();sampled_inputs={}
    for position,frame in enumerate(streams['new']['frame_ids']):
        frame=int(frame);source_split='test' if a.split=='test' else 'train'
        rgb_path=Path(plan['data_root'])/source_split/a.sequence/'visible'/f'{frame:06d}.jpg'
        ir_path=Path(plan['generated_ir_root'])/a.split/a.sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
        rgb=cv2.imread(str(rgb_path));ir=cv2.imread(str(ir_path),cv2.IMREAD_UNCHANGED)
        if rgb is None or ir is None or ir.ndim!=2 or ir.dtype!=np.uint8 or rgb.shape[:2]!=ir.shape:raise ValueError('Invalid actual RGB/new-IR pair')
        scale=np.array([960/rgb.shape[1],540/rgb.shape[0]]*2,np.float32)
        rgb=cv2.resize(rgb,(960,540),interpolation=cv2.INTER_AREA)
        ir=cv2.resize(ir,(960,540),interpolation=cv2.INTER_AREA)
        current={}
        for arm,data in streams.items():
            lo,hi=map(int,data['offsets'][position:position+2])
            current[arm]=point_groups(rgb,ir,data['boxes'][lo:hi]*scale,maximum=cfg['points_per_observation'])
            outputs[arm]['quality'][lo:hi]=current[arm]['quality']
        for lag in cfg['lags']:
            if frame-lag not in history:continue
            previous=history[frame-lag]
            result=frame_correspondences(previous['rgb'],rgb,previous['regions'],current)
            for arm,output in outputs.items():
                vo0,vo1=map(int,output[f'vote_offsets_{lag}'][position:position+2])
                co0,co1=map(int,output[f'count_offsets_{lag}'][position:position+2])
                for source_mode in MODES:
                    output[f'total_{lag}_{source_mode}'][co0:co1]=previous['regions'][arm]['groups'][source_mode]['total']
                    for target_mode in MODES:
                        matrix,good=result[(arm,source_mode,target_mode)]
                        output[f'vote_{lag}_{source_mode}_{target_mode}'][vo0:vo1]=matrix.ravel()
                        output[f'good_{lag}_{source_mode}'][co0:co1]=good
        history[frame]=dict(rgb=rgb,regions=current)
        history={f:r for f,r in history.items() if frame-f<max(cfg['lags'])}
        if position in [0,len(streams['new']['frame_ids'])//2,len(streams['new']['frame_ids'])-1]:
            sampled_inputs[str(frame)]=dict(rgb_sha256=sha256(rgb_path),new_IR_sha256=sha256(ir_path))
        if frame%100==0 or position+1==len(streams['new']['frame_ids']):
            progress=dict(frame=frame,frames=len(streams['new']['frame_ids']),seconds=time.perf_counter()-started,device='CPU classical colour LK',lag_frames=list(cfg['lags']))
            dump(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    saved={}
    for arm,arrays in outputs.items():
        target=out/(arm+'.npz');temporary=out/(arm+'.tmp.npz')
        np.savez_compressed(temporary,**arrays);os.replace(temporary,target)
        saved[arm]=dict(path=str(target),sha256=sha256(target),bytes=target.stat().st_size,frames=len(arrays['frame_ids']))
    report=dict(status='SMOKE_COMPLETE' if a.max_frames else 'COMPLETE',split=a.split,sequence=a.sequence,
                frames=len(streams['new']['frame_ids']),outputs=saved,proposal_sources=sources,source_hashes=source_hashes,
                seconds=time.perf_counter()-started,sampled_input_hashes=sampled_inputs,
                future_input=False,GT_read=False,past_image_buffer_frames=max(cfg['lags'])+1,
                input_semantics='Current actual RGB/new DiffV2IR and declared earlier RGB/new DiffV2IR; no previous output is rewritten')
    dump(out/'complete.json',report);print('CORRESPONDENCE_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
