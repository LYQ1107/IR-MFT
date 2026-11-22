#!/usr/bin/env python3
"""Frozen RGB crop detector with current RGB/new-IR attention controls."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch

from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.tracker.foveated_regions import exact_half_preproc as preproc
from yolox.exp import get_exp
from yolox.tracker.foveated_regions import coarse_attention,attention_windows,uniform_windows,integrate_crops,working_region
from yolox.utils import postprocess


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['dev','test'],default='dev')
    parser.add_argument('--sequence',required=True);parser.add_argument('--max-frames',type=int,default=0);args=parser.parse_args()
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in {'2','3'}:raise ValueError('Only physical GPUs2/3')
    main_plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(main_plan['work_root'])
    config_path=ROOT/'configs/perceptual_foveated_regions_20261123.json';config=json.loads(config_path.read_text())
    protocol=json.loads((ROOT/main_plan['protocol']).read_text())
    if args.sequence not in protocol['dev' if args.split=='dev' else 'historical_test']:raise ValueError('Outside frozen split')
    if args.split=='test' and not (work/'foveated_selection_frozen.json').exists():raise ValueError('Freeze complete Dev selection before foveated Test')
    directory=work/'foveated_region_proposals'/args.split/args.sequence
    if args.max_frames:directory=directory/f'smoke_{args.max_frames}'
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():raise ValueError('Completed foveated proposals exist')
    if shutil.disk_usage(directory).free<40*2**30:raise RuntimeError('Keep work filesystem reserve')
    model_path=Path(config['primary_crop_detector'])
    if sha256(model_path)!=config['primary_crop_detector_sha256']:raise ValueError('Frozen crop detector changed')
    print('FOVEATED_SOURCE_VERIFIED',str(model_path),flush=True)
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.cuda.set_per_process_memory_fraction(.45,0)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    exp=get_exp(str(ROOT/'exps/yolox_mft26ir_visible.py'),None);model=exp.get_model()
    checkpoint=torch.load(model_path,map_location='cpu',weights_only=False)
    if checkpoint.get('selection',{}).get('selected_full_checkpoint_sha256')!=config['original_full_crop_detector_sha256']:
        raise ValueError('Compact model is not the registered original EMA conversion')
    model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
    model=model.cuda().half().eval();sources={};base={}
    print('FOVEATED_MODEL_READY',physical,flush=True)
    for arm,key in [('rgb','rgb'),('new','new_diffv2ir')]:
        path=Path(main_plan['frozen_detection_sources'][key+'_'+args.split])/'detections'/(args.sequence+'.npz')
        with np.load(path) as bank:base[arm]={name:bank[name].copy() for name in ['frame_ids','offsets','boxes','scores']}
        sources[arm]=dict(path=str(path),sha256=sha256(path))
    np.testing.assert_array_equal(base['rgb']['frame_ids'],base['new']['frame_ids'])
    frames=base['rgb']['frame_ids'];limit=min(len(frames),args.max_frames or len(frames))
    scales=config.get('attention_scales',{'half':.5})
    policies=[(cue+'_'+name,count) for cue in ['rgb','ir'] for name in scales for count in [2,4]]+[('uniform',4)]
    buffers={f'{arm}_{cue}_k{count}_{fusion}':[] for arm in ['rgb','new']
             for cue,count in policies for fusion in ['refine','nms']}
    roi_offsets=[0];roi_boxes=[];roi_scores=[];roi_window=[];windows_saved=[]
    started=time.perf_counter();forward_crops=0
    def load_current_images(frame):
        rgb_path=Path(main_plan['data_root'])/('train' if args.split=='dev' else 'test')/args.sequence/'visible'/f'{frame:06d}.jpg'
        ir_path=Path(main_plan['generated_ir_root'])/args.split/args.sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
        return cv2.imread(str(rgb_path)),cv2.imread(str(ir_path),cv2.IMREAD_UNCHANGED)
    # Bounded I/O prefetch does not enter the model: each independent eval
    # forward consumes only the image pair whose output timestamp is current.
    pool=ThreadPoolExecutor(max_workers=2)
    pending={frame:pool.submit(load_current_images,frame) for frame in range(1,min(limit,2)+1)}
    with torch.inference_mode():
        for frame in range(1,limit+1):
            rgb,ir=pending.pop(frame).result()
            if frame+2<=limit:pending[frame+2]=pool.submit(load_current_images,frame+2)
            if rgb is None or ir is None or ir.ndim!=2 or ir.dtype!=np.uint8 or rgb.shape[:2]!=ir.shape:raise ValueError('Actual current RGB/new-IR image missing')
            a,b=map(int,base['rgb']['offsets'][frame-1:frame+1]);boxes=base['rgb']['boxes'][a:b];scores=base['rgb']['scores'][a:b]
            support=working_region(rgb,boxes,scores)
            rgb_heat=coarse_attention(rgb,boxes,scores,rgb.shape,support=support)
            ir_heat=rgb_heat if float(ir.std())<1e-6 else coarse_attention(ir,boxes,scores,rgb.shape,support=support)
            selected={cue+'_'+name:attention_windows(heat,rgb.shape,4,fraction=fraction)
                      for cue,heat in [('rgb',rgb_heat),('ir',ir_heat)] for name,fraction in scales.items()}
            selected['uniform']=uniform_windows(rgb.shape)
            windows=list(dict.fromkeys(window for values in selected.values() for window in values))
            detected={};forward_crops+=len(windows)
            for lo in range(0,len(windows),4):
                batch=[];ratios=[]
                for x1,y1,x2,y2 in windows[lo:lo+4]:
                    value,ratio=preproc(rgb[y1:y2,x1:x2],config['network_size'],(.485,.456,.406),(.229,.224,.225))
                    batch.append(value);ratios.append(ratio)
                outputs=postprocess(model(torch.from_numpy(np.stack(batch)).cuda().half()),1,.1,.7)
                for window,ratio,output in zip(windows[lo:lo+4],ratios,outputs):
                    x1,y1,x2,y2=window
                    if output is None:detection=np.empty((0,5),np.float32)
                    else:
                        raw=output.float().cpu().numpy();detection=np.c_[raw[:,:4]/ratio,raw[:,4]*raw[:,5]].astype(np.float32)
                        # An interior crop boundary cannot certify a complete fish.
                        epsilon=2/ratio;valid=np.ones(len(detection),bool)
                        if x1>0:valid&=detection[:,0]>epsilon
                        if y1>0:valid&=detection[:,1]>epsilon
                        if x2<rgb.shape[1]:valid&=detection[:,2]<x2-x1-epsilon
                        if y2<rgb.shape[0]:valid&=detection[:,3]<y2-y1-epsilon
                        detection=detection[valid];detection[:,:4]+=np.array([x1,y1,x1,y1],np.float32)
                    detected[window]=detection
                    roi_boxes.extend(detection[:,:4]);roi_scores.extend(detection[:,4]);roi_window.extend([len(windows_saved)]*len(detection))
                    windows_saved.append((frame,*window));roi_offsets.append(len(roi_boxes))
            for arm in ['rgb','new']:
                a,b=map(int,base[arm]['offsets'][frame-1:frame+1]);original=np.c_[base[arm]['boxes'][a:b],base[arm]['scores'][a:b]]
                for cue,count in policies:
                    views=[]
                    for window in selected[cue][:count]:
                        detection=detected[window].copy()
                        # Keep low-score ROI observations for matched localization,
                        # but do not add a weak new object without other support.
                        overlap=np.zeros(len(detection),bool)
                        if len(original) and len(detection):
                            from yolox.tracker.matching import ious
                            overlap=ious(original[:,:4],detection[:,:4]).max(0)>=.5
                        detection=detection[overlap|(detection[:,4]>=.6)]
                        resolution=rgb.shape[1]/(window[2]-window[0]);views.append((detection,resolution))
                    for fusion in ['refine','nms']:
                        buffers[f'{arm}_{cue}_k{count}_{fusion}'].append(integrate_crops(original,views,refine=fusion=='refine'))
            if frame%50==0 or frame==limit:
                progress=dict(frame=frame,frames=limit,forward_crops=forward_crops,seconds=time.perf_counter()-started,physical_gpu=physical,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
                dump(directory/'progress.json',progress);print(json.dumps(progress),flush=True)
    pool.shutdown(wait=True)
    outputs={}
    for name,values in buffers.items():
        detections=np.concatenate(values).reshape(-1,5);path=directory/(name+'.npz');temporary=directory/(name+'.tmp.npz')
        np.savez_compressed(temporary,frame_ids=np.arange(1,limit+1),offsets=np.r_[0,np.cumsum([len(v) for v in values])],boxes=detections[:,:4],scores=detections[:,4]);os.replace(temporary,path)
        outputs[name]=dict(path=str(path),sha256=sha256(path),frames=limit,detections=len(detections),bytes=path.stat().st_size)
    roi_path=directory/'raw_roi_observations.npz';np.savez_compressed(roi_path,windows=np.asarray(windows_saved,np.int32),offsets=np.asarray(roi_offsets,np.int64),boxes=np.asarray(roi_boxes,np.float32).reshape(-1,4),scores=np.asarray(roi_scores,np.float32))
    dump(directory/'complete.json',dict(status='SMOKE_COMPLETE' if args.max_frames else 'COMPLETE_FOVEATED_CURRENT_OBSERVATIONS',sequence=args.sequence,split=args.split,frames=limit,outputs=outputs,sources=sources,config_sha256=sha256(config_path),crop_detector_sha256=config['primary_crop_detector_sha256'],source_sha256=sha256(__file__),attention_source_sha256=sha256(ROOT/'yolox/tracker/foveated_regions.py'),raw_roi_sha256=sha256(roi_path),GT_read=False,future_frame_input=False,physical_gpu=physical,forward_crops=forward_crops,seconds=time.perf_counter()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,runtime_scope='Shared production of all20 policies, cached global detector observations; not per-method end-to-end FPS.'))


if __name__=='__main__':main()
