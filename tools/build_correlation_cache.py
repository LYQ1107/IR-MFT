#!/usr/bin/env python3
"""Train-only normalization and shared-frozen-backbone feature cache."""
import argparse
import configparser
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np

from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.tracker.foveated_regions import exact_half_preproc


def read_pair(plan,sequence,frame):
    rgb_path=Path(plan['data_root'])/'train'/sequence/'visible'/f'{frame:06d}.jpg'
    ir_path=Path(plan['generated_ir_root'])/'train'/sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
    rgb,ir=cv2.imread(str(rgb_path)),cv2.imread(str(ir_path),-1)
    if rgb is None or ir is None or ir.ndim!=2 or ir.dtype!=np.uint8 or rgb.shape[:2]!=ir.shape:
        raise ValueError('Actual Train RGB/November IR required')
    return rgb,ir


def auxiliary_input(image,size,mean,std):
    ratio=min(size[0]/image.shape[0],size[1]/image.shape[1])
    h,w=int(image.shape[0]*ratio),int(image.shape[1]*ratio)
    resized=cv2.resize(image,(w,h),interpolation=cv2.INTER_LINEAR)
    table=((np.arange(256,dtype=np.float64)/255-mean)/std).astype(np.float32).astype(np.float16)
    value=np.zeros((3,*size),np.float16)
    value[:,:h,:w]=table[resized][None]
    return value,ratio


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['statistics','merge','features'],required=True)
    p.add_argument('--sequence');p.add_argument('--max-frames',type=int,default=0);a=p.parse_args()
    cv2.setNumThreads(1)
    main_path=ROOT/'configs/perceptual_mft_20261122.json'
    plan=json.loads(main_path.read_text());work=Path(plan['work_root'])
    config_path=ROOT/'configs/perceptual_correlation_repair_20261123.json';config=json.loads(config_path.read_text())
    if sha256(main_path)!=config['original_main_plan_sha256']:raise ValueError('Original task setting changed')
    protocol=json.loads((ROOT/plan['protocol']).read_text());base=work/'correlation_cache';base.mkdir(exist_ok=True)
    if a.stage=='merge':
        destination=base/'calibration.json'
        if destination.exists():raise ValueError('Train calibration already frozen')
        inputs={seq:json.loads((base/seq/'statistics.json').read_text())for seq in protocol['train']}
        for info in inputs.values():
            if info['config_sha256']!=sha256(config_path)or info['frames']!=config['Train_frames_per_video']:
                raise ValueError('Complete matched Train calibration required')
        stats={}
        for cue in ['gray','ir']:
            count=sum(x['moments'][cue]['count']for x in inputs.values())
            mean=sum(x['moments'][cue]['sum']for x in inputs.values())/count/255
            second=sum(x['moments'][cue]['sum_squares']for x in inputs.values())/count/255**2
            stats[cue]=dict(mean=mean,std=max(float(np.sqrt(max(second-mean**2,0))),.001),pixels=count)
        dump(destination,dict(status='COMPLETE_TRAIN_ONLY_CALIBRATION',config_sha256=sha256(config_path),statistics=stats,
             sequences=protocol['train'],source_statistics_sha256={seq:sha256(base/seq/'statistics.json')for seq in inputs},
             Dev_rows=0,Test_rows=0))
        print(json.dumps(dict(status='TRAIN_CALIBRATION_FROZEN',statistics=stats)));return
    if a.sequence not in protocol['train']:raise ValueError('Only original Train6')
    gt_root=Path(plan['evaluation']['GT_root'])/'train'/a.sequence
    ini=configparser.ConfigParser();ini.read(gt_root/'seqinfo.ini');length=ini.getint('Sequence','seqLength')
    frames=np.unique(np.linspace(1,length,config['Train_frames_per_video'],dtype=int))
    if len(frames)!=config['Train_frames_per_video']:raise ValueError('Unequal frame count')
    directory=base/a.sequence;directory.mkdir(exist_ok=True)
    if a.stage=='statistics':
        if a.max_frames:raise ValueError('Do not fit normalization on an accidental smoke subset')
        path=directory/'statistics.json'
        if path.exists():raise ValueError('Sequence statistics already complete')
        moments={cue:dict(count=0,sum=0,sum_squares=0)for cue in ['gray','ir']};started=time.perf_counter()
        for index,frame in enumerate(frames):
            rgb,ir=read_pair(plan,a.sequence,int(frame))
            for cue,image in [('gray',cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)),('ir',ir)]:
                pixels=image.astype(np.int64);moments[cue]['count']+=pixels.size
                moments[cue]['sum']+=int(pixels.sum());moments[cue]['sum_squares']+=int((pixels*pixels).sum())
            if (index+1)%32==0:print(json.dumps(dict(sequence=a.sequence,statistics_frames=index+1,total=len(frames))),flush=True)
        dump(path,dict(status='COMPLETE_TRAIN_PIXEL_STATISTICS',sequence=a.sequence,frames=len(frames),frame_ids=frames.tolist(),
            moments=moments,config_sha256=sha256(config_path),GT_labels_read=False,Dev_rows=0,Test_rows=0,
            source_sha256=sha256(__file__),seconds=time.perf_counter()-started));return
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in {'2','3'}:raise ValueError('Use only physical GPU2 or3')
    calibration_path=base/'calibration.json';calibration=json.loads(calibration_path.read_text())
    if calibration['config_sha256']!=sha256(config_path):raise ValueError('Train normalization plan mismatch')
    if a.max_frames:frames=frames[:a.max_frames]
    out=directory/(f'feature_smoke_{a.max_frames}'if a.max_frames else'features');out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():raise ValueError('Feature cache already complete')
    if shutil.disk_usage(out).free<40*2**30:raise RuntimeError('Keep independent-disk reserve')
    import torch
    from yolox.exp import get_exp
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.35,0)
    checkpoint=Path(config['source_RGB_checkpoint'])
    if sha256(checkpoint)!=config['source_RGB_sha256']:raise ValueError('Shared RGB source changed')
    model=get_exp(str(ROOT/'exps/yolox_mft26ir_visible.py'),None).get_model()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(saved['model'],strict=True);del saved
    encoder=model.backbone.cuda().half().eval().requires_grad_(False);del model
    maps={};paths={};image_shapes=[];ratios=[];started=time.perf_counter()
    with torch.inference_mode():
        for index,frame in enumerate(frames):
            rgb,ir=read_pair(plan,a.sequence,int(frame));image_shapes.append(rgb.shape[:2])
            value,ratio=exact_half_preproc(rgb,config['training_input_hw'],(.485,.456,.406),(.229,.224,.225))
            ratios.append(ratio);views={'rgb':value}
            for cue,image in [('gray',cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)),('ir',ir)]:
                norm=calibration['statistics'][cue]
                views[cue],_=auxiliary_input(image,config['training_input_hw'],norm['mean'],norm['std'])
            for cue,value in views.items():
                features=encoder(torch.from_numpy(value[None]).cuda())
                for level,feature in enumerate(features):
                    key=f'{cue}_P{level+3}';shape=(len(frames),*feature.shape[1:])
                    if key not in maps:
                        needed=int(np.prod(shape))*2
                        used=sum(p.stat().st_size for p in work.rglob('*')if p.is_file())
                        if used+needed>64*2**30:raise RuntimeError('64GiB task work-cache cap')
                        temporary=out/(key+'.tmp.npy');paths[key]=temporary
                        maps[key]=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float16,shape=shape)
                    maps[key][index]=feature[0].cpu().numpy()
            if (index+1)%16==0 or index+1==len(frames):
                progress=dict(sequence=a.sequence,frames=index+1,total=len(frames),physical_gpu=physical,seconds=time.perf_counter()-started)
                dump(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    outputs={}
    for key,value in maps.items():
        value.flush();target=out/(key+'.npy');os.replace(paths[key],target)
        outputs[key]=dict(path=str(target),shape=list(value.shape),bytes=target.stat().st_size,sha256=sha256(target))
    gt_path=gt_root/'gt/gt.txt'
    if sha256(gt_path)!=protocol['derived_gt_sha256']['train/'+a.sequence]:raise ValueError('Original Train GT changed')
    gt=np.loadtxt(gt_path,delimiter=',',ndmin=2);gt=gt[(gt[:,6]>0)&(gt[:,7]==1)]
    targets=np.zeros((len(frames),500,5),np.float32)
    for index,frame in enumerate(frames):
        boxes=gt[gt[:,0]==frame,2:6].astype(np.float32).copy();boxes[:,:2]+=boxes[:,2:]/2
        boxes*=ratios[index]
        if len(boxes)>500:raise ValueError('Original maximum label count exceeded')
        targets[index,:len(boxes),1:]=boxes
    target_path=out/'targets.npy';np.save(target_path,targets)
    dump(out/'complete.json',dict(status='SMOKE_COMPLETE'if a.max_frames else'COMPLETE_TRAIN_FROZEN_FPN',sequence=a.sequence,
        frames=len(frames),frame_ids=frames.tolist(),image_shapes=image_shapes,ratios=ratios,outputs=outputs,
        targets=dict(path=str(target_path),sha256=sha256(target_path)),GT_sha256=sha256(gt_path),config_sha256=sha256(config_path),
        calibration_sha256=sha256(calibration_path),source_RGB_sha256=config['source_RGB_sha256'],encoder_shared_and_frozen=True,
        source_sha256=sha256(__file__),physical_gpu=physical,Dev_rows=0,Test_rows=0,seconds=time.perf_counter()-started))


if __name__=='__main__':main()
