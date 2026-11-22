#!/usr/bin/env python3
"""Train-only paired geometric evidence; no Dev/Test fitting images or labels."""
import argparse
import configparser
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
from yolox.tracker.foreground_transport import point_groups,frame_correspondences,MODES
from yolox.tracker.association_features import pair_features
from yolox.tracker.perceptual_features import clipped_crop,foreground,hog,color_histogram,letterbox_tensor_array
from yolox.tracker.matching import ious,linear_assignment


def proposals(group,rng,sigma,width,height,dropout):
    group=group[rng.random(len(group))>=dropout]
    boxes=group[:,2:6].astype(np.float32).copy();boxes[:,2:]+=boxes[:,:2]
    size=boxes[:,2:]-boxes[:,:2];centre=(boxes[:,:2]+boxes[:,2:])/2
    centre+=rng.normal(0,sigma,centre.shape)*size;size*=np.exp(rng.normal(0,sigma,size.shape))
    boxes=np.c_[centre-size/2,centre+size/2].astype(np.float32)
    background=[]
    for _ in range(2):
        if len(size):extent=size[int(rng.integers(len(size)))].copy()
        else:extent=np.array([50.,30.])
        center=np.array([rng.uniform(0,width),rng.uniform(0,height)])
        background.append(np.r_[center-extent/2,center+extent/2])
    boxes=np.r_[boxes,np.asarray(background,np.float32)]
    boxes[:,[0,2]]=boxes[:,[0,2]].clip(0,width-1);boxes[:,[1,3]]=boxes[:,[1,3]].clip(0,height-1)
    boxes[:,2:]=np.maximum(boxes[:,2:],boxes[:,:2]+1)
    return boxes


def describe(rgb,ir,boxes,model):
    n=len(boxes);data={k:np.zeros((n,dim),np.float16) for k,dim in [('embedding',512),('rgb_mask_embedding',512),('ir_mask_embedding',512),('color',128),('rgb_hog',144),('ir_hog',144),('quality',2)]}
    data['boxes']=boxes;views=[];places=[]
    for i,box in enumerate(boxes):
        r=clipped_crop(rgb,box,.05);g=cv2.cvtColor(r,cv2.COLOR_BGR2GRAY);t=clipped_crop(ir,box,.05)
        rm,rq=foreground(g);tm,tq=foreground(t)
        data['quality'][i]=[rq,tq];data['color'][i]=color_histogram(r)
        data['rgb_hog'][i]=hog(g);data['ir_hog'][i]=hog(t)
        for field,mask in [('embedding',None),('rgb_mask_embedding',rm),('ir_mask_embedding',tm)]:
            views.append(letterbox_tensor_array(rgb,box,mask));places.append((field,i))
    with torch.inference_mode():
        for lo in range(0,len(views),64):
            result=model(torch.from_numpy(np.stack(views[lo:lo+64])).cuda().half()).cpu().numpy()
            for value,(field,index) in zip(result,places[lo:lo+64]):data[field][index]=value
    return data


def main():
    p=argparse.ArgumentParser();p.add_argument('--sequence',required=True);p.add_argument('--smoke-sources',type=int,default=0);args=p.parse_args()
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in {'2','3'}:raise RuntimeError('Only physical GPU2 or3')
    config_path=ROOT/'configs/perceptual_pair_calibration_20261123.json';config=json.loads(config_path.read_text())
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());protocol=json.loads((ROOT/plan['protocol']).read_text())
    if args.sequence not in protocol['train']:raise ValueError('Only frozen Train videos allowed')
    gt_root=Path(plan['evaluation']['GT_root'])/'train'/args.sequence
    gt_path=gt_root/'gt/gt.txt'
    if sha256(gt_path)!=protocol['derived_gt_sha256']['train/'+args.sequence]:raise ValueError('Train GT mismatch')
    gt=np.loadtxt(gt_path,delimiter=',',ndmin=2);gt=gt[(gt[:,6]>0)&(gt[:,7]==1)]
    ini=configparser.ConfigParser();ini.read(gt_root/'seqinfo.ini');length=ini.getint('Sequence','seqLength')
    width=ini.getint('Sequence','imWidth');height=ini.getint('Sequence','imHeight')
    frames=np.unique(np.linspace(1,length-8,config['source_frames_per_train_video'],dtype=int))
    out=Path(plan['work_root'])/'association_train'/args.sequence
    if args.smoke_sources:out=out/f'smoke_{args.smoke_sources}';frames=frames[:args.smoke_sources]
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise ValueError('Completed bank cannot be overwritten')
    if shutil.disk_usage(out).free<40*2**30:raise RuntimeError('Work disk reserve')
    source_hashes={name:sha256(ROOT/path) for name,path in dict(extractor='tools/extract_association_train_pairs.py',features='yolox/tracker/association_features.py',transport='yolox/tracker/foreground_transport.py').items()}
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35,0)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    ckpt=Path('/data3/liuyeqiang/IR-MFT_assets/cfp/phase1/reid_rgb_seed0/epoch12.pt')
    c=torch.load(ckpt,map_location='cpu',weights_only=False);model=AppearanceEncoder();model.load_state_dict(c['model'],strict=True);del c
    model=model.cuda().half().eval();rng=np.random.default_rng(20261123+protocol['train'].index(args.sequence))
    arrays={k:[] for k in ['base','reference','rgb','ir','IR_available','label','lag','query']};query=0;start=time.perf_counter()
    groups={int(f):gt[gt[:,0]==f] for f in range(1,length+1)}
    def load(frame,dropout):
        rgb_path=Path(plan['data_root'])/'train'/args.sequence/'visible'/f'{frame:06d}.jpg'
        ir_path=Path(plan['generated_ir_root'])/'train'/args.sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
        rgb=cv2.imread(str(rgb_path));ir=cv2.imread(str(ir_path),cv2.IMREAD_UNCHANGED)
        if rgb is None or ir is None or ir.dtype!=np.uint8 or ir.ndim!=2 or rgb.shape[:2]!=ir.shape:raise ValueError('Invalid actual Train pair')
        small_rgb=cv2.resize(rgb,(960,540),interpolation=cv2.INTER_AREA);small_ir=cv2.resize(ir,(960,540),interpolation=cv2.INTER_AREA)
        streams={};regions={}
        for name,sigma in [('small',.03),('large',.08)]:
            boxes=proposals(groups[frame],rng,sigma,width,height,dropout)
            data=describe(rgb,ir,boxes,model)
            data['label']=np.full(len(boxes),-1,int)
            group=groups[frame];gb=group[:,2:6].copy();gb[:,2:]+=gb[:,:2]
            matched,_,_=linear_assignment(1-ious(gb,boxes),.5)
            for gi,pi in matched:data['label'][pi]=int(group[gi,1])
            region=point_groups(small_rgb,small_ir,boxes*np.array([960/width,540/height]*2,np.float32))
            data['point_quality']=region['quality'];streams[name]=data;regions[name]=region
        return dict(rgb=small_rgb,streams=streams,regions=regions)
    for position,frame in enumerate(frames):
        previous=load(int(frame),0.)
        for lag in config['lags']:
            current=load(int(frame)+lag,config['target_dropout'])
            results=frame_correspondences(previous['rgb'],current['rgb'],previous['regions'],current['regions'])
            for arm,past in previous['streams'].items():
                target=current['streams'][arm];matrices={(s,t):results[(arm,s,t)][0] for s in MODES for t in MODES}
                good={s:results[(arm,s,'rgb')][1] for s in MODES};total={s:previous['regions'][arm]['groups'][s]['total'] for s in MODES}
                b,extra_rgb,ref=pair_features(past,target,matrices,good,total,lag,'rgb')
                _,extra_ir,ref2=pair_features(past,target,matrices,good,total,lag,'ir');np.testing.assert_array_equal(ref,ref2)
                label=(past['label'][:,None]==target['label'][None])&(past['label'][:,None]>=0)&(target['label'][None]>=0)
                keep=past['label']>=0;edges=int(keep.sum())*len(target['boxes'])
                available=(past['point_quality'][:,1:2]>=.15)&(target['point_quality'][None,:,1]>=.15)
                ids=np.repeat(np.arange(query,query+int(keep.sum())),len(target['boxes']));query+=int(keep.sum())
                for name,value in [('base',b[keep].reshape(-1,23)),('reference',ref[keep].reshape(-1,17)),('rgb',extra_rgb[keep].reshape(-1,17)),('ir',extra_ir[keep].reshape(-1,17)),('IR_available',available[keep].ravel()),('label',label[keep].ravel()),('lag',np.full(edges,lag,np.uint8)),('query',ids)]:arrays[name].append(value)
        if (position+1)%8==0 or position+1==len(frames):
            progress=dict(source_frames=position+1,total_source_frames=len(frames),pair_rows=sum(len(x) for x in arrays['label']),seconds=time.perf_counter()-start,physical_gpu=physical)
            dump(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    saved={k:np.concatenate(v).astype(np.float16 if k in {'base','reference','rgb','ir'} else np.uint8 if k in {'IR_available','label','lag'} else np.int64) for k,v in arrays.items()}
    path=out/'pairs.npz';temporary=out/'pairs.tmp.npz';np.savez_compressed(temporary,**saved);os.replace(temporary,path)
    report=dict(status='SMOKE_COMPLETE' if args.smoke_sources else 'COMPLETE_TRAIN_ONLY',sequence=args.sequence,source_frames=len(frames),pair_rows=len(saved['label']),positive_rows=int(saved['label'].sum()),queries=query,GT_sha256=sha256(gt_path),config_sha256=sha256(config_path),checkpoint_sha256=sha256(ckpt),bank_sha256=sha256(path),bytes=path.stat().st_size,source_hashes=source_hashes,physical_gpu=physical,Dev_rows=0,Test_rows=0,seconds=time.perf_counter()-start)
    dump(out/'complete.json',report);print('ASSOCIATION_TRAIN_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
