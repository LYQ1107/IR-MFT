#!/usr/bin/env python3
"""Reuse unchanged observations and recompute only genuinely new evidence."""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch

from tools.extract_perceptual_mft_features import AppearanceEncoder
from tools.extract_association_train_pairs import describe
from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.tracker.foreground_transport import MODES,point_groups,frame_correspondences
from yolox.tracker.perceptual_features import local_transport


def main():
    p=argparse.ArgumentParser();p.add_argument('--sequence',choices=['BT-005','SN-013'],required=True);a=p.parse_args()
    physical=os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in {'2','3'}:raise ValueError('Only physical GPU2/3')
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.25,0)
    main_path=ROOT/'configs/perceptual_mft_20261122.json';main=json.loads(main_path.read_text());old_work=Path(main['work_root'])
    plan_path=ROOT/'configs/perceptual_priority_backend_20261123.json';plan=json.loads(plan_path.read_text());work=Path(plan['work_root'])
    if plan['priority_backend_registration']['original_plan_sha256']!=sha256(main_path):raise ValueError('Original setting changed')
    out=work/'features/dev'/a.sequence;point_out=work/'transport/dev'/a.sequence
    out.mkdir(parents=True,exist_ok=True);point_out.mkdir(parents=True,exist_ok=True)
    if(out/'complete.json').exists()or(point_out/'complete.json').exists():raise ValueError('Completed evidence exists')
    source=Path(plan['frozen_detection_sources']['new_diffv2ir_dev'])/'detections'/(a.sequence+'.npz')
    with np.load(source)as z:proposal={key:z[key]for key in z.files}
    old_path=old_work/'features/dev'/a.sequence/'new.npz'
    old_meta=json.loads((old_path.parent/'complete.json').read_text())
    if old_meta['outputs']['new']['sha256']!=sha256(old_path):raise ValueError('Original feature cache changed')
    with np.load(old_path)as z:old={key:z[key]for key in z.files}
    mapping=proposal['original_index'];known=mapping>=0;added=np.flatnonzero(~known)
    np.testing.assert_array_equal(proposal['boxes'][known],old['boxes'][mapping[known]])
    data={key:proposal[key]for key in ['frame_ids','offsets','boxes','scores']}
    for key,value in old.items():
        if key in data:continue
        if len(value)!=len(old['boxes']):raise ValueError('Unexpected frozen feature schema')
        data[key]=np.zeros((len(mapping),*value.shape[1:]),dtype=value.dtype)
        data[key][known]=value[mapping[known]]
    def read(frame):
        rgb=cv2.imread(str(Path(main['data_root'])/'train'/a.sequence/'visible'/f'{frame:06d}.jpg'))
        ir=cv2.imread(str(Path(main['generated_ir_root'])/'dev'/a.sequence/'infrared_diffv2ir'/f'{frame:06d}.png'),-1)
        if rgb is None or ir is None or ir.ndim!=2 or rgb.shape[:2]!=ir.shape:raise ValueError('Actual pair missing')
        return rgb,ir
    added_frames=np.searchsorted(proposal['offsets'],added,side='right')
    teacher=Path('/data3/liuyeqiang/IR-MFT_assets/cfp/phase1/reid_rgb_seed0/epoch12.pt')
    if sha256(teacher)!=old_meta['appearance_sha256']:raise ValueError('Frozen appearance source mismatch')
    started=time.perf_counter()
    if len(added):
        model=AppearanceEncoder();saved=torch.load(teacher,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'],strict=True);del saved;model=model.cuda().half().eval().requires_grad_(False)
        for frame in np.unique(added_frames):
            indices=added[added_frames==frame];rgb,ir=read(int(frame))
            described=describe(rgb,ir,proposal['boxes'][indices],model)
            for key in described:
                if key!='boxes':data[key][indices]=described[key]
            if frame<len(proposal['frame_ids']):
                next_rgb,next_ir=read(int(frame)+1)
                for key,first,second in [('flow_rgb',cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY),cv2.cvtColor(next_rgb,cv2.COLOR_BGR2GRAY)),
                                          ('flow_ir',ir,next_ir)]:
                    first=cv2.resize(first,(960,540),interpolation=cv2.INTER_AREA);second=cv2.resize(second,(960,540),interpolation=cv2.INTER_AREA)
                    data[key][indices]=local_transport(first,second,proposal['boxes'][indices],rgb.shape[:2])
        del model;torch.cuda.empty_cache()
    for key in old:
        if key not in {'frame_ids','offsets','boxes','scores'}:
            np.testing.assert_array_equal(data[key][known],old[key][mapping[known]])
    feature_path=out/'new.npz';np.savez_compressed(feature_path,**data)
    control=old_work/'features/dev'/a.sequence/'rgb.npz';(out/'rgb.npz').symlink_to(control)
    feature_outputs={alias:dict(path=str(out/(alias+'.npz')),sha256=sha256(out/(alias+'.npz')),
                                frames=len(data['frame_ids']))for alias in ['new','rgb']}
    feature_report=dict(status='COMPLETE',split='dev',sequence=a.sequence,outputs=feature_outputs,
        source_sha256=sha256(__file__),plan_sha256=sha256(plan_path),GT_read=False,added_observations=len(added),
        original_descriptors_bit_exact=True,appearance_sha256=sha256(teacher),
        frozen_original_manifest_sha256=sha256(old_path.parent/'complete.json'),
        flow_semantics='Copied/delayed flow at source rowt uses t,t+1 only while processing t+1, identical to the original feature convention.')
    dump(out/'complete.json',feature_report)
    old_points_path=old_work/'transport/dev'/a.sequence/'new.npz'
    old_point_meta=json.loads((old_points_path.parent/'complete.json').read_text())
    if old_point_meta['outputs']['new']['sha256']!=sha256(old_points_path):raise ValueError('Original point cache changed')
    with np.load(old_points_path)as z:old_points={key:z[key]for key in z.files}
    other_path=Path(main['frozen_detection_sources']['rgb_dev'])/'detections'/(a.sequence+'.npz')
    with np.load(other_path)as z:other={key:z[key]for key in ['frame_ids','offsets','boxes','scores']}
    @lru_cache(maxsize=24)
    def regions(frame):
        rgb,ir=read(frame);scale=np.array([960/rgb.shape[1],540/rgb.shape[0]]*2,np.float32)
        rgb=cv2.resize(rgb,(960,540),interpolation=cv2.INTER_AREA);ir=cv2.resize(ir,(960,540),interpolation=cv2.INTER_AREA)
        packets={}
        for alias,bank in [('new',proposal),('rgb',other)]:
            lo,hi=bank['offsets'][frame-1:frame+1]
            packets[alias]=point_groups(rgb,ir,bank['boxes'][lo:hi]*scale,maximum=20)
        return dict(rgb=rgb,regions=packets)
    counts=np.diff(proposal['offsets']);old_counts=np.diff(old['offsets']);length=len(counts)
    point=dict(frame_ids=proposal['frame_ids'],detection_offsets=proposal['offsets'],quality=np.zeros((len(mapping),2),np.float16))
    point['quality'][known]=old_points['quality'][mapping[known]]
    for frame in np.unique(added_frames):
        indices=added[added_frames==frame];start=proposal['offsets'][frame-1]
        point['quality'][indices]=regions(int(frame))['regions']['new']['quality'][indices-start]
    changed=set(int(x)for x in added_frames);recomputed=0;copied=0
    for lag in [1,2,4,8]:
        previous=np.r_[np.zeros(lag,int),counts[:-lag]];elements=counts*previous
        point[f'vote_offsets_{lag}']=np.r_[0,np.cumsum(elements)];point[f'count_offsets_{lag}']=np.r_[0,np.cumsum(previous)]
        for source_mode in MODES:
            for field in ['good','total']:point[f'{field}_{lag}_{source_mode}']=np.zeros(int(previous.sum()),np.uint8)
            for target in MODES:point[f'vote_{lag}_{source_mode}_{target}']=np.zeros(int(elements.sum()),np.float16)
        for position in range(lag,length):
            frame=position+1;past_frame=frame-lag
            vo0,vo1=point[f'vote_offsets_{lag}'][position:position+2];co0,co1=point[f'count_offsets_{lag}'][position:position+2]
            old_vo0,old_vo1=old_points[f'vote_offsets_{lag}'][position:position+2]
            old_co0,old_co1=old_points[f'count_offsets_{lag}'][position:position+2]
            affected=frame in changed or past_frame in changed
            if affected:
                past,current=regions(past_frame),regions(frame)
                result=frame_correspondences(past['rgb'],current['rgb'],past['regions'],current['regions']);recomputed+=1
            else:copied+=1
            for source_mode in MODES:
                for field in ['good','total']:
                    key=f'{field}_{lag}_{source_mode}'
                    if not affected:point[key][co0:co1]=old_points[key][old_co0:old_co1]
                    elif field=='good':point[key][co0:co1]=result[('new',source_mode,'rgb')][1]
                    else:point[key][co0:co1]=past['regions']['new']['groups'][source_mode]['total']
                for target in MODES:
                    key=f'vote_{lag}_{source_mode}_{target}'
                    point[key][vo0:vo1]=(result[('new',source_mode,target)][0].ravel()if affected else old_points[key][old_vo0:old_vo1])
    np.testing.assert_array_equal(point['quality'][known],old_points['quality'][mapping[known]])
    point_path=point_out/'new.npz';np.savez_compressed(point_path,**point)
    (point_out/'rgb.npz').symlink_to(old_work/'transport/dev'/a.sequence/'rgb.npz')
    outputs={alias:dict(path=str(point_out/(alias+'.npz')),sha256=sha256(point_out/(alias+'.npz')),
                        frames=len(data['frame_ids']))for alias in ['new','rgb']}
    dump(point_out/'complete.json',dict(status='COMPLETE',split='dev',sequence=a.sequence,outputs=outputs,GT_read=False,
        future_input=False,source_sha256=sha256(__file__),algorithm_sha256=sha256(ROOT/'yolox/tracker/foreground_transport.py'),
        original_point_manifest_sha256=sha256(old_points_path.parent/'complete.json'),
        recomputed_pairs=recomputed,bit_exact_copied_pairs=copied,added_observations=len(added),
        original_descriptor_rows_preserved=True,RGB_control_feature_and_point_files_reused_exactly=True,seconds=time.perf_counter()-started))
    print(json.dumps(dict(sequence=a.sequence,added=len(added),recomputed_pairs=recomputed,copied_pairs=copied,
                         GT_read=False,physical_gpu=physical,seconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':main()
