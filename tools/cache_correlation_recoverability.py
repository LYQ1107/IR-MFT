#!/usr/bin/env python3
"""Train-only eligibility, shared identically by every future auxiliary control."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np

from tools.run_pseudo_ir_comparison import dump,sha256
from tools.build_correlation_cache import read_pair
from yolox.tracker.foreground_transport import region_masks


def main():
    p=argparse.ArgumentParser();p.add_argument('--sequence',required=True);a=p.parse_args();cv2.setNumThreads(1)
    main=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(main['work_root'])
    config_path=ROOT/'configs/perceptual_correlation_recoverability_20261123.json';config=json.loads(config_path.read_text())
    parent_path=ROOT/config['parent_architecture_and_cache_plan']
    if sha256(parent_path)!=config['parent_sha256']:raise ValueError('Frozen parent setting changed')
    protocol=json.loads((ROOT/main['protocol']).read_text())
    if a.sequence not in protocol['train']:raise ValueError('Train-only eligibility')
    source=work/'correlation_cache'/a.sequence/'features/complete.json';meta=json.loads(source.read_text())
    if meta['status']!='COMPLETE_TRAIN_FROZEN_FPN'or meta['config_sha256']!=config['parent_sha256']:
        raise ValueError('Complete original Train feature cache required')
    gt_path=Path(main['evaluation']['GT_root'])/'train'/a.sequence/'gt/gt.txt'
    if sha256(gt_path)!=meta['GT_sha256']:raise ValueError('Original Train labels changed')
    truth=np.loadtxt(gt_path,delimiter=',',ndmin=2);truth=truth[(truth[:,6]>0)&(truth[:,7]==1)]
    targets=np.load(meta['targets']['path'],mmap_mode='r')
    out=work/'correlation_recoverability'/a.sequence;out.mkdir(parents=True,exist_ok=True)
    if(out/'complete.json').exists():raise ValueError('Eligibility already complete')
    quality=np.zeros(targets.shape[:2],np.float16);overlap=np.zeros_like(quality);eligible=np.zeros_like(quality,dtype=bool)
    counts=[];started=time.perf_counter()
    for index,frame in enumerate(meta['frame_ids']):
        rgb,ir=read_pair(main,a.sequence,int(frame))
        boxes=truth[truth[:,0]==frame,2:6].astype(np.float32).copy()
        expected=boxes.copy();expected[:,:2]+=expected[:,2:]/2;expected*=meta['ratios'][index]
        np.testing.assert_array_equal(targets[index,:len(boxes),1:],expected)
        boxes[:,2:]+=boxes[:,:2];counts.append(len(boxes))
        for obj,box in enumerate(boxes):
            masks,q=region_masks(rgb,ir,box)
            first=masks['rgb']>=.25;second=masks['ir']>=.25
            second=cv2.dilate(second.astype(np.uint8),np.ones((3,3),np.uint8))>0
            agreement=float((first&second).sum()/max(1,min(first.sum(),second.sum())))
            quality[index,obj]=q[1];overlap[index,obj]=agreement
            eligible[index,obj]=q[1]>=.15 and agreement>=.25
        if(index+1)%32==0:print(json.dumps(dict(sequence=a.sequence,frames=index+1,total=len(targets))),flush=True)
    path=out/'eligibility.npz';np.savez_compressed(path,frame_ids=np.array(meta['frame_ids']),eligible=eligible,IR_quality=quality,
                                                 tolerant_overlap=overlap,objects_per_frame=np.array(counts,np.int32))
    dump(out/'complete.json',dict(status='COMPLETE_TRAIN_ONLY_RECOVERABILITY_DIAGNOSTIC',sequence=a.sequence,frames=len(targets),
        objects=sum(counts),eligible_objects=int(eligible.sum()),config_sha256=sha256(config_path),parent_sha256=config['parent_sha256'],
        source_feature_manifest_sha256=sha256(source),GT_sha256=sha256(gt_path),path=str(path),sha256=sha256(path),
        shared_RGB_gray_IR_selection=True,Dev_rows=0,Test_read=False,source_sha256=sha256(__file__),seconds=time.perf_counter()-started))


if __name__=='__main__':main()
