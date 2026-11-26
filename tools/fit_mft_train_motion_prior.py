#!/usr/bin/env python3
"""Train-only robust motion statistics; no tracker, NN or accuracy evaluation."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.fusion_memory_observation import LENGTHS
from tools.native_pyramid_experiment import context,frozen_json,read
from tools.native_pyramid_training_loop import guard_storage
from tools.run_pseudo_ir_comparison import sha256

SPEC=ROOT/'configs/perceptual_train_motion_prior_20261127.json'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    started=time.monotonic()
    plan,study,protocol,calibration,work,checkpoint=context()
    base=work/'train_motion_prior_20261127';base.mkdir(exist_ok=True)
    output=base/'Train_motion_statistics.json'
    assert not output.exists()and not(base/'fit_registered.json').exists()
    sources={str(p):sha256(p)for p in [SPEC,Path(__file__),ROOT/'configs/perceptual_mft_20261122.json',ROOT/plan['protocol']]}
    GT_root=Path(plan['evaluation']['GT_root'])/'train'
    paths={s:GT_root/s/'gt/gt.txt'for s in protocol['train']}
    assert set(paths)==set(LENGTHS['train'])
    for sequence,path in paths.items():
        assert sha256(path)==protocol['derived_gt_sha256']['train/'+sequence]
        sources[str(path)]=sha256(path)
    frozen_json(base/'fit_registered.json',dict(status='REGISTERED_BEFORE_ALL_TRAIN_MOTION_STATISTICS',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources,
        fit_split='train',Dev_rows=0,Test_rows=0,GPU_work=False,new_NN_fits=0))
    rows=[];sigmas=[];total_boxes=total_triplets=0
    for sequence,path in paths.items():
        raw=np.loadtxt(path,delimiter=',',ndmin=2)
        assert np.isfinite(raw).all()and raw.shape[1]>=8
        valid=raw[(raw[:,6]>0)&(raw[:,7]==1)]
        assert np.all(valid[:,4:6]>0)and len(np.unique(valid[:,:2],axis=0))==len(valid)
        assert np.all(valid[:,0]==valid[:,0].astype(np.int64))and valid[:,0].min()>=1 and valid[:,0].max()<=LENGTHS['train'][sequence]
        acceleration=[];identity_counts={}
        for label in sorted(np.unique(valid[:,1])):
            values=valid[valid[:,1]==label];values=values[np.argsort(values[:,0],kind='stable')]
            frame=values[:,0].astype(np.int64);box=values[:,2:6]
            xyah=np.column_stack((box[:,0]+box[:,2]/2,box[:,1]+box[:,3]/2,box[:,2]/box[:,3],box[:,3]))
            if len(values)<3:
                identity_counts[str(int(label))]=0;continue
            mask=(np.diff(frame)[:-1]==1)&(np.diff(frame)[1:]==1)
            delta=(xyah[2:]-2*xyah[1:-1]+xyah[:-2])[mask]
            heights=xyah[1:-1,3][mask]
            delta[:,[0,1,3]]/=heights[:,None]
            assert np.isfinite(delta).all()
            identity_counts[str(int(label))]=len(delta)
            if len(delta):acceleration.append(delta)
        assert acceleration,sequence
        acceleration=np.concatenate(acceleration,axis=0)
        q90=np.quantile(np.abs(acceleration),.9,axis=0)
        sigma=np.maximum(q90/1.6448536269514722,1e-5)
        sigmas.append(sigma);total_boxes+=len(valid);total_triplets+=len(acceleration)
        rows.append(dict(sequence=sequence,frames=LENGTHS['train'][sequence],valid_GT_boxes=len(valid),
            video_local_identities=len(identity_counts),consecutive_triplets=len(acceleration),
            per_identity_consecutive_triplets=identity_counts,absolute_acceleration_q90=q90.tolist(),
            equal_video_sigma_component=sigma.tolist(),GT_sha256=sha256(path)))
    combined=np.sqrt(np.mean(np.square(np.stack(sigmas)),axis=0))
    for path,digest in sources.items():assert sha256(path)==digest,path
    guard_storage(work)
    value=dict(status='COMPLETE_ALL6_TRAIN_ONLY_ROBUST_MOTION_STATISTICS',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),coordinates=['x_over_h','y_over_h','aspect_absolute','h_over_h'],
        equal_video_RMS_acceleration_sigma=combined.tolist(),videos=rows,
        complete_Train_videos=6,complete_Train_frame_span=sum(LENGTHS['train'].values()),
        valid_GT_boxes=total_boxes,consecutive_triplets=total_triplets,source_hashes=sources,
        specification_sha256=sha256(SPEC),source_sha256=sha256(Path(__file__)),
        includes_annotation_noise_and_interpolation=True,physical_noise_identification_claim=False,
        Dev_rows=0,Test_rows=0,images_read=0,NN_weights_read=0,GPU_work=False,new_NN_fits=0,new_seeds=0,
        new_tracking_accuracy=False,tracking_pipeline_qualified=False,goal_met=False,supported_innovations=0,
        seconds=time.monotonic()-started)
    frozen_json(output,value)
    print(json.dumps({k:v for k,v in value.items()if k not in['videos','source_hashes']}),flush=True)


if __name__=='__main__':main()
