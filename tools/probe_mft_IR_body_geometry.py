#!/usr/bin/env python3
"""Small fixed-Train probe of pixel geometry; no tracker, fitting or new images."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from tools.fit_mft_candidate_risk import locations
from tools.mft_candidate_risk_data import qualify_Train,load_arrays,load_GT
from tools.mft_candidate_risk_labels import current_detection_labels
from tools.native_pyramid_experiment import frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.run_pseudo_ir_comparison import sha256


def measure(image,box):
    """Identical grayscale/IR algorithm; no GT or identity argument."""
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)if image.ndim==3 else image
    box=np.asarray(box,np.float64);center=(box[:2]+box[2:])/2;size=box[2:]-box[:2]
    assert (size>0).all()
    grid=np.linspace(-1,1,96,dtype=np.float32);xx,yy=np.meshgrid(grid,grid)
    mx=(center[0]+xx*.6*size[0]).astype(np.float32);my=(center[1]+yy*.6*size[1]).astype(np.float32)
    patch=cv2.remap(gray,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).astype(np.float64)/255.
    design=np.column_stack([np.ones(xx.size),xx.ravel(),yy.ravel()]);border=(np.abs(xx)>.78)|(np.abs(yy)>.78)
    x=design[border.ravel()];y=patch[border];weights=np.ones(len(y));sigma=1/255.
    for _ in range(3):
        beta=np.linalg.solve(x.T@(weights[:,None]*x)+1e-8*np.eye(3),x.T@(weights*y))
        residual=y-x@beta;sigma=max(1/255.,float(1.4826*np.median(np.abs(residual-np.median(residual)))))
        weights=np.minimum(1.,2.5*sigma/np.maximum(np.abs(residual),1e-9))
    residual=np.abs(patch-(design@beta).reshape(patch.shape));mass=np.maximum(residual-3*sigma,0)
    binary=(mass>0).astype(np.uint8);count,labels,stats,centroids=cv2.connectedComponentsWithStats(binary,8)
    scores=[]
    preference=np.exp(-(xx*xx+yy*yy)/(.4**2*2))
    for index in range(1,count):
        if stats[index,cv2.CC_STAT_AREA]<4:continue
        component=labels==index;scores.append((float((mass*preference)[component].sum()),index))
    if not scores:return dict(valid=False,center_box=box.copy(),shape_box=box.copy(),contrast_sigma=0.,area_fraction=0.,axis_cos2=0.,axis_sin2=0.,anisotropy=0.,dominance=0.)
    scores.sort(reverse=True);component=labels==scores[0][1];weight=mass*component;total=weight.sum()
    px=mx.astype(np.float64);py=my.astype(np.float64);cx=float((px*weight).sum()/total);cy=float((py*weight).sum()/total)
    dx,dy=px-cx,py-cy;moments=np.asarray([[(dx*dx*weight).sum(),(dx*dy*weight).sum()],[(dx*dy*weight).sum(),(dy*dy*weight).sum()]])/total
    eigen,vectors=np.linalg.eigh(moments);vector=vectors[:,-1]
    angle=float(np.arctan2(vector[1],vector[0]));ys,xs=np.where(component)
    extent=np.asarray([px[ys,xs].min(),py[ys,xs].min(),px[ys,xs].max(),py[ys,xs].max()])
    extent[:2]-=size*.6/95;extent[2:]+=size*.6/95
    # Two fixed, mild probes; no GT chooses a correction strength or polarity.
    moved=box+np.r_[np.asarray([cx,cy])-center,np.asarray([cx,cy])-center]*.5
    shaped=.75*box+.25*extent
    return dict(valid=True,center_box=moved,shape_box=shaped,contrast_sigma=float(residual[component].mean()/sigma),
        area_fraction=float(component.mean()),axis_cos2=float(np.cos(2*angle)),axis_sin2=float(np.sin(2*angle)),
        anisotropy=float((eigen[1]-eigen[0])/max(eigen.sum(),1e-12)),
        dominance=float(scores[0][0]/max(sum(v[0]for v in scores),1e-12)))


def iou(a,b):
    overlap=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    union=np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-overlap
    return float(overlap/max(union,1e-12))


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';cv2.setNumThreads(1)
    plan,protocol,work,parent=locations();base=work/'IR_body_geometry_probe_20261127';started=time.monotonic()
    assert not(base/'complete.json').exists();qualified=qualify_Train(work,protocol)
    source_sha=sha256(Path(__file__));inputs=[];rows=[];all_packets=0
    # Five fixed uniformly spaced frames per Train video; no Dev or Test data.
    selection={s:sorted({1+round((q['frames']-1)*j/4)for j in range(5)})for s,q in qualified.items()}
    frozen_json(base/'selection_before_images.json',dict(status='FIXED30_TRAIN_FRAMES_BEFORE_PIXEL_READ',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=source_sha,frames=selection,
        qualified_Train=qualified,Dev_frames=0,Test_frames=0,new_tracker=False,new_model=False))
    for sequence,q in qualified.items():
        arrays=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256'])
        for frame in selection[sequence]:
            lo,hi=map(int,arrays['offsets'][frame-1:frame+1]);detections=arrays['detections'][lo:hi]
            paths=[Path('/data3/liuyeqiang/datasets/mft26-ir/train')/sequence/'visible'/f'{frame:06d}.jpg',
                Path('/data3/liuyeqiang/nas/A100/IR-MFT/diffv2ir_final_20261104/full_generation/train')/sequence/'infrared_diffv2ir'/f'{frame:06d}.png']
            images=[cv2.imread(str(p),cv2.IMREAD_COLOR)for p in paths]
            assert all(image is not None for image in images)and images[0].shape==images[1].shape
            inputs.extend(dict(path=str(p),sha256=sha256(p),sequence=sequence,frame=frame)for p in paths)
            measurements=[[measure(image,d[:4])for d in detections]for image in images]
            all_packets+=len(detections)
            # GT only joins already-computed features for analysis; no GT crop.
            left,right=map(int,bounds[frame-1:frame+1]);current=gt[left:right]
            labels=current_detection_labels(detections,current,frame=frame,split='train')
            for index,(d,label)in enumerate(zip(detections,labels)):
                if label<=0 or d[4]<=.6:continue
                truth=current[(current[:,1]==label)&(current[:,6]>0)&(current[:,7]==1)]
                assert len(truth)==1;b=truth[0,2:6].copy();b[2:]+=b[:2];original=iou(d[:4],b)
                result={}
                for cue,values in zip(['RGB_gray','IR'],measurements):
                    m=values[index];result[cue]={k:v for k,v in m.items()if k not in ['center_box','shape_box']}
                    for correction in ['center_box','shape_box']:result[cue][correction+'_IoU_delta']=iou(m[correction],b)-original
                rows.append(dict(sequence=sequence,frame=frame,proposal_index=index,original_IoU=original,**result))
            guard_storage(work)
    summaries={}
    for sequence in qualified:
        own=[r for r in rows if r['sequence']==sequence];summaries[sequence]={}
        for cue in ['RGB_gray','IR']:
            summaries[sequence][cue]=dict(samples=len(own),valid=sum(r[cue]['valid']for r in own),
                **{correction:dict(mean_delta=float(np.mean([r[cue][correction+'_IoU_delta']for r in own]))if own else None,
                    improved_gt_001=sum(r[cue][correction+'_IoU_delta']>.01 for r in own),
                    harmed_gt_001=sum(r[cue][correction+'_IoU_delta']<-.01 for r in own))for correction in ['center_box','shape_box']})
    aggregate={}
    for correction in ['center_box','shape_box']:
        aggregate[correction]={cue:float(np.mean([s[cue][correction]['mean_delta']for s in summaries.values()if s[cue]['samples']]))for cue in ['RGB_gray','IR']}
        aggregate[correction]['IR_minus_RGB_gray_equal_video_delta']=aggregate[correction]['IR']-aggregate[correction]['RGB_gray']
    assert sha256(Path(__file__))==source_sha
    frozen_json(base/'complete.json',dict(status='COMPLETE_FIXED30_TRAIN_PIXEL_GEOMETRY_PROBE_NO_TRACKING_RESULT',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=source_sha,frames=30,
        sequences=6,all_proposals_measured_before_GT=all_packets,high_score_known_GT_analysis_samples=len(rows),
        per_sequence=summaries,equal_video_aggregate=aggregate,observations=rows,original_image_inputs=inputs,
        algorithm_identical_for_RGB_gray_and_IR=True,foreground_polarity='absolute robust plane residual for both cues',
        GT_reads='Current Train labels after pixel features;no GT crops or inference decision',
        private_image_or_feature_disk_dump=False,changed_tracker=False,new_Dev=False,new_Test=False,new_models=0,
        new_GPU_work=False,seed=0,seconds=time.monotonic()-started,not_full_Train_or_accuracy_validation=True,
        goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_FIXED30_TRAIN_PIXEL_GEOMETRY_PROBE_NO_TRACKING_RESULT',
        measured_proposals=all_packets,analysis_samples=len(rows),equal_video_aggregate=aggregate,seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':main()
