#!/usr/bin/env python3
"""Sparse current-image moment axes; all predictions precede posthoc Dev GT."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.tracker.perceptual_features import clipped_crop,foreground
from yolox.tracker.matching import ious,linear_assignment


def moment_axis(gray):
    mask,quality=foreground(gray);weights=np.maximum(mask-.35,0);total=float(weights.sum())
    if total<1:return 0.,0.
    y,x=np.mgrid[:32,:64];x=(x+.5)*gray.shape[1]/64;y=(y+.5)*gray.shape[0]/32
    cx=float((x*weights).sum()/total);cy=float((y*weights).sum()/total);x-=cx;y-=cy
    covariance=np.array([[(weights*x*x).sum(),(weights*x*y).sum()],[(weights*x*y).sum(),(weights*y*y).sum()]])/total
    values,vectors=np.linalg.eigh(covariance);vector=vectors[:,-1]
    return float(np.arctan2(vector[1],vector[0])),float((values[-1]-values[0])/max(values.sum(),1e-6))


def main():
    cv2.setNumThreads(1)
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(plan['work_root'])
    config_path=ROOT/'configs/perceptual_shape_axis_audit_20261123.json';protocol=json.loads((ROOT/plan['protocol']).read_text())
    out=work/'shape_axis_audit';out.mkdir(exist_ok=True)
    if(out/'predictions_complete.json').exists():raise ValueError('Existing pose predictions')
    proofs={};frames_by_seq={};banks={}
    for seq in protocol['dev']:
        data={}
        for front in ['rgb','new']:
            path=work/'features/dev'/seq/(front+'.npz')
            with np.load(path)as z:
                frames=z['frame_ids'][::32];offsets=z['offsets'];selected=np.concatenate([np.arange(offsets[f-1],offsets[f])for f in frames])
                data[front]=dict(boxes=z['boxes'][selected],quality=z['quality'][selected],frame=np.repeat(frames,np.diff(offsets)[frames-1]),
                    rgb_angle=np.zeros(len(selected),np.float32),ir_angle=np.zeros(len(selected),np.float32),rgb_concentration=np.zeros(len(selected),np.float32),ir_concentration=np.zeros(len(selected),np.float32))
            proofs[seq+'_'+front]=dict(source_feature_sha256=sha256(path));frames_by_seq[seq]=frames
        for frame in frames_by_seq[seq]:
            rgb=cv2.imread(str(Path(plan['data_root'])/'train'/seq/'visible'/f'{frame:06d}.jpg'))
            ir=cv2.imread(str(Path(plan['generated_ir_root'])/'dev'/seq/'infrared_diffv2ir'/f'{frame:06d}.png'),-1)
            if rgb is None or ir is None or ir.ndim!=2 or ir.shape!=rgb.shape[:2]:raise ValueError('Actual pair missing')
            for front,d in data.items():
                for index in np.flatnonzero(d['frame']==frame):
                    box=d['boxes'][index]
                    first=cv2.cvtColor(clipped_crop(rgb,box,.05),cv2.COLOR_BGR2GRAY);second=clipped_crop(ir,box,.05)
                    d['rgb_angle'][index],d['rgb_concentration'][index]=moment_axis(first)
                    d['ir_angle'][index],d['ir_concentration'][index]=moment_axis(second)
        for front,d in data.items():
            path=out/(seq+'_'+front+'.npz');np.savez_compressed(path,**d);proofs[seq+'_'+front].update(path=str(path),sha256=sha256(path))
            banks[seq,front]=d
        print(json.dumps(dict(sequence=seq,unlabeled_frames=len(frames_by_seq[seq]),pose_production_complete=True)),flush=True)
    dump(out/'predictions_complete.json',dict(status='ALL_CURRENT_POSES_COMPLETE_BEFORE_GT_READ',sources=proofs,GT_read=False,Test_read=False,config_sha256=sha256(config_path)))
    rows=[]
    for seq in protocol['dev']:
        gt_path=Path(plan['evaluation']['GT_root'])/'train'/seq/'gt/gt.txt'
        gt=np.loadtxt(gt_path,delimiter=',',ndmin=2);gt=gt[(gt[:,6]>0)&(gt[:,7]==1)]
        truth={(int(r[0]),int(r[1])):r[2:4]+r[4:6]/2 for r in gt}
        for front in ['rgb','new']:
            d=banks[seq,front];boxes=d['boxes'];labels=np.full(len(boxes),-1,int)
            for frame in frames_by_seq[seq]:
                indices=np.flatnonzero(d['frame']==frame);current=gt[gt[:,0]==frame];gb=current[:,2:6].copy();gb[:,2:]+=gb[:,:2]
                pairs,_,_=linear_assignment(1-ious(gb,boxes[indices]),.5)
                for gi,pi in pairs:labels[indices[pi]]=int(current[gi,1])
            bbox=np.where((boxes[:,2]-boxes[:,0])>=(boxes[:,3]-boxes[:,1]),0,np.pi/2)
            rgb_ok=(d['quality'][:,0]>=.15)&(d['rgb_concentration']>=.15)
            ir_ok=(d['quality'][:,1]>=.15)&(d['ir_concentration']>=.15)
            rgb=np.where(rgb_ok,d['rgb_angle'],bbox);ir=np.where(ir_ok,d['ir_angle'],rgb)
            confidence_rgb=d['quality'][:,0]*d['rgb_concentration'];confidence_ir=d['quality'][:,1]*d['ir_concentration']
            combined=np.where(ir_ok&(confidence_ir>confidence_rgb),ir,rgb)
            for lag in [1,4,8]:
                selected=[];motion=[]
                for index,(frame,identity)in enumerate(zip(d['frame'],labels)):
                    if identity<0 or(int(frame+lag),int(identity))not in truth:continue
                    delta=truth[int(frame+lag),int(identity)]-truth[int(frame),int(identity)]
                    if np.linalg.norm(delta)<max(2,.02*np.linalg.norm(boxes[index,2:]-boxes[index,:2])):continue
                    selected.append(index);motion.append(np.arctan2(delta[1],delta[0]))
                selected=np.array(selected,int);motion=np.asarray(motion)
                for name,angles in [('bbox_axis',bbox),('RGB_moment_axis',rgb),('IR_moment_RGB_fallback',ir),('higher_quality_concentration',combined)]:
                    error=np.abs((angles[selected]-motion+np.pi/2)%np.pi-np.pi/2)*180/np.pi
                    rows.append(dict(sequence=seq,frontend=front,lag=lag,predictor=name,observations=len(selected),
                        mean_angular_error_degrees=float(error.mean()),median_angular_error_degrees=float(np.median(error)),within30_degrees=float((error<=30).mean()),
                        RGB_valid=int(rgb_ok[selected].sum()),IR_valid=int(ir_ok[selected].sum())))
    report=dict(status='COMPLETE_SPARSE_DEV_MOMENT_AXIS_DIAGNOSTIC',rows=rows,sources=proofs,config_sha256=sha256(config_path),source_sha256=sha256(__file__),
        scope='Every32frame fixed sparse current-proposal moments, all predictions before GT scoring. FutureGTdisplacement is only an unsigned-motion proxy, not trueorientation or trackingHOTA. No model fit or deployed motion change.',Test_read=False,supported_innovations=0,goal_met=False)
    dump(out/'complete.json',report);dump(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_MOMENT_AXIS_DEV_AUDIT.json',report)
    print(json.dumps(dict(status=report['status'],rows=len(rows),Test_read=False)))


if __name__=='__main__':main()
