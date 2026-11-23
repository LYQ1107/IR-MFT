#!/usr/bin/env python3
"""Posthoc Dev body-axis diagnosis from frozen current-frame HOG fields."""
import configparser
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np

from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.tracker.matching import ious,linear_assignment


def axis(descriptor,boxes,width,height):
    hist=descriptor.astype(np.float32).reshape(-1,16,9).sum(1)
    angles=(np.arange(9)+.5)*np.pi/9
    phase=hist@np.exp(2j*angles)
    concentration=np.abs(phase)/np.maximum(hist.sum(1),1e-8)
    direction=np.angle(phase)/2+np.pi/2
    size=boxes[:,2:]-boxes[:,:2]
    lo=np.maximum(boxes[:,:2]-.05*size,0);hi=np.minimum(boxes[:,2:]+.05*size,[width,height])
    extent=np.maximum(hi-lo,1)
    physical=np.arctan2(np.sin(direction)*extent[:,1]/32,np.cos(direction)*extent[:,0]/64)
    return physical,concentration


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(plan['work_root'])
    protocol=json.loads((ROOT/plan['protocol']).read_text());rows=[]
    for sequence in protocol['dev']:
        gt_root=Path(plan['evaluation']['GT_root'])/'train'/sequence
        gt=np.loadtxt(gt_root/'gt/gt.txt',delimiter=',',ndmin=2);gt=gt[(gt[:,6]>0)&(gt[:,7]==1)]
        ini=configparser.ConfigParser();ini.read(gt_root/'seqinfo.ini')
        width,height=ini.getint('Sequence','imWidth'),ini.getint('Sequence','imHeight')
        truth={(int(r[0]),int(r[1])):r[2:4]+r[4:6]/2 for r in gt}
        for front in ['new','rgb']:
            path=work/'features/dev'/sequence/(front+'.npz')
            with np.load(path)as z:
                boxes,offsets,frames=z['boxes'],z['offsets'],z['frame_ids']
                rgb,cr=axis(z['rgb_hog'],boxes,width,height);ir,ci=axis(z['ir_hog'],boxes,width,height);quality=z['quality'][:,1]
            labels=np.full(len(boxes),-1,int);frame_of=np.repeat(frames,np.diff(offsets))
            for position,frame in enumerate(frames):
                lo,hi=offsets[position:position+2];current=gt[gt[:,0]==frame];gb=current[:,2:6].copy();gb[:,2:]+=gb[:,:2]
                pairs,_,_=linear_assignment(1-ious(gb,boxes[lo:hi]),.5)
                for gi,pi in pairs:labels[lo+pi]=int(current[gi,1])
            available=quality>=.15
            ir_fallback=np.where(available,ir,rgb)
            conservative=np.where(available&(ci>cr),ir,rgb)
            bbox=np.where((boxes[:,2]-boxes[:,0])>=(boxes[:,3]-boxes[:,1]),0,np.pi/2)
            for lag in [1,4,8]:
                select=[];motion=[]
                for index,(frame,identity)in enumerate(zip(frame_of,labels)):
                    if identity<0 or (int(frame+lag),int(identity))not in truth:continue
                    delta=truth[(int(frame+lag),int(identity))]-truth[(int(frame),int(identity))]
                    if np.linalg.norm(delta)<max(2,.02*np.linalg.norm(boxes[index,2:]-boxes[index,:2])):continue
                    select.append(index);motion.append(np.arctan2(delta[1],delta[0]))
                select=np.array(select,int);motion=np.asarray(motion)
                for name,value in [('bbox_axis',bbox),('rgb_HOG_axis',rgb),('IR_HOG_RGB_fallback',ir_fallback),('higher_concentration_RGB_IR',conservative)]:
                    error=np.abs((value[select]-motion+np.pi/2)%np.pi-np.pi/2)*180/np.pi
                    low=cr[select]<.15
                    rows.append(dict(sequence=sequence,frontend=front,lag=lag,predictor=name,observations=len(select),
                        IR_available=int(available[select].sum()),mean_angular_error_degrees=float(error.mean()),
                        median_angular_error_degrees=float(np.median(error)),within30_degrees=float((error<=30).mean()),
                        low_RGB_concentration_count=int(low.sum()),
                        low_RGB_mean_error_degrees=float(error[low].mean())if low.any()else None,
                        source_feature_sha256=sha256(path)))
    report=dict(status='COMPLETE_POSTHOC_DEV_AXIS_DIAGNOSTIC_NOT_TRACKING',rows=rows,source_sha256=sha256(__file__),
        scope='Frozen current-frame HOG unsigned edge orientation, converted back to original crop aspect. Future GT displacement is a posthoc label only. No model fitting, motion-prior change, Test read or HOTA claim.',
        Test_read=False,GT_inference_input=False,goal_met=False,supported_multimodal_innovations=0)
    dump(work/'body_axis_dev_audit.json',report)
    dump(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_BODY_AXIS_DEV_AUDIT.json',report)
    print(json.dumps(dict(status=report['status'],rows=len(rows),Test_read=False)))


if __name__=='__main__':main()
