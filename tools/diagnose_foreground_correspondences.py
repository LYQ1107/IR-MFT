#!/usr/bin/env python3
"""Offline Dev-only correspondence diagnosis, never imported by inference."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from yolox.tracker.matching import ious,linear_assignment
from tools.run_pseudo_ir_comparison import dump,sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root']);rows=[]
    for sequence in protocol['dev']:
        directory=w/'transport/dev'/sequence
        while not (directory/'complete.json').exists():
            if not args.wait:raise RuntimeError('Complete Dev producer required')
            time.sleep(5)
        gt_path=Path(plan['evaluation']['GT_root'])/'train'/sequence/'gt/gt.txt'
        gt=np.loadtxt(gt_path,delimiter=',',ndmin=2);gt=gt[(gt[:,6]>0)&(gt[:,7]==1)]
        for arm,prefix in [('rgb','rgb'),('new','new_diffv2ir')]:
            source=Path(plan['frozen_detection_sources'][prefix+'_dev'])/'detections'/(sequence+'.npz')
            with np.load(source) as z:data={k:z[k] for k in ['boxes','scores','offsets','frame_ids']}
            with np.load(directory/(arm+'.npz')) as z:point={k:z[k] for k in z.files}
            labels=np.full(len(data['boxes']),-1,int)
            for frame in data['frame_ids']:
                a,b=map(int,data['offsets'][frame-1:frame+1]);group=gt[gt[:,0]==frame]
                boxes=group[:,2:6].copy();boxes[:,2:]+=boxes[:,:2]
                matches,_,_=linear_assignment(1-ious(boxes,data['boxes'][a:b]),.5)
                for gi,di in matches:labels[a+di]=int(group[gi,1])
            for lag in [1,2,4,8]:
                for mode in ['bbox','rgb','ir','joint']:
                    counts=dict(query=0,identity_present=0,confident=0,confident_correct=0,opening_confident=0,opening_correct=0,known_target_confident=0,known_target_correct=0)
                    quality=[];maximum_votes=[]
                    for frame in data['frame_ids']:
                        if frame<=lag:continue
                        a,b=map(int,data['offsets'][frame-1:frame+1]);c,d=map(int,data['offsets'][frame-lag-1:frame-lag+1])
                        targets=np.flatnonzero(data['scores'][a:b]>.6)
                        if not len(targets):continue
                        v0,v1=map(int,point[f'vote_offsets_{lag}'][frame-1:frame+1]);p0,p1=map(int,point[f'count_offsets_{lag}'][frame-1:frame+1])
                        matrix=point[f'vote_{lag}_{mode}_{mode}'][v0:v1].reshape(d-c,b-a)[:,targets].astype(np.float64)
                        good=point[f'good_{lag}_{mode}'][p0:p1];total=point[f'total_{lag}_{mode}'][p0:p1]
                        target_labels=labels[a:b][targets]
                        for local in np.flatnonzero((labels[c:d]>=0)&(data['scores'][c:d]>.6)):
                            identity=labels[c+local];counts['query']+=1
                            present=identity in target_labels;counts['identity_present']+=int(present)
                            value=matrix[local];best=int(value.argmax());maximum=float(value[best])
                            second=float(np.partition(value,-2)[-2]) if len(value)>1 else 0.
                            purity=max(0.,(maximum-second)/max(maximum,1e-8))
                            certainty=float(good[local])/max(float(total[local]),1.)*min(1.,maximum/.4)*min(1.,purity/.5)*math.exp(-lag/32.)
                            certainty*=min(1.,purity/.5)
                            quality.append(float(point['quality'][c+local,1]));maximum_votes.append(maximum)
                            eligible=good[local]>=4 and maximum>=.15
                            correct=target_labels[best]==identity
                            if eligible and certainty>=.3:
                                counts['confident']+=1;counts['confident_correct']+=int(correct)
                                if target_labels[best]>=0:
                                    counts['known_target_confident']+=1;counts['known_target_correct']+=int(correct)
                            if eligible and certainty>=.7:
                                counts['opening_confident']+=1;counts['opening_correct']+=int(correct)
                    rows.append(dict(sequence=sequence,arm=arm,lag=lag,mode=mode,**counts,
                                     confident_precision=counts['confident_correct']/max(counts['confident'],1),
                                     opening_precision=counts['opening_correct']/max(counts['opening_confident'],1),
                                     maximum_vote_quantiles=np.quantile(maximum_votes,[0,.25,.5,.75,1]).tolist(),
                                     source_IR_quality_median=float(np.median(quality))))
            print(json.dumps(dict(sequence=sequence,arm=arm,complete=True)),flush=True)
    result=dict(status='COMPLETE_DEV_ONLY_DIAGNOSIS',scope='Proposal-level IoU Hungarian GT mapping after independent point extraction; not official IDSW/HOTA or a tracking result. No GT input to runtime. Single-anchor confidence ignores reciprocal competition and multi-history averaging, so this diagnoses raw evidence rather than predicting final decisions.',rows=rows,source_sha256=sha256(__file__))
    dump(w/'foreground_correspondence_diagnosis.json',result)


if __name__=='__main__':main()
