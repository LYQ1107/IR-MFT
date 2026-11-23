#!/usr/bin/env python3
"""Posthoc Dev identity-change diagnosis; not official IDSW or inference input."""
import json
from pathlib import Path
import sys
from collections import Counter,defaultdict

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import linear_sum_assignment
from tools.run_pseudo_ir_comparison import dump,sha256


def overlap(first,second):
    lower=np.maximum(first[:,None,:2],second[None,:,:2]);upper=np.minimum(first[:,None,2:],second[None,:,2:])
    intersection=np.maximum(upper-lower,0).prod(-1)
    area1=np.maximum(first[:,2:]-first[:,:2],0).prod(-1)
    area2=np.maximum(second[:,2:]-second[:,:2],0).prod(-1)
    return intersection/np.maximum(area1[:,None]+area2[None]-intersection,1e-6)


def diagnose(truth,prediction,frames):
    by_gt=defaultdict(list);events=Counter();matched=0;objects=0
    gt_frames={int(f):truth[truth[:,0]==f]for f in np.unique(truth[:,0])}
    pred_frames={int(f):prediction[prediction[:,0]==f]for f in np.unique(prediction[:,0])}
    for frame in range(1,frames+1):
        gt=gt_frames.get(frame,np.empty((0,truth.shape[1])));objects+=len(gt)
        pred=pred_frames.get(frame,np.empty((0,prediction.shape[1])))
        if not len(gt)or not len(pred):continue
        boxes=gt[:,2:6].copy();boxes[:,2:]+=boxes[:,:2]
        predicted=pred[:,2:6].copy();predicted[:,2:]+=predicted[:,:2]
        matrix=overlap(boxes,predicted);cost=1-matrix;cost[matrix<.5]=1e6
        rows,columns=linear_sum_assignment(cost)
        between=overlap(boxes,boxes);np.fill_diagonal(between,0)
        for row,col in zip(rows,columns):
            if matrix[row,col]<.5:continue
            identity=int(gt[row,1]);current=int(pred[col,1]);history=by_gt[identity];matched+=1
            if history and history[-1][1]!=current:
                gap=frame-history[-1][0]
                events['all_identity_changes']+=1
                events['consecutive_frame'if gap==1 else'interrupted_2_to25'if gap<=25 else'interrupted_over25']+=1
                events['box_overlap_at_least0_1'if between[row].max(initial=0)>=.1 else'box_overlap_below0_1']+=1
                events['box_overlap_at_least0_3']+=int(between[row].max(initial=0)>=.3)
            history.append((frame,current))
    tracks=[]
    for identity,history in by_gt.items():
        counts=Counter(x[1]for x in history)
        tracks.append(dict(GT_id=identity,matched_frames=len(history),unique_predicted_IDs=len(counts),
                           largest_predicted_ID_fraction=max(counts.values())/len(history)))
    return dict(matched_GT_observations=matched,GT_observations=objects,events=dict(events),tracks=tracks)


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(plan['work_root'])
    state=work/'correlation_pilot_policies_dev_state.json'
    if json.loads(state.read_text())['status']!='COMPLETE':raise ValueError('Complete all parent Dev first')
    selection=json.loads((work/'correlation_recoverability_activation.json').read_text())['parent_best']
    names=['CGA_original_RGB_ByteTrack']+[selection[cue]['id']for cue in ['rgb','gray','ir']]
    protocol=json.loads((ROOT/plan['protocol']).read_text());rows=[]
    for name in names:
        run=work/'correlation_pilot_policies_dev'/name;metrics=json.loads((run/'metrics.json').read_text())
        for seq in protocol['dev']:
            gt_path=Path(plan['evaluation']['GT_root'])/'train'/seq/'gt/gt.txt'
            truth=np.loadtxt(gt_path,delimiter=',',ndmin=2);truth=truth[(truth[:,6]>0)&(truth[:,7]==1)]
            pred_path=run/'track_results'/(seq+'.txt');prediction=np.loadtxt(pred_path,delimiter=',',ndmin=2)
            frames=metrics['manifest']['coverage'][seq]['frames']
            rows.append(dict(candidate=name,sequence=seq,frames=frames,GT_sha256=sha256(gt_path),
                             prediction_sha256=sha256(pred_path),**diagnose(truth,prediction,frames)))
    path=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_CORRELATION_IDENTITY_AUDIT.json'
    dump(path,dict(status='COMPLETE_POSTHOC_DEV_DIAGNOSIS',rows=rows,source_sha256=sha256(__file__),
        scope='One-to-one IoU>=0.5 diagnostic mapping of class1 GT to complete predictions. These events are not official TrackEval IDSW. Box overlap is only an occlusion proxy. No Test read, inference change or future-input claim.',
        GT_used_after_inference=True,Test_read=False))
    for row in rows:print(row['candidate'],row['sequence'],row['events'])


if __name__=='__main__':main()
