#!/usr/bin/env python3
"""Dev-only GT diagnosis, separated from annotation-free inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from yolox.tracker.matching import ious,linear_assignment
from tools.run_pseudo_ir_comparison import dump,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',default='configs/perceptual_mft_20261122.json');args=p.parse_args()
    plan=json.loads(Path(args.plan).read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());output={}
    for arm,source_key in [('new','new_diffv2ir_dev'),('rgb','rgb_dev')]:
        source=Path(plan['frozen_detection_sources'][source_key]);result={}
        for sequence in protocol['dev']:
            gt_path=Path(plan['evaluation']['GT_root'])/'train'/sequence/'gt/gt.txt'
            gt=np.loadtxt(gt_path,delimiter=',',ndmin=2)
            gt=gt[(gt[:,6]>0)&(gt[:,7]==1)]
            prediction_path=source/'track_results'/f'{sequence}.txt'
            prediction=np.loadtxt(prediction_path,delimiter=',',ndmin=2)
            with np.load(source/'detections'/f'{sequence}.npz') as z: data={k:z[k] for k in z.files}
            groups={f:gt[gt[:,0]==f] for f in data['frame_ids']}
            pred_groups={f:prediction[prediction[:,0]==f] for f in data['frame_ids']}
            counts={str(t):dict(TP=0,FN=0,FP=0) for t in [.1,.6,.7]}
            assignments={};switches=[];gaps=[];previous={};motion_errors=[];histories={}
            tracked_count=0;good_but_not_tracked=0
            for frame in data['frame_ids']:
                frame=int(frame);g=groups[frame];gboxes=g[:,2:6].copy();gboxes[:,2:]+=gboxes[:,:2]
                a,b=data['offsets'][frame-1:frame+1];boxes=data['boxes'][a:b];scores=data['scores'][a:b]
                proposed_good=set()
                for threshold in [.1,.6,.7]:
                    pick=np.flatnonzero(scores>=threshold);pairs,misses,false=linear_assignment(1-ious(gboxes,boxes[pick]),.5)
                    c=counts[str(threshold)];c['TP']+=len(pairs);c['FN']+=len(misses);c['FP']+=len(false)
                    if threshold==.1:proposed_good=set(pairs[:,0])
                rows=pred_groups[frame];pboxes=rows[:,2:6].copy();pboxes[:,2:]+=pboxes[:,:2]
                pairs,_,_=linear_assignment(1-ious(gboxes,pboxes),.5)
                tracked=set(pairs[:,0]);tracked_count+=len(pairs)
                good_but_not_tracked+=len(proposed_good-tracked)
                for gi,pi in pairs:
                    identity=int(g[gi,1]);track=int(rows[pi,1])
                    if identity in previous:
                        old_frame,old_track=previous[identity]
                        if old_track!=track:
                            switches.append(dict(frame=frame,GT_identity=identity,old_track=old_track,new_track=track,gap=frame-old_frame))
                        if frame-old_frame>1:gaps.append(frame-old_frame)
                    previous[identity]=(frame,track)
                for row in g:
                    identity=int(row[1]);center=row[2:4]+row[4:6]/2
                    history=histories.setdefault(identity,[])
                    if len(history)>=2 and history[-1][0]==frame-1 and history[-2][0]==frame-2:
                        error=np.linalg.norm((center-2*history[-1][1]+history[-2][1])/np.maximum(row[4:6],1))
                        motion_errors.append(float(error))
                    history.append((frame,center));histories[identity]=history[-2:]
            for c in counts.values():
                c['recall']=c['TP']/max(c['TP']+c['FN'],1);c['precision']=c['TP']/max(c['TP']+c['FP'],1)
            result[sequence]=dict(frames=len(data['frame_ids']),GT_observations=len(gt),proposal_thresholds=counts,
                                  matched_track_observations=tracked_count,good_proposals_not_emitted_as_correct_tracks=good_but_not_tracked,
                                  diagnostic_ID_changes=len(switches),ID_change_gap_histogram={str(k):sum(x['gap']==k for x in switches) for k in [1,2,3,5,10]},
                                  long_reacquisition_gaps=sum(x>5 for x in gaps),
                                  constant_velocity_GT_residual_quantiles=np.quantile(motion_errors,[.5,.9,.95,.99]).tolist(),
                                  first_switch_examples=switches[:20],GT_sha256=sha256(gt_path),prediction_sha256=sha256(prediction_path))
        output[arm]=result
    report=dict(status='COMPLETE_DEV_DIAGNOSIS',scope='Offline GT-assisted Dev diagnosis only; per-frame IoU Hungarian mapping is not the official HOTA global association and its event counts must not be substituted for TrackEval IDSW.',results=output)
    target=Path(plan['work_root'])/'dev_failure_diagnosis.json';dump(target,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
