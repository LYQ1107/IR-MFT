#!/usr/bin/env python3
"""Dev-only audit of Lost identities excluded from low-score association."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from yolox.tracker.basetrack import TrackState,BaseTrack
from yolox.tracker.candidate_foreground_tracker import CandidateForegroundTracker
from yolox.tracker.matching import ious,linear_assignment
from tools.run_pseudo_ir_comparison import dump,sha256


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root']);rows=[]
    for arm,prefix in [('new','new_diffv2ir'),('rgb','rgb')]:
        for sequence in protocol['dev']:
            source=Path(plan['frozen_detection_sources'][prefix+'_dev'])/'detections'/(sequence+'.npz')
            with np.load(source) as z:data={k:z[k] for k in z.files}
            with np.load(w/'transport/dev'/sequence/(arm+'.npz')) as z:data['foreground_transport']={k:z[k] for k in z.files}
            inference=[]
            for birth in [.7,.6]:
                BaseTrack._count=0
                owner=CandidateForegroundTracker(dict(point_weight=0,global_birth_threshold=birth,lags=[1,2,4,8]))
                events=[]
                for frame in data['frame_ids']:
                    frame=int(frame);owner.update(frame,data)
                    used={t.evidence['index'] for t in owner.inner.tracked_stracks if t.frame_id==frame}
                    a,b=map(int,data['offsets'][frame-1:frame+1])
                    weak=[i for i in range(a,b) if i not in used and .3<=data['scores'][i]<=.6]
                    if not weak:continue
                    for track in owner.inner.lost_stracks:
                        if track.state!=TrackState.Lost:continue
                        contexts=[c for c in reversed(track.object_observations) if frame-c['frame'] in [1,2,4,8]]
                        if not contexts:continue
                        ctx=contexts[0];lag=frame-ctx['frame']
                        start,end=map(int,data['offsets'][frame-lag-1:frame-lag+1]);r=ctx['index']-start
                        point=data['foreground_transport'];v0,v1=map(int,point[f'vote_offsets_{lag}'][frame-1:frame+1]);p0,p1=map(int,point[f'count_offsets_{lag}'][frame-1:frame+1])
                        overlap=ious(track.tlbr[None],data['boxes'][weak])[0]
                        for j,index in enumerate(weak):
                            values={}
                            for target in ['rgb','joint']:
                                matrix=point[f'vote_{lag}_rgb_{target}'][v0:v1].reshape(end-start,b-a)
                                values[target]=float(matrix[r,index-a])
                            events.append(dict(frame=frame,track=track.track_id,source=ctx['index'],target=index,lag=lag,IoU=float(overlap[j]),score=float(data['scores'][index]),votes=values))
                inference.append((birth,events))
            # GT enters only after both independent inference traces finish.
            gt_path=Path(plan['evaluation']['GT_root'])/'train'/sequence/'gt/gt.txt'
            gt=np.loadtxt(gt_path,delimiter=',',ndmin=2);gt=gt[(gt[:,6]>0)&(gt[:,7]==1)]
            labels=np.full(len(data['boxes']),-1,int)
            for frame in data['frame_ids']:
                a,b=map(int,data['offsets'][frame-1:frame+1]);g=gt[gt[:,0]==frame];boxes=g[:,2:6].copy();boxes[:,2:]+=boxes[:,:2]
                matched,_,_=linear_assignment(1-ious(boxes,data['boxes'][a:b]),.5)
                for gi,pi in matched:labels[a+pi]=int(g[gi,1])
            for birth,events in inference:
                eligible=[e for e in events if labels[e['source']]>=0]
                good=[e for e in eligible if labels[e['target']]==labels[e['source']]]
                rows.append(dict(arm=arm,sequence=sequence,birth_threshold=birth,candidate_pairs=len(events),known_source_pairs=len(eligible),GT_same_identity_pairs=len(good),unique_recoverable_frames=len(set((e['frame'],e['track']) for e in good)),lag_histogram={str(lag):sum(e['lag']==lag for e in good) for lag in [1,2,4,8]},RGB_vote_ge_03=sum(e['votes']['rgb']>=.3 for e in good),IR_vote_ge_03=sum(e['votes']['joint']>=.3 for e in good),mean_original_IoU=float(np.mean([e['IoU'] for e in good])) if good else None,examples=good[:12],GT_sha256=sha256(gt_path)))
                print(json.dumps({k:v for k,v in rows[-1].items() if k not in {'examples','GT_sha256'}}),flush=True)
    dump(w/'weak_recovery_diagnosis.json',dict(status='COMPLETE_DEV_ONLY_DIAGNOSIS',scope='GT-mapped proposal-level Lost/weak pairs after annotation-free inference, not official HOTA or IDSW. Shows potential recoverability, not guaranteed correct matching.',rows=rows,source_sha256=sha256(__file__)))


if __name__=='__main__':main()
