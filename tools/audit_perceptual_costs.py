#!/usr/bin/env python3
"""Trace a frozen Dev replay first, then diagnose association feasibility with GT."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from yolox.tracker.matching import ious,linear_assignment
from yolox.tracker.perceptual_tracker import PerceptualTracker,box_from_mean
from tools.run_pseudo_ir_comparison import dump,sha256


class TracedTracker(PerceptualTracker):
    def __init__(self,config):
        super().__init__(config);self.events=[]

    def update_observation(self,track,index,data,cost,alternative):
        overlap=float(ious(box_from_mean(track.mean)[None],data['boxes'][index:index+1])[0,0])
        self.events.append(dict(frame=self.frame,previous_frame=track.last_frame,
                                previous_index=track.last_index,index=int(index),IoU=overlap,
                                cost=cost,margin=float(alternative-cost) if np.isfinite(alternative) else None,score=float(data['scores'][index])))
        super().update_observation(track,index,data,cost,alternative)


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    protocol=json.loads((ROOT/plan['protocol']).read_text());work=Path(plan['work_root'])
    config_path=ROOT/'configs/perceptual_screen_20261122/C02_new_raw_appearance.json';config=json.loads(config_path.read_text())
    result={}
    for sequence in protocol['dev']:
        source=work/'features/dev'/sequence/'new.npz'
        with np.load(source) as z:data={k:z[k] for k in z.files}
        tracker=TracedTracker(config)
        for frame in data['frame_ids']:tracker.update(int(frame),data)
        # GT enters only after the entire annotation-free replay has finished.
        gt_path=Path(plan['evaluation']['GT_root'])/'train'/sequence/'gt/gt.txt'
        gt=np.loadtxt(gt_path,delimiter=',',ndmin=2);gt=gt[(gt[:,6]>0)&(gt[:,7]==1)]
        labels=np.full(len(data['boxes']),-1,int)
        for frame in data['frame_ids']:
            a,b=data['offsets'][frame-1:frame+1];g=gt[gt[:,0]==frame]
            boxes=g[:,2:6].copy();boxes[:,2:]+=boxes[:,:2]
            pairs,_,_=linear_assignment(1-ious(boxes,data['boxes'][a:b]),.5)
            for gi,pi in pairs:labels[a+pi]=int(g[gi,1])
        events=tracker.events;zero=[e for e in events if e['IoU']<.01]
        def wrong(event):
            p,q=labels[event['previous_index']],labels[event['index']]
            return bool(p>=0 and q>=0 and p!=q)
        result[sequence]=dict(matches=len(events),zero_IoU_matches=len(zero),
                              zero_IoU_high_score_matches=sum(e['score']>=.6 for e in zero),
                              local_identity_discontinuities=sum(wrong(e) for e in events),
                              zero_IoU_local_identity_discontinuities=sum(wrong(e) for e in zero),
                              source_sha256=sha256(source),GT_sha256=sha256(gt_path),
                              examples=[dict(e,previous_proposal_GT=int(labels[e['previous_index']]),current_proposal_GT=int(labels[e['index']])) for e in zero if wrong(e)][:15])
    output=dict(status='COMPLETE_DEV_COST_DIAGNOSIS',scope='Diagnostic proposal-level identity changes, not official IDSW. Unmodified candidate C02 replay finishes before GT labelling.',
                config_sha256=sha256(config_path),results=result,
                issue='Weighted motion cost shrinks the absolute rejection scale; semantically similar nonoverlapping proposals can pass an unchanged 0.8 threshold. Preserve the original feasible IoU graph before applying ambiguous-edge evidence.')
    dump(work/'cost_feasibility_diagnosis.json',output)
    print(json.dumps({s:{k:v for k,v in d.items() if k not in ['examples','source_sha256','GT_sha256']} for s,d in result.items()}),flush=True)


if __name__=='__main__':main()
