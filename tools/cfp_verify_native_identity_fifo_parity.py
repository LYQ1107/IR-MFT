#!/usr/bin/env python3
"""Actual native prefix: same-input FIFO through the learned adapter is neutral."""
import argparse,json,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from cfp.native_local_identity import RecentIdentityControl,install_native_local_identity
from cfp.predictive_association import PredictiveAssociationTracker
from cfp.provenance import sha256,snapshot_sources,write_json
from yolox.tracker.basetrack import BaseTrack

class ExactFIFO(nn.Module):
    def __init__(self):super().__init__();self.logit_scale=nn.Parameter(torch.tensor(0.),requires_grad=False)
    def forward(self,history_global,history_valid,candidate_global,candidate_fraction,**unused):
        identity=F.normalize((history_global*history_valid[:,:,None]).sum(1),dim=-1)
        return dict(scores=identity@F.normalize(candidate_global,dim=-1).T,history_available=history_valid.any(1),current=dict(valid=candidate_fraction>0))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--features',type=Path,required=True);ap.add_argument('--detections',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--frames',type=int,default=512);a=ap.parse_args()
    if a.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New CPU proof required')
    torch.set_num_threads(2);m=json.loads((a.features/'manifest.json').read_text());video=m['video']
    if m['status']!='COMPLETE_NATIVE_CURRENT_DINO_FEATURES':raise ValueError('Complete producer required')
    bank={k:np.load(a.features/(k+'.npy'),mmap_mode='r') for k in ['embedding','raw_cls','local4','visibility4','valid']}
    with np.load(a.detections/(video+'.npz')) as d:frames,offsets,boxes,scores=[d[k].copy() for k in ['frame_ids','offsets','boxes','scores']]
    bank['frame_of_proposal']=np.repeat(frames,np.diff(offsets));runs=[]
    config=json.loads(Path('configs/cfp/phase_vii_native_current_identity_replay_v1.json').read_text())
    for adapter in [False,True]:
        BaseTrack._count=0
        t=PredictiveAssociationTracker(SimpleNamespace(track_thresh=config['track_thresh'],track_buffer=config['track_buffer'],match_thresh=config['match_thresh'],mot20=config['mot20']),frame_rate=25,predictor=RecentIdentityControl(),memory='predictor',history_view='recent',appearance_weight=.3,proximity_iou=.1,ema=.9)
        if adapter:t=install_native_local_identity(t,ExactFIFO(),bank)
        costs=[];method=t.association_cost
        def capture(*args):
            result=method(*args);costs.append(result.copy());return result
        t.association_cost=capture;states=[]
        for i,frame in enumerate(frames[:a.frames]):
            lo,hi=offsets[i:i+2];tracks=t.update(np.c_[boxes[lo:hi],scores[lo:hi]],(1,1),(1,1),np.array(bank['embedding'][lo:hi],copy=True),proposal_indices=np.arange(lo,hi))
            states.append([(int(x.track_id),int(x.current_proposal_index),x.tlwh.tolist(),float(x.score),list(x.identity_proposals)) for x in tracks])
        runs.append((states,costs))
    if runs[0][0]!=runs[1][0] or len(runs[0][1])!=len(runs[1][1]):raise AssertionError('Native FIFO adapter changed lifecycle/output/history')
    error=max([float(np.max(np.abs(x-y))) if x.size else 0. for x,y in zip(runs[0][1],runs[1][1])])
    if error>3e-7:raise AssertionError('FIFO native association cost differs')
    write_json(a.output,dict(status='VERIFIED_ACTUAL_NATIVE_FIFO_ADAPTER_PARITY',video=video,frames=min(a.frames,len(frames)),association_calls=len(runs[0][1]),maximum_cost_error=error,exact_output_lifecycle_history=True,feature_manifest_sha256=sha256(a.features/'manifest.json'),detection_sha256=sha256(a.detections/(video+'.npz')),sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__,'cfp/native_local_identity.py','cfp/predictive_association.py']),scope='Actual native prefix512 only; exact FIFOmodel replacing learned cue must preserve accepted measurements/localIDs/memories/lifecycle. Not a learned gain or whole-video byte parity claim.'))
    print(json.dumps(dict(video=video,maximum_cost_error=error,frames=a.frames)))
if __name__=='__main__':main()
