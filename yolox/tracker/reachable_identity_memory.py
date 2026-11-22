"""Current memory edges must be reachable from actual past RGB observations."""
import json
from pathlib import Path

import numpy as np

from yolox.tracker.consensus_identity_memory import ConsensusIdentityMemoryTracker
from yolox.tracker.ordered_identity_memory import OrderedIdentityMemoryTracker


class ReachabilityMixin:
    def __init__(self,config,motion_statistics=None,**kwargs):
        super().__init__(config,**kwargs)
        if motion_statistics is None:
            from tools.run_pseudo_ir_comparison import sha256
            path=Path(config['reachability_statistics'])
            if sha256(path)!=config['reachability_statistics_sha256']:
                raise ValueError('Train motion statistics changed')
            motion_statistics=json.loads(path.read_text())
            root=Path(__file__).resolve().parents[2]
            protocol=json.loads((root/'configs/cfp/protocol_v2.json').read_text())
            if (motion_statistics['status']!='COMPLETE_TRAIN_ONLY_REACHABILITY'
                    or motion_statistics['fit_sequences']!=protocol['train']
                    or motion_statistics['Dev_fit_rows'] or motion_statistics['Test_fit_rows']):
                raise ValueError('Motion bounds violate Train-only fitting')
        self.motion_statistics=motion_statistics;self.reachability_cache={}
        self.statistics.update(reachability_queries=0,reachability_missing_history=0,
                               reachability_RGB_rejected_pairs=0,reachability_native_rejected_pairs=0)

    def update(self,frame,data):
        self.reachability_cache={}
        return super().update(frame,data)

    def reachable(self,frame,context,data):
        key=int(context['index'])
        if key in self.reachability_cache:return self.reachability_cache[key]
        a,b=map(int,data['offsets'][frame-1:frame+1])
        if context['frame']>=frame or key>=a:raise ValueError('Motion history must be strictly past')
        entry=next((e for e in self.identity_archive.values() if e['latest']['index']==key),None)
        if entry is None:raise ValueError('Reachability source is not a retained identity')
        history=[h for h in entry['track'].object_observations if h['frame']<=context['frame']][-8:]
        self.statistics['reachability_queries']+=1
        if len(history)<int(self.config.get('reachability_minimum_observations',2)):
            result=np.zeros(b-a,bool);self.statistics['reachability_missing_history']+=1
        else:
            indices=np.array([h['index'] for h in history],np.int64)
            times=np.array([h['frame'] for h in history],np.float64)
            if np.any(indices>=a) or np.any(np.diff(times)<=0):raise ValueError('Invalid past observation chronology')
            boxes=data['boxes'][indices].astype(np.float64);centers=(boxes[:,:2]+boxes[:,2:])/2
            scale=max(float(np.linalg.norm(data['boxes'][key,2:]-data['boxes'][key,:2])),1.)
            speed=np.linalg.norm(np.diff(centers,axis=0),axis=1)/np.diff(times)/scale
            median=float(np.median(speed));mad=float(np.median(abs(speed-median)))
            upper=max(median+2*1.4826*mad,float(self.motion_statistics['speed_floor']))
            gap=frame-context['frame']
            radius=(float(self.motion_statistics['position_slack_body_diagonals'])+gap*upper
                    +.5*float(self.motion_statistics['acceleration_upper'])*gap*gap)*scale
            current=(data['boxes'][a:b,:2]+data['boxes'][a:b,2:])/2
            result=np.linalg.norm(current-centers[-1],axis=1)<=radius
        self.reachability_cache[key]=result
        return result

    def reference_memory_scores(self,frame,context,data):
        mask=self.reachable(frame,context,data)
        if not mask.any():return np.zeros(len(mask),np.float64),np.zeros(len(mask),np.float32)
        probability,increment=super().reference_memory_scores(frame,context,data)
        rejected=(probability>0)&~mask
        self.statistics['reachability_RGB_rejected_pairs']+=int(rejected.sum())
        probability=np.asarray(probability).copy();probability[~mask]=0
        return probability,increment

    def recovery_scores(self,frame,context,data):
        mask=self.reachable(frame,context,data)
        if not mask.any():return np.zeros(len(mask),np.float64),np.zeros(len(mask),np.float32)
        probability,increment=super().recovery_scores(frame,context,data)
        rejected=(probability>0)&~mask
        self.statistics['reachability_native_rejected_pairs']+=int(rejected.sum())
        probability=np.asarray(probability).copy();probability[~mask]=0
        return probability,increment


class ReachableIdentityMemoryTracker(ReachabilityMixin,OrderedIdentityMemoryTracker):
    pass


class ReachableConsensusMemoryTracker(ReachabilityMixin,ConsensusIdentityMemoryTracker):
    pass
