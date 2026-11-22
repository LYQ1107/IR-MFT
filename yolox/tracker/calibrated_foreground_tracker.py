"""Candidate association using Train-fitted signed foreground evidence."""
import json
from pathlib import Path

import numpy as np
import torch

from yolox.tracker.association_calibrator import AssociationCalibrator,prepare_calibration_cue
from yolox.tracker.association_features import pair_features
from yolox.tracker.candidate_foreground_tracker import CandidateForegroundTracker
from yolox.tracker.cycle_candidate_tracker import CycleBirthMixin
from yolox.tracker.weak_recovery_tracker import WeakRecoveryMixin


class CalibratedForegroundTracker(CandidateForegroundTracker):
    def __init__(self,config,model=None,cue=None):
        config=dict(dict(source_mode='rgb',target_mode='rgb',lags=[1,2,4,8]),**config)
        super().__init__(config)
        if model is None:
            checkpoint=torch.load(config['association_checkpoint'],map_location='cpu',weights_only=False)
            root=Path(__file__).resolve().parents[2]
            protocol=json.loads((root/'configs/cfp/protocol_v2.json').read_text())
            if checkpoint['fit_sequences']!=protocol['train'] or checkpoint['Dev_fit_rows'] or checkpoint['Test_fit_rows']:
                raise ValueError('Association model violates Train-only fitting')
            model=AssociationCalibrator(**checkpoint['architecture']);model.load_state_dict(checkpoint['model'],strict=True)
            self.calibration_metadata={k:checkpoint[k] for k in ['cue','seed','epoch','sources','fit_sequences','config_sha256','prior_plan_sha256','capacity_plan_sha256']}
            self.cue=checkpoint['cue']
            if config.get('association_plan_sha256',checkpoint['config_sha256'])!=checkpoint['config_sha256']:
                raise ValueError('Association fitting plan mismatch')
            if config.get('association_prior_sha256',checkpoint['prior_plan_sha256'])!=checkpoint['prior_plan_sha256']:
                raise ValueError('Association missing-point policy mismatch')
            if config.get('association_capacity_sha256',checkpoint['capacity_plan_sha256'])!=checkpoint['capacity_plan_sha256']:
                raise ValueError('Association RGB capacity control mismatch')
        else:
            if cue not in {'rgb','ir'}:raise ValueError('Injected model needs an explicit cue')
            self.cue=cue;self.calibration_metadata=dict(cue=cue,scope='injected engineering-test model')
        if config.get('calibration_cue',self.cue)!=self.cue:raise ValueError('Checkpoint cue mismatch')
        self.model=model.cpu().eval();self.pair_cache={}
        self.statistics.update(calibrated_rows=0,IR_increment_rows=0,IR_applied_rows=0,motion_preserved_rows=0)

    def update(self,frame,data):
        self.data=data;self.pair_cache={}
        return super().update(frame,data)

    def predict_pairs(self,frame,lag):
        if lag in self.pair_cache:return self.pair_cache[lag]
        data=self.data;point=data['foreground_transport']
        a,b=map(int,data['offsets'][frame-1:frame+1]);c,d=map(int,data['offsets'][frame-lag-1:frame-lag+1])
        fields=['boxes','embedding','rgb_mask_embedding','rgb_hog','color','quality']
        if self.cue=='ir':fields+=['ir_mask_embedding','ir_hog']
        past={k:data[k][c:d] for k in fields};current={k:data[k][a:b] for k in fields}
        past['point_quality']=point['quality'][c:d];current['point_quality']=point['quality'][a:b]
        v0,v1=map(int,point[f'vote_offsets_{lag}'][frame-1:frame+1]);p0,p1=map(int,point[f'count_offsets_{lag}'][frame-1:frame+1])
        mode='joint' if self.cue=='ir' else 'bbox'
        matrices={}
        for source,target in [('rgb','rgb'),('rgb',mode),(mode,'rgb')]:
            matrices[(source,target)]=point[f'vote_{lag}_{source}_{target}'][v0:v1].reshape(d-c,b-a)
        good={s:point[f'good_{lag}_{s}'][p0:p1] for s in ['rgb',mode]}
        total={s:point[f'total_{lag}_{s}'][p0:p1] for s in ['rgb',mode]}
        base,extra,reference=pair_features(past,current,matrices,good,total,lag,self.cue)
        extra=prepare_calibration_cue(extra,self.cue)
        if self.cue=='ir':available=(past['point_quality'][:,1:2]>=.15)&(current['point_quality'][None,:,1]>=.15)
        else:available=np.ones((d-c,b-a),bool)
        n=(d-c)*(b-a)
        probability=np.zeros(n,np.float32);increments=np.zeros(n,np.float32)
        with torch.inference_mode():
            for lo in range(0,n,4096):
                bs=torch.from_numpy(base.reshape(-1,23)[lo:lo+4096]);cs=torch.from_numpy(extra.reshape(-1,17)[lo:lo+4096]);rs=torch.from_numpy(reference.reshape(-1,17)[lo:lo+4096]);valid=torch.from_numpy(available.ravel()[lo:lo+4096])
                if self.config.get('shuffle_cue',False):cs=torch.roll(cs,shifts=1,dims=0)
                delta=torch.zeros(len(bs)) if self.config.get('disable_learned_increment',False) else self.model.increment(bs,cs,rs,valid)
                probability[lo:lo+len(bs)]=(self.model.RGB_offset(bs)+delta).sigmoid().numpy()
                increments[lo:lo+len(bs)]=delta.numpy()
        value=(c,probability.reshape(d-c,b-a),increments.reshape(d-c,b-a),matrices[('rgb','rgb')],good['rgb'],total['rgb'])
        self.pair_cache[lag]=value
        return value

    def rank_cost(self,tracks,observations,base,frame,stage):
        if not base.size or stage not in self.config.get('point_stages',['primary','secondary']):return base
        amount=float(self.config.get('association_weight',.8))
        if amount<=0:return base
        result=base.copy();columns=np.array([o.evidence['index']-self.current_start for o in observations],int)
        limit={'primary':.9,'secondary':.5,'pending':.7}[stage]
        ambiguity=None
        if self.config.get('preserve_clear_motion',False):
            feasible=base<=limit;masked=np.where(feasible,base,np.inf)
            r=feasible&(base-np.min(masked,axis=1,keepdims=True)<.12);c=feasible&(base-np.min(masked,axis=0,keepdims=True)<.12)
            ambiguity=(r&(r.sum(1,keepdims=True)>1))|(c&(c.sum(0,keepdims=True)>1))
        for row,track in enumerate(tracks):
            history=track.object_observations if self.config.get('object_files',True) else [track.evidence]
            contexts=[c for c in reversed(history) if frame-c['frame'] in self.config.get('lags',[1,2,4,8])]
            if not contexts:continue
            context=contexts[0];lag=frame-context['frame']
            start,probabilities,delta,vote,good,total=self.predict_pairs(frame,lag);index=context['index']-start
            if index<0 or index>=len(probabilities):raise ValueError('Object-file source index mismatch')
            if good[index]<4:continue
            p=probabilities[index,columns].astype(np.float64);best=int(p.argmax());maximum=float(p[best])
            second=float(np.partition(p,-2)[-2]) if len(p)>1 else 0.
            confidence=maximum*min(1.,max(0.,maximum-second)/self.config.get('probability_margin',.15))
            if maximum<self.config.get('minimum_probability',.8) or confidence<.3:continue
            self.statistics['history_anchors']+=1
            self.trusted[(track.track_id,observations[best].evidence['index'])]=confidence
            ir_changed=self.cue=='ir' and abs(float(delta[index,columns[best]]))>1e-4
            if ir_changed:self.statistics['IR_increment_rows']+=1
            if ambiguity is not None and not ambiguity[row].any():
                if (base[row]<=limit).any() or maximum<.95:
                    self.statistics['motion_preserved_rows']+=1;continue
            scores=np.array([o.score for o in observations]) if stage!='secondary' else np.ones(len(observations))
            weight=amount*confidence
            adjusted=(1-weight)*base[row]+weight*(1-p*scores)
            rejected=base[row]>limit
            opening=(p>=self.config.get('opening_probability',.95))&(vote[index,columns]>=.15)
            adjusted[rejected&~opening]=base[row,rejected&~opening]
            result[row]=adjusted
            self.statistics['calibrated_rows']+=1;self.statistics['point_rows']+=1
            self.statistics['point_opened_edges']+=int((rejected&(adjusted<=limit)).sum())
            if ir_changed:self.statistics['IR_applied_rows']+=1;self.statistics['IR_supported_rows']+=1
        return result


class CalibratedCycleTracker(CycleBirthMixin,CalibratedForegroundTracker):
    pass


class RecoverableCalibratedTracker(CycleBirthMixin,WeakRecoveryMixin,CalibratedForegroundTracker):
    pass
