"""A cue correction around an immutable, fitted RGB association scorer."""
import json
from pathlib import Path

import numpy as np
import torch

from yolox.tracker.association_calibrator import AssociationCalibrator,prepare_calibration_cue
from yolox.tracker.association_features import pair_features
from yolox.tracker.calibrated_foreground_tracker import CalibratedForegroundTracker
from yolox.tracker.candidate_foreground_tracker import CandidateForegroundTracker
from yolox.tracker.rgb_reference_context import extend_rgb_context


class StrongRGBForegroundTracker(CalibratedForegroundTracker):
    def __init__(self,config,model=None,frozen_RGB=None,cue=None):
        config=dict(dict(source_mode='rgb',target_mode='rgb',lags=[1,2,4,8]),**config)
        if config['source_mode']!='rgb' or config['target_mode']!='rgb':
            raise ValueError('The fitted RGB reference uses the preserved RGB point stream')
        CandidateForegroundTracker.__init__(self,config)
        if model is None:
            checkpoint=torch.load(config['association_checkpoint'],map_location='cpu',weights_only=False)
            root=Path(__file__).resolve().parents[2]
            protocol=json.loads((root/'configs/cfp/protocol_v2.json').read_text())
            if checkpoint['fit_sequences']!=protocol['train'] or checkpoint['Dev_fit_rows'] or checkpoint['Test_fit_rows']:
                raise ValueError('Cue residual violates Train-only fitting')
            for option,field in [('association_plan_sha256','config_sha256'),
                                 ('association_prior_sha256','prior_plan_sha256'),
                                 ('association_capacity_sha256','capacity_plan_sha256'),
                                 ('RGB_base_selection_sha256','base_selection_sha256')]:
                if config.get(option,checkpoint[field])!=checkpoint[field]:
                    raise ValueError('Strong-RGB residual provenance mismatch: '+field)
            model=AssociationCalibrator(**checkpoint['architecture'])
            model.load_state_dict(checkpoint['model'],strict=True)
            frozen_RGB=AssociationCalibrator(**checkpoint['base_architecture'])
            frozen_RGB.load_state_dict(checkpoint['base_model'],strict=True)
            self.cue=checkpoint['cue']
            self.calibration_metadata={k:checkpoint[k] for k in ['cue','seed','epoch','sources','fit_sequences',
                'config_sha256','prior_plan_sha256','capacity_plan_sha256','base_metadata','base_frontend','base_selection_sha256']}
        else:
            if frozen_RGB is None or cue not in {'rgb','ir'}:
                raise ValueError('Injected residual requires an explicit frozen RGB model and cue')
            self.cue=cue
            self.calibration_metadata=dict(cue=cue,scope='injected engineering-test models')
        if model.base_dimensions!=26 or frozen_RGB.base_dimensions!=23:
            raise ValueError('Residual and fitted reference feature dimensions do not match')
        if config.get('calibration_cue',self.cue)!=self.cue:raise ValueError('Checkpoint cue mismatch')
        self.model=model.cpu().eval();self.frozen_RGB=frozen_RGB.cpu().eval()
        for parameter in self.frozen_RGB.parameters():parameter.requires_grad_(False)
        self.pair_cache={}
        self.statistics.update(calibrated_rows=0,IR_increment_rows=0,IR_applied_rows=0,motion_preserved_rows=0)

    def score_features(self,base,RGB_extra,extra,reference,available):
        """Score complete current-frame candidate rows, then add the new cue.

        The inherited fitted RGB increment is always retained. Disabling this
        module or losing the auxiliary cue zeros only the newly added residual.
        """
        n,m=base.shape[:2];count=n*m
        if not count:return np.zeros((n,m),np.float32),np.zeros((n,m),np.float32)
        RGB_extra=prepare_calibration_cue(RGB_extra,'rgb')
        extra=prepare_calibration_cue(extra,self.cue)
        if self.config.get('shuffle_cue',False):extra=np.roll(extra,1,axis=1)
        logits=np.zeros(count,np.float32)
        with torch.inference_mode():
            for lo in range(0,count,4096):
                bs=torch.from_numpy(base.reshape(-1,23)[lo:lo+4096])
                cs=torch.from_numpy(RGB_extra.reshape(-1,17)[lo:lo+4096])
                rs=torch.from_numpy(reference.reshape(-1,17)[lo:lo+4096])
                valid=torch.ones(len(bs),dtype=torch.bool)
                logits[lo:lo+len(bs)]=self.frozen_RGB(bs,cs,rs,valid).numpy()
            probability=torch.from_numpy(logits).sigmoid().numpy()
            context=extend_rgb_context(base.reshape(-1,23),probability,np.repeat(np.arange(n),m))
            increments=np.zeros(count,np.float32)
            if not self.config.get('disable_learned_increment',False):
                for lo in range(0,count,4096):
                    bs=torch.from_numpy(context[lo:lo+4096])
                    cs=torch.from_numpy(extra.reshape(-1,17)[lo:lo+4096])
                    rs=torch.from_numpy(reference.reshape(-1,17)[lo:lo+4096])
                    valid=torch.from_numpy(available.ravel()[lo:lo+4096])
                    increments[lo:lo+len(bs)]=self.model.increment(bs,cs,rs,valid).numpy()
            probability=torch.from_numpy(logits+increments).sigmoid().numpy()
        return probability.reshape(n,m),increments.reshape(n,m)

    def predict_pairs(self,frame,lag):
        if lag in self.pair_cache:return self.pair_cache[lag]
        data=self.data;point=data['foreground_transport']
        a,b=map(int,data['offsets'][frame-1:frame+1]);c,d=map(int,data['offsets'][frame-lag-1:frame-lag+1])
        fields=['boxes','embedding','rgb_mask_embedding','rgb_hog','color','quality']
        if self.cue=='ir':fields+=['ir_mask_embedding','ir_hog']
        past={k:data[k][c:d] for k in fields};current={k:data[k][a:b] for k in fields}
        past['point_quality']=point['quality'][c:d];current['point_quality']=point['quality'][a:b]
        v0,v1=map(int,point[f'vote_offsets_{lag}'][frame-1:frame+1])
        p0,p1=map(int,point[f'count_offsets_{lag}'][frame-1:frame+1])
        modes=['rgb','bbox']+(['joint'] if self.cue=='ir' else [])
        matrices={}
        for mode in modes:
            for source,target in [('rgb',mode),(mode,'rgb')]:
                matrices[(source,target)]=point[f'vote_{lag}_{source}_{target}'][v0:v1].reshape(d-c,b-a)
        good={s:point[f'good_{lag}_{s}'][p0:p1] for s in modes}
        total={s:point[f'total_{lag}_{s}'][p0:p1] for s in modes}
        base,RGB_extra,reference=pair_features(past,current,matrices,good,total,lag,'rgb')
        if self.cue=='ir':
            _,extra,_=pair_features(past,current,matrices,good,total,lag,'ir')
            available=(past['point_quality'][:,1:2]>=.15)&(current['point_quality'][None,:,1]>=.15)
        else:extra=RGB_extra;available=np.ones((d-c,b-a),bool)
        probability,increments=self.score_features(base,RGB_extra,extra,reference,available)
        value=(c,probability,increments,matrices[('rgb','rgb')],good['rgb'],total['rgb'])
        self.pair_cache[lag]=value
        return value
