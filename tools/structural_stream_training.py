"""Live Train inputs and the unchanged structural loss for a future coverage study.

This module does not launch fitting, choose a model or read Dev/Test labels.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from tools.build_correlation_cache import auxiliary_input
from tools.run_pseudo_ir_comparison import sha256
from tools.structural_grid import structural_grid
from yolox.models.frozen_head_repair import head_predictions,original_supervised_loss
from yolox.models.recoverable_curriculum import valid_counterfactual_locations
from yolox.models.structural_factorial_repair import StructuralFactorial
from yolox.models.structural_relative_repair import relative_semantics
from yolox.models.structural_relative_losses import preserve_semantics
from yolox.models.yolo_head import YOLOXHead
from yolox.tracker.foveated_regions import exact_half_preproc


@dataclass
class TrainBatch:
    rgb: tuple
    auxiliary: tuple
    target: torch.Tensor
    structure: tuple
    frames: list


def initialize_head_adapter(saved,architecture,components,device):
    """Caller sets seed0; preserve the legacy head-then-adapter RNG order."""
    head=YOLOXHead(1,width=1.25)
    head.load_state_dict({k.removeprefix('head.'):v for k,v in saved['model'].items()if k.startswith('head.')},strict=True)
    head=head.to(device).float().eval().requires_grad_(False)
    a=architecture
    adapter=StructuralFactorial(latent=a['latent_channels'],radius=a['local_radius'],
        temperature=a['structural_temperature'],bound=a['feature_residual_bound_RMS_fraction'],
        risk_threshold=a['teacher_risk_threshold'],components=components,
        tolerance=a['forward_backward_tolerance_cells']).to(device)
    return head,adapter


def initialize_encoder(saved,experiment,device):
    """Adding a frozen backbone must not change adapter initialization or RNG."""
    from yolox.exp import get_exp
    with torch.random.fork_rng(devices=[]):
        model=get_exp(str(experiment),None).get_model()
    model.load_state_dict(saved['model'],strict=True)
    encoder=model.backbone.to(device).half().eval().requires_grad_(False)
    return encoder


class StreamingTrainInputs:
    def __init__(self,plan,protocol,config,calibration,schedule,encoder,device,feature_backend=None):
        self.plan,self.protocol,self.config,self.calibration=plan,protocol,config,calibration
        self.encoder,self.device,self.feature_backend=encoder,torch.device(device),feature_backend
        self.schedule=schedule
        if set(schedule)!=set(protocol['train']):raise ValueError('Only the fixed six Train videos are allowed')
        self.targets={};self.gt_hashes={};self.draw_hash=hashlib.sha256();self.prepared=0
        for sequence in protocol['train']:
            path=Path(plan['evaluation']['GT_root'])/'train'/sequence/'gt/gt.txt'
            digest=sha256(path)
            if digest!=protocol['derived_gt_sha256']['train/'+sequence]:raise ValueError('Train GT changed')
            gt=np.loadtxt(path,delimiter=',',ndmin=2)
            self.targets[sequence]=gt[(gt[:,6]>0)&(gt[:,7]==1)]
            self.gt_hashes[sequence]=digest

    def addresses(self,draw,epoch):
        if not 1<=epoch<=24:raise ValueError('Registered24cache-equivalent passes only')
        result=[]
        for sequence,slot in draw:
            if sequence not in self.schedule or not 0<=slot<192:raise ValueError('Invalid Train slot')
            result.append((sequence,int(self.schedule[sequence][epoch-1,slot])))
        return result

    def _images(self,sequence,frame,cue):
        rgb=cv2.imread(str(Path(self.plan['data_root'])/'train'/sequence/'visible'/f'{frame:06d}.jpg'))
        if rgb is None:raise ValueError('Required original Train RGB missing')
        ir=None
        if cue=='ir':
            path=Path(self.plan['generated_ir_root'])/'train'/sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
            ir=cv2.imread(str(path),-1)
            if ir is None or ir.ndim!=2 or ir.dtype!=np.uint8 or ir.shape!=rgb.shape[:2]:
                raise ValueError('Required actual November Train IR missing or changed')
        return rgb,ir

    def _encode(self,sequence,frame,cue,value):
        with torch.no_grad():
            if self.feature_backend is None:
                result=self.encoder(torch.from_numpy(value[None]).to(self.device))
                result=tuple(v[0].float()for v in result)
            else:
                # Engineering CPU parity may replay previously GPU-verified B1
                # cache slices. This is explicitly not live CPU/GPU encoding.
                result=tuple(v.to(self.device).float()for v in self.feature_backend(sequence,frame,cue,value))
        if any(v.is_inference()or v.requires_grad for v in result):
            raise ValueError('Frozen inputs must be normal no-grad tensors, not inference tensors')
        return result

    def prepare(self,draw,cue,epoch):
        if cue not in ['rgb','gray','ir']or len(draw)!=4:raise ValueError('Matched batch4 and explicit cue required')
        frames=self.addresses(draw,epoch);rgb_rows=[];aux_rows=[];targets=[];q=[];k=[];qv=[];kv=[]
        for sequence,frame in frames:
            rgb,ir=self._images(sequence,frame,cue);gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
            value,ratio=exact_half_preproc(rgb,self.config['training_input_hw'],(.485,.456,.406),(.229,.224,.225))
            r=self._encode(sequence,frame,'rgb',value);rgb_rows.append(r)
            if cue=='rgb':aux_rows.append(r)
            else:
                norm=self.calibration['statistics'][cue]
                value,_=auxiliary_input(ir if cue=='ir'else gray,self.config['training_input_hw'],norm['mean'],norm['std'])
                aux_rows.append(self._encode(sequence,frame,cue,value))
            query,query_valid=structural_grid(gray,self.config['training_input_hw'])
            key,key_valid=structural_grid(ir,self.config['training_input_hw'])if cue=='ir'else(query,query_valid)
            q.append(query);k.append(key);qv.append(query_valid);kv.append(key_valid)
            target=np.zeros((500,5),np.float32)
            gt=self.targets[sequence];boxes=gt[gt[:,0]==frame,2:6].astype(np.float32).copy()
            boxes[:,:2]+=boxes[:,2:]/2;boxes*=ratio
            if len(boxes)>500:raise ValueError('Original maximum target count exceeded')
            target[:len(boxes),1:]=boxes;targets.append(target)
        batch=lambda rows:tuple(torch.stack([row[level]for row in rows])for level in range(3))
        transfer=lambda rows:torch.from_numpy(np.stack(rows)).to(self.device)
        self.draw_hash.update(json.dumps(frames,separators=(',',':')).encode());self.prepared+=1
        return TrainBatch(batch(rgb_rows),batch(aux_rows),transfer(targets),
                          tuple(transfer(v)for v in [q,k,qv,kv]),frames)


def structural_loss(adapter,head,batch,config):
    """Same obj/class, reliable RGB and class-valid wrong-frame losses as run_fit."""
    rgb,auxiliary,target=batch.rgb,batch.auxiliary,batch.target
    query,key,query_valid,key_valid=batch.structure
    with torch.no_grad():reference,meta=head_predictions(head,rgb)
    def predict(view,descriptor,validity,null=False):
        repaired=adapter(rgb,view,meta['scores'],query,descriptor,query_valid,validity,null=null)
        return head_predictions(head,repaired)[0]
    current=predict(auxiliary,key,key_valid)
    null=predict(auxiliary,key,key_valid,null=True)
    prediction=relative_semantics(current,null,reference,centered=adapter.centered,
                                 bound=config['architecture']['semantic_logit_bound'])
    assert torch.equal(prediction[:,:,:4],reference[:,:,:4])
    initial=bool(torch.equal(current,reference)and torch.equal(null,reference)and torch.equal(prediction,reference))
    parts=original_supervised_loss(head,prediction,meta,target,config['training_input_hw'])
    supervised=parts[2]+parts[3]
    reliable=preserve_semantics(prediction,reference,reliable_only=True)
    permutation=torch.tensor([1,0,3,2],device=target.device)
    wrong_view=[v[permutation]for v in auxiliary]
    wrong_current=predict(wrong_view,key[permutation],key_valid[permutation])
    wrong_null=predict(wrong_view,key[permutation],key_valid[permutation],null=True)
    wrong_prediction=relative_semantics(wrong_current,wrong_null,reference,centered=adapter.centered,
                                       bound=config['architecture']['semantic_logit_bound'])
    validity=valid_counterfactual_locations(target,permutation,meta)
    wrong=preserve_semantics(wrong_prediction,reference,validity)
    fit=config['fit'];loss=supervised+fit['lambda_reliable_RGB']*reliable+fit['lambda_wrong_auxiliary']*wrong
    return loss,(supervised,reliable,wrong),initial


def optimizer_step(adapter,head,batch,config,optimizer,position):
    fit=config['fit'];steps=288
    factor=((position+1)/steps if position<steps else
            .03+.97*.5*(1+math.cos(math.pi*(position-steps)/((config['epochs']-1)*steps))))
    for group in optimizer.param_groups:group['lr']=fit['learning_rate']*factor
    adapter.train();loss,parts,initial=structural_loss(adapter,head,batch,config)
    if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite streaming structural loss')
    optimizer.zero_grad(set_to_none=True);loss.backward()
    gradient=torch.nn.utils.clip_grad_norm_(adapter.parameters(),fit['gradient_clip_norm'],error_if_nonfinite=True)
    optimizer.step()
    return dict(values=[float(v.detach())for v in (loss,*parts,gradient)],initial_raw_RGB_head_exact=initial)
