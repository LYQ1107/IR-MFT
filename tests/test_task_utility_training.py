import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tools.native_pyramid_train_data import TrainClip,PixelClip,address_bytes,epoch_clips
from tools.task_utility_fusion import TaskUtilityFusion
from tools.task_utility_fit import optimizer_step,LOSS_FIELDS,COUNT_FIELDS
from tools.task_utility_training_loop import state_sha256
import tools.task_utility_training_loop as loop
from yolox.models.yolo_head import YOLOXHead


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.ModuleList([nn.Conv2d(3,c,1) for c in [32,64,128]])
    def forward(self,x):
        return tuple(layer(F.avg_pool2d(x,s)) for layer,s in zip(self.layers,[8,16,32]))


def configuration(foreground=True,utility=True):
    return dict(training_input_hw=[64,96],geometry_fixed=False,audit_supervision_gradients=True,
        task_supervision=dict(foreground=foreground,utility=utility,lambda_foreground=.1,lambda_utility=.1,utility_temperature=.1),
        fit=dict(horizon_updates=1000,warmup_updates=2,minimum_lr_fraction=.03,learning_rate=.001,
            lambda_reliable_RGB=.02,gradient_clip_norm=5))


def fixture(available=(True,True)):
    torch.manual_seed(0);encoder=TinyEncoder().eval().requires_grad_(False)
    head=YOLOXHead(1,width=.125).eval().requires_grad_(False)
    module=TaskUtilityFusion(channels=(32,64,128))
    n=len(available);rgb=torch.randn(n,3,64,96);targets=torch.zeros(n,2,5)
    targets[:,0]=torch.tensor([0,40,28,16,12])
    batch=PixelClip(TrainClip('synthetic',tuple(range(1,n+1))),rgb,rgb.clone(),targets,
        torch.ones(n,1,64,96,dtype=torch.bool),available,tuple(None for _ in available))
    optimizer=torch.optim.AdamW(module.parameters(),lr=.001)
    return module,encoder,head,batch,optimizer


def test_clip_supervision_reaches_heads_and_is_in_total_loss():
    module,encoder,head,batch,optimizer=fixture();config=configuration()
    frozen=(state_sha256(encoder.state_dict()),state_sha256(head.state_dict()))
    rows=[]
    for position in range(3):
        row,state,previous=optimizer_step(module,encoder,head,batch,optimizer,config,position)
        assert state is previous is None and set(LOSS_FIELDS).issubset(row)
        assert row['loss']==pytest.approx(row['full_YOLOX']+.02*row['reliable_RGB']+.1*row['foreground']+.1*row['utility'],rel=1e-6)
        rows.append(row)
    assert rows[0]['initial_full_RGB_exact'] and rows[0]['utility_supervision_direct_gradient_L1']==0
    assert rows[0]['foreground_supervision_direct_gradient_L1']>0
    assert any(r['utility_supervision_direct_gradient_L1']>0 for r in rows[1:])
    assert frozen==(state_sha256(encoder.state_dict()),state_sha256(head.state_dict()))


def test_no_auxiliary_labels_preserves_capacity_but_zeroes_aux_gradients():
    module,encoder,head,batch,optimizer=fixture();config=configuration(False,False)
    row,_,_=optimizer_step(module,encoder,head,batch,optimizer,config,0)
    assert row['foreground_supervision_direct_gradient_L1']==row['utility_supervision_direct_gradient_L1']==0
    assert all(not bool(t.foreground.weight.detach().any()) for t in module.task)
    assert row['optimizer_updated']


def test_wholly_missing_clip_skips_AdamW_and_retains_existing_moments():
    module,encoder,head,batch,optimizer=fixture();config=configuration()
    optimizer_step(module,encoder,head,batch,optimizer,config,0)
    before=state_sha256(module.state_dict());moments=copy.deepcopy(optimizer.state_dict()['state'])
    batch.available=(False,False)
    row,_,_=optimizer_step(module,encoder,head,batch,optimizer,config,1)
    assert not row['optimizer_updated'] and row['available_frames']==0 and row['missing_frames']==2
    assert row['valid_task_cells']==row['utility_label_mass']==0 and before==state_sha256(module.state_dict())
    for key,value in moments.items():
        for name,tensor in value.items():assert torch.equal(tensor,optimizer.state_dict()['state'][key][name])


def test_partial_missing_clip_masks_task_labels_and_geometry_removal_is_exact():
    module,encoder,head,batch,optimizer=fixture((True,False));config=configuration();config['geometry_fixed']=True
    row,_,_=optimizer_step(module,encoder,head,batch,optimizer,config,0)
    row2,_,_=optimizer_step(module,encoder,head,batch,optimizer,config,1)
    assert row['available_frames']==row['missing_frames']==1 and row['optimizer_updated']
    assert row['valid_task_cells']==8*12+4*6+2*3
    assert row2['decoded_box_max_change']==0


class SyntheticProvider:
    """Artificial768-frame address schedule, never a real MFT fit."""
    def __init__(self,fail_at=None):
        self.lengths={name:128 for name in ['A','B','C','D','E','F']};self.epoch=None
        self.fail_at=fail_at;self.reads=0
    def begin_epoch(self,epoch):
        self.epoch=epoch;self.prepared=0;self.digest=hashlib.sha256();self.used={s:[] for s in self.lengths}
        self.expected_next=iter(epoch_clips(self.lengths,epoch))
    def prepare(self,clip,cue):
        if self.fail_at==self.prepared:raise RuntimeError('deliberate synthetic failure before next delivery')
        assert next(self.expected_next)==clip;self.digest.update(address_bytes(clip));self.used[clip.sequence].extend(clip.frames)
        self.prepared+=1;self.reads+=1
        return SimpleNamespace(address=clip)
    def finish_epoch(self):
        assert all(sorted(v)==list(range(1,self.lengths[s]+1)) for s,v in self.used.items())
        proof=dict(epoch=self.epoch,address_sha256=self.digest.hexdigest(),frames=768,clips=self.prepared,synthetic=True)
        self.epoch=None;return proof


def synthetic_step(module,encoder,head,batch,optimizer,config,position,state=None,previous_gray=None):
    # Actual AdamW and Torch RNG, but no NN/GT data. Exercise checkpoint state.
    p=next(module.parameters());loss=(p.flatten()[0]-torch.randn(())).square()
    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    row={k:float(loss.detach()) for k in LOSS_FIELDS};row.update({k:1. for k in COUNT_FIELDS})
    row.update(optimizer_updated=True,initial_full_RGB_exact=True)
    return row,None,None


def synthetic_run(tmp_path,name,provider,config_path,config):
    torch.manual_seed(0);module=TaskUtilityFusion();encoder=nn.Linear(1,1).eval().requires_grad_(False)
    head=nn.Linear(1,1).eval().requires_grad_(False);optimizer=torch.optim.AdamW(module.parameters(),lr=.001)
    metadata=dict(family='task_utility',task_supervision=config['task_supervision'],seed=0,cue='ir',
        fit_id='synthetic_not_a_formal_fit',fit_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),source_hashes={})
    histories=loop.run_training_to_epoch(module,encoder,head,optimizer,provider,config,metadata,
        config_path,tmp_path/name,1,tmp_path)
    return module,optimizer,histories


def loop_configuration(tmp_path):
    c=configuration();c.update(budget_and_Dev=dict(stages=[1]),architecture=dict(quality_parameters=553705),
        Train_delivery=dict(clips_per_epoch=192,lengths={name:128 for name in ['A','B','C','D','E','F']}),
        storage=dict(work_cap_GiB=64,minimum_independent_filesystem_free_GiB=40))
    p=tmp_path/'synthetic_config.json';p.write_text(json.dumps(c));return p,c


def test_partial_checkpoint_resume_matches_uninterrupted_weights_rng_and_all_losses(tmp_path,monkeypatch):
    monkeypatch.setattr(loop,'optimizer_step',synthetic_step)
    path,config=loop_configuration(tmp_path)
    model,opt,history=synthetic_run(tmp_path,'uninterrupted',SyntheticProvider(),path,config)
    expected=state_sha256(model.state_dict());next_rng=torch.randn(4)
    failed=SyntheticProvider(fail_at=128)
    with pytest.raises(RuntimeError,match='deliberate synthetic failure'):
        synthetic_run(tmp_path,'resumed',failed,path,config)
    saved=torch.load(tmp_path/'resumed/latest.pt',weights_only=False)
    assert saved['prefix']['delivered_clips']==128 and saved['loop_family']=='task_utility_v1'
    provider=SyntheticProvider();resumed,resumed_opt,resumed_history=synthetic_run(tmp_path,'resumed',provider,path,config)
    assert provider.reads==64 and expected==state_sha256(resumed.state_dict())
    assert torch.equal(next_rng,torch.randn(4)) and resumed_history==history
    assert set(history[0]['mean_clip_losses'])==set(LOSS_FIELDS)
    assert set(history[0]['task_epoch_totals'])==set(COUNT_FIELDS)
    assert all(v==192 for v in history[0]['task_epoch_totals'].values())
    for key,value in opt.state_dict()['state'].items():
        for name,tensor in value.items():assert torch.equal(tensor,resumed_opt.state_dict()['state'][key][name])


def test_legacy_checkpoint_without_task_loss_envelope_is_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(loop,'optimizer_step',synthetic_step);path,config=loop_configuration(tmp_path)
    with pytest.raises(RuntimeError):synthetic_run(tmp_path,'legacy',SyntheticProvider(fail_at=128),path,config)
    p=tmp_path/'legacy/latest.pt';saved=torch.load(p,weights_only=False);saved.pop('loop_family');torch.save(saved,p)
    with pytest.raises(ValueError,match='old loop'):
        synthetic_run(tmp_path,'legacy',SyntheticProvider(),path,config)
