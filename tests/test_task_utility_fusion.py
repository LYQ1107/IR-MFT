import inspect

import pytest
import torch

from tools.native_pyramid_training_loop import state_sha256
from tools.task_utility_fusion import TaskUtilityFusion
from tools.task_utility_objectives import (FixedAssignment,balanced_loss,box_occupancy,
    fixed_anchor_cost,reference_assignment,task_supervision,utility_targets)
from yolox.models.frozen_head_repair import head_predictions,original_supervised_loss
from yolox.models.yolo_head import YOLOXHead
from yolox.models.native_pyramid_fusion import NativePyramidFusion


def sample(batch=2):
    torch.manual_seed(0)
    rgb=tuple(torch.randn(batch,c,64//s,96//s) for c,s in zip((8,16,32),(8,16,32)))
    return rgb,torch.randn(batch,3,64,96),tuple(torch.full_like(v[:,:1],.3) for v in rgb)


def test_all_removals_have_identical_initial_tensors_and_exact_RGB():
    rgb,cue,scores=sample();states=[];counts=[]
    for fg,utility in [(False,False),(True,False),(False,True),(True,True)]:
        torch.manual_seed(0);m=TaskUtilityFusion(channels=(8,16,32),use_foreground=fg,use_utility=utility)
        outputs,_,d=m(rgb,cue,scores,return_proposals=True)
        assert all(torch.equal(a,b) for a,b in zip(rgb,outputs))
        assert all(not v.requires_grad for v in d['detached_proposal_features'])
        states.append(state_sha256(m.state_dict()));counts.append(sum(p.numel() for p in m.parameters()))
    assert len(set(states))==len(set(counts))==1
    assert 'targets' not in inspect.signature(TaskUtilityFusion.forward).parameters


def test_zero_task_logits_preserve_nonzero_quality_parent_exactly():
    rgb,cue,scores=sample();parent=NativePyramidFusion(channels=(8,16,32))
    with torch.no_grad():
        for layer in parent.output:layer.weight.normal_(0,.05)
    child=TaskUtilityFusion(channels=(8,16,32));loaded=child.load_state_dict(parent.state_dict(),strict=False)
    assert not loaded.unexpected_keys and all(n.startswith('task.') for n in loaded.missing_keys)
    assert all(torch.equal(a,b) for a,b in zip(parent(rgb,cue,scores)[0],child(rgb,cue,scores)[0]))


def test_missing_does_not_read_encoder_and_partial_observations_exclude_labels(monkeypatch):
    rgb,cue,scores=sample();m=TaskUtilityFusion(channels=(8,16,32))
    with monkeypatch.context() as context:
        context.setattr(m.encoder,'forward',lambda *a:(_ for _ in ()).throw(AssertionError('read unavailable cue')))
        assert all(torch.equal(a,b) for a,b in zip(rgb,m(rgb,None,scores,available=False)[0]))
    with torch.no_grad():
        for layer in m.output:layer.weight.normal_(0,.1)
    outputs,_,d=m(rgb,cue,scores,available=[True,False])
    assert all(torch.equal(a[1],b[1]) for a,b in zip(rgb,outputs))
    assert all(not bool(row['valid'][1].any()) for row in d['levels'])


def test_protection_and_channel_bound_hold_with_nonzero_task_heads():
    rgb,cue,scores=sample();m=TaskUtilityFusion(channels=(8,16,32))
    scores=list(scores);scores[0][:,:,0,0]=.8
    with torch.no_grad():
        for layer in m.output:layer.weight.normal_(0,1)
        for task in m.task:task.foreground.weight.normal_();task.utility.weight.normal_()
    outputs,_,d=m(rgb,cue,scores)
    assert torch.equal(outputs[0][:,:,0,0],rgb[0][:,:,0,0])
    for base,new,score,row in zip(rgb,outputs,scores,d['levels']):
        scale=base.square().mean(1,keepdim=True).add(1e-6).sqrt()
        assert bool(((new-base).abs()<=.5*scale*(1-score)*(score<.6)+1e-6).all())
        assert bool(((row['gate']>0)&(row['gate']<1)).all())


def real_head_fixture():
    torch.manual_seed(0);head=YOLOXHead(1,width=.125).eval().requires_grad_(False)
    features=tuple(torch.randn(1,c,64//s,96//s) for c,s in zip((32,64,128),(8,16,32)))
    with torch.no_grad():reference,metadata=head_predictions(head,features)
    targets=torch.zeros(1,3,5);targets[0,0]=torch.tensor([0,40,28,16,12])
    return head,reference,metadata,targets


def test_local_cost_recomposes_original_YOLOX_loss_at_same_reference():
    head,reference,metadata,targets=real_head_fixture()
    assignment=reference_assignment(head,reference,metadata,targets,(64,96))
    cost=fixed_anchor_cost(head,reference,assignment).sum()/assignment.foreground.sum().clamp_min(1)
    original=original_supervised_loss(head,reference,metadata,targets,(64,96))[0]
    torch.testing.assert_close(cost,original,rtol=1e-6,atol=1e-6)
    assert all(not v.requires_grad for v in [assignment.foreground,assignment.classes,assignment.boxes])


def test_known_geometry_and_objectness_improvements_get_positive_utility():
    head=YOLOXHead(1,width=.125).eval()
    assignment=FixedAssignment(torch.tensor([[True]]),torch.ones(1,1,1),torch.tensor([[[10.,10.,4.,4.]]]))
    reference=torch.tensor([[[12.,10.,4.,4.,0.,0.]]],requires_grad=True)
    helpful=torch.tensor([[[10.,10.,4.,4.,3.,3.]]],requires_grad=True)
    harmful=torch.tensor([[[16.,10.,4.,4.,-3.,-3.]]],requires_grad=True)
    old=fixed_anchor_cost(head,reference,assignment)
    positive,weight,delta=utility_targets(old,fixed_anchor_cost(head,helpful,assignment))
    negative,_,_=utility_targets(old,fixed_anchor_cost(head,harmful,assignment))
    assert positive.item()>.5 and negative.item()<.5 and weight.item()>0 and delta.item()>0
    assert not positive.requires_grad and not weight.requires_grad and not delta.requires_grad


def test_equal_proposal_has_no_utility_supervision_or_fake_gradient():
    cost=torch.tensor([[1.,2.]],requires_grad=True);target,mass,delta=utility_targets(cost,cost)
    logits=torch.zeros_like(cost,requires_grad=True);valid=torch.ones_like(cost,dtype=torch.bool)
    loss=balanced_loss(logits,target,valid,valid,mass);loss.backward()
    assert loss.item()==0 and not bool(logits.grad.any()) and not bool(mass.any()) and not bool(delta.any())


def test_empty_GT_and_all_invalid_cells_are_supported():
    head,reference,metadata,targets=real_head_fixture();targets.zero_()
    assignment=reference_assignment(head,reference,metadata,targets,(64,96))
    assert not bool(assignment.foreground.any())
    logits=torch.randn(2,requires_grad=True);valid=torch.zeros(2,dtype=torch.bool)
    loss=balanced_loss(logits,torch.zeros(2),valid,valid);loss.backward()
    assert loss.item()==0 and not bool(logits.grad.any())


def test_box_targets_use_half_open_cell_centers_and_padding_is_not_supervised():
    targets=torch.tensor([[[0.,16.,16.,16.,16.]]])
    occupancy=box_occupancy(targets,(1,1,4,4),8,'cpu')
    expected=torch.zeros(1,1,4,4,dtype=torch.bool);expected[:,:,1:3,1:3]=True
    assert torch.equal(occupancy,expected)
    logits=torch.tensor([0.,2.,-2.]);valid=torch.tensor([True,False,False]);foreground=torch.tensor([True,True,False])
    loss=balanced_loss(logits,torch.ones(3),valid,foreground)
    torch.testing.assert_close(loss,torch.tensor(2.).log())


def test_auxiliary_supervision_keeps_assignment_once_and_labels_detached(monkeypatch):
    head,reference,metadata,targets=real_head_fixture();calls=[];original=head.get_assignments
    def counted(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(head,'get_assignments',counted)
    rows=[]
    for h,w in [(8,12),(4,6),(2,3)]:
        rows.append(dict(foreground_logits=torch.zeros(1,1,h,w,requires_grad=True),
            utility_logits=torch.zeros(1,1,h,w,requires_grad=True),valid=torch.ones(1,1,h,w,dtype=torch.bool)))
    foreground,utility,stats=task_supervision(head,reference,reference,metadata,targets,dict(levels=rows),(64,96))
    assert len(calls)==1 and utility.item()==0 and all(r['utility_label_mass']==0 for r in stats)
    assert foreground.requires_grad and all(r['helpful_cells']==r['harmful_cells']==0 for r in stats)


@pytest.mark.parametrize('temperature',[0,-1])
def test_invalid_utility_temperature_rejected(temperature):
    with pytest.raises(ValueError):utility_targets(torch.zeros(1),torch.zeros(1),temperature)
