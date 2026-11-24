"""Train-only supervision with one frozen RGB SimOTA assignment for both arms.

Loss-difference targets are detached local teacher signals. They are not a
calibrated uncertainty, causal effect or inference-time use of annotations.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class FixedAssignment:
    foreground:torch.Tensor
    classes:torch.Tensor
    boxes:torch.Tensor


@torch.no_grad()
def reference_assignment(head,reference,metadata,targets,image_hw):
    if head.use_l1 or reference.ndim!=3 or reference.shape[-1]!=5+head.num_classes or reference.requires_grad:
        raise ValueError('Detached original no-L1 YOLOX outputs required')
    if targets.ndim!=3 or targets.shape[0]!=len(reference) or targets.shape[-1]!=5:
        raise ValueError('Corresponding Train targets required')
    batch,anchors,_=reference.shape
    foreground=torch.zeros((batch,anchors),device=reference.device,dtype=torch.bool)
    classes=reference.new_zeros((batch,anchors,head.num_classes))
    boxes=reference.new_zeros((batch,anchors,4))
    strides=torch.cat(metadata['strides'],1);x=torch.cat(metadata['x_shifts'],1);y=torch.cat(metadata['y_shifts'],1)
    images=reference.new_empty((batch,0,*image_hw))
    for index in range(batch):
        labels=targets[index];labels=labels[(labels[:,3]>0)&(labels[:,4]>0)]
        if not len(labels):continue
        matched,mask,iou,indices,count=head.get_assignments(index,len(labels),anchors,labels[:,1:5],labels[:,0],
            reference[index,:,:4],strides,x,y,reference[:,:,5:],reference[:,:,:4],reference[:,:,4:5],targets,images)
        if int(mask.sum())!=count:raise ValueError('Original assignment mask/count differs')
        foreground[index]=mask
        classes[index,mask]=F.one_hot(matched.long(),head.num_classes).to(reference)*iou[:,None]
        boxes[index,mask]=labels[indices,1:5]
    return FixedAssignment(foreground,classes,boxes)


def fixed_anchor_cost(head,prediction,assignment):
    mask=assignment.foreground
    if prediction.shape[:2]!=mask.shape or prediction.shape[2]!=5+head.num_classes:
        raise ValueError('Same original dense anchors and class count required')
    cost=F.binary_cross_entropy_with_logits(prediction[:,:,4],mask.to(prediction),reduction='none')
    if bool(mask.any()):
        selected=5*head.iou_loss(prediction[:,:,:4][mask],assignment.boxes[mask])
        selected=selected+F.binary_cross_entropy_with_logits(prediction[:,:,5:][mask],assignment.classes[mask],reduction='none').sum(-1)
        extra=torch.zeros_like(cost);extra[mask]=selected;cost=cost+extra
    return cost


def utility_targets(reference_cost,proposal_cost,temperature=.1):
    if reference_cost.shape!=proposal_cost.shape or temperature<=0 or not bool(torch.isfinite(reference_cost).all() & torch.isfinite(proposal_cost).all()):
        raise ValueError('Corresponding finite local costs and positive temperature required')
    advantage=(reference_cost-proposal_cost).detach()
    target=torch.sigmoid(advantage/temperature)
    weight=(advantage.abs()/temperature).clamp_max(1).detach()
    return target.detach(),weight,advantage


def balanced_loss(logits,target,valid,foreground,weight=None):
    if not (logits.shape==target.shape==valid.shape==foreground.shape) or valid.dtype!=torch.bool or foreground.dtype!=torch.bool:
        raise ValueError('Corresponding task logits, labels and boolean masks required')
    if weight is None:weight=torch.ones_like(logits)
    if weight.shape!=logits.shape or not bool(torch.isfinite(weight).all()) or bool((weight<0).any()):
        raise ValueError('Finite nonnegative matching weights required')
    loss=F.binary_cross_entropy_with_logits(logits,target.detach(),reduction='none')
    terms=[]
    for region in [foreground,~foreground]:
        used=valid & region;mass=weight[used].detach()
        if bool(mass.sum()>0):terms.append((loss[used]*mass).sum()/mass.sum())
    return torch.stack(terms).mean() if terms else logits.sum()*0


def box_occupancy(targets,shape,stride,device):
    batch,_,height,width=shape
    yy,xx=torch.meshgrid((torch.arange(height,device=device)+.5)*stride,
        (torch.arange(width,device=device)+.5)*stride,indexing='ij')
    output=torch.zeros(shape,device=device,dtype=torch.bool)
    for index in range(batch):
        for label in targets[index]:
            _,x,y,w,h=label
            if bool((w>0)&(h>0)):
                output[index,0]|=(xx>=x-w/2)&(xx<x+w/2)&(yy>=y-h/2)&(yy<y+h/2)
    return output


def task_supervision(head,reference,proposal,metadata,targets,diagnostics,image_hw,temperature=.1):
    assignment=reference_assignment(head,reference.detach(),metadata,targets,image_hw)
    old=fixed_anchor_cost(head,reference.detach(),assignment)
    new=fixed_anchor_cost(head,proposal.detach(),assignment)
    target,weight,advantage=utility_targets(old,new,temperature)
    offset=0;foreground_losses=[];utility_losses=[];rows=[]
    for stride,row in zip(head.strides,diagnostics['levels']):
        foreground_logits=row['foreground_logits'];utility_logits=row['utility_logits'];valid=row['valid']
        shape=foreground_logits.shape;count=shape[-2]*shape[-1]
        occupancy=box_occupancy(targets,shape,stride,foreground_logits.device)
        truth=target[:,offset:offset+count].reshape_as(foreground_logits)
        mass=weight[:,offset:offset+count].reshape_as(foreground_logits)
        matched=assignment.foreground[:,offset:offset+count].reshape_as(foreground_logits)
        delta=advantage[:,offset:offset+count].reshape_as(foreground_logits)
        foreground_losses.append(balanced_loss(foreground_logits,occupancy.float(),valid,occupancy))
        utility_losses.append(balanced_loss(utility_logits,truth,valid,matched,mass))
        rows.append(dict(stride=stride,valid_cells=int(valid.sum()),box_foreground_cells=int((valid&occupancy).sum()),
            reference_matched_cells=int((valid&matched).sum()),helpful_cells=int(((delta>0)&valid).sum()),
            harmful_cells=int(((delta<0)&valid).sum()),equal_cells=int(((delta==0)&valid).sum()),
            utility_label_mass=float(mass[valid].sum()),mean_signed_local_advantage=float(delta[valid].mean()) if bool(valid.any()) else None))
        offset+=count
    if offset!=reference.shape[1]:raise ValueError('FPN task logits do not cover original anchor order')
    return torch.stack(foreground_losses).mean(),torch.stack(utility_losses).mean(),rows
