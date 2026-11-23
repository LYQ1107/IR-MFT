"""Train-only shared recovery perturbations and class-valid counterfactual masks."""
import numpy as np
import torch
from torch.nn import functional as F


def recovery_draw(targets, eligible, seed, epoch, step, sample_probability=.5, instance_probability=.5):
    """A draw depends on labels/eligibility and its seed, never the auxiliary cue."""
    rng=np.random.default_rng(np.random.SeedSequence([20261123,int(seed),int(epoch),int(step)]))
    eligible=np.asarray(eligible,dtype=bool)
    if eligible.shape!=tuple(targets.shape[:2]):raise ValueError('Eligibility and original targets differ')
    foreground=targets.detach().cpu().numpy()[:,:,3:5].min(-1)>0
    samples=rng.random((len(eligible),1))<sample_probability
    instances=rng.random(eligible.shape)<instance_probability
    return eligible & foreground & samples & instances


def weaken_recoverable_features(features, targets, selected, image_hw, strength=.75, kernel=9):
    """Blend local spatial means only inside selected Train boxes; keep input intact."""
    if not 0<=strength<=1 or kernel%2!=1:raise ValueError('Bounded strength and odd pool required')
    if not np.asarray(selected).any():return tuple(features)
    boxes=targets.detach().cpu().numpy();result=[]
    for value in features:
        h,w=value.shape[-2:];mask=value.new_zeros((len(value),1,h,w))
        for batch,obj in np.argwhere(selected):
            cx,cy,bw,bh=boxes[batch,obj,1:5]
            x1=max(0,int(np.floor((cx-bw/2)*w/image_hw[1])))
            x2=min(w,int(np.ceil((cx+bw/2)*w/image_hw[1])))
            y1=max(0,int(np.floor((cy-bh/2)*h/image_hw[0])))
            y2=min(h,int(np.ceil((cy+bh/2)*h/image_hw[0])))
            mask[batch,:,y1:y2,x1:x2]=1
        background=F.avg_pool2d(value,kernel,stride=1,padding=kernel//2,count_include_pad=False)
        result.append(torch.where(mask.bool(),value*(1-strength)+background*strength,value))
    return tuple(result)


def foreground_locations(targets, metadata):
    """Original head cell centers inside any labeled foreground box, Train only."""
    locations=[]
    for x,y,stride in zip(metadata['x_shifts'],metadata['y_shifts'],metadata['strides']):
        centers=torch.stack([(x+.5)*stride,(y+.5)*stride],-1)
        contained=torch.zeros((len(targets),centers.shape[1]),dtype=torch.bool,device=targets.device)
        for batch,rows in enumerate(targets):
            rows=rows[(rows[:,3]>0)&(rows[:,4]>0)]
            if not len(rows):continue
            lower=rows[:,1:3]-rows[:,3:5]/2;upper=rows[:,1:3]+rows[:,3:5]/2
            contained[batch]=((centers[0,:,None]>=lower)&(centers[0,:,None]<=upper)).all(-1).any(-1)
        locations.append(contained)
    return torch.cat(locations,1)


def valid_counterfactual_locations(targets, permutation, metadata):
    foreground=foreground_locations(targets,metadata)
    return ~(foreground & foreground[permutation])


def preserve_valid_reference(predictions, reference, strides, validity):
    """Exclude jointly valid class foreground without renormalizing the remainder."""
    if validity.shape!=predictions.shape[:2]:raise ValueError('One validity flag per original head cell required')
    reference=reference.detach();p=reference[:,:,4:].sigmoid();probability=p[:,:,0]*p[:,:,1]
    divergence=p*(F.logsigmoid(reference[:,:,4:])-F.logsigmoid(predictions[:,:,4:]))
    divergence+=(1-p)*(F.logsigmoid(-reference[:,:,4:])-F.logsigmoid(-predictions[:,:,4:]))
    weight=validity.to(probability);denominator=probability.sum().clamp_min(1.)
    classification=((divergence[:,:,0]+divergence[:,:,1]*probability)*weight).sum()/denominator
    scale=torch.cat(strides,1).to(predictions)[:,:,None]
    distance=F.smooth_l1_loss(predictions[:,:,:4]/scale,reference[:,:,:4]/scale,reduction='none').mean(-1)
    geometry=(distance*probability*weight).sum()/denominator
    return classification.clamp_min(0)+geometry
