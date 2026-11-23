"""Original YOLOX head outputs/losses with frozen weights and trainable inputs."""
import torch
from torch.nn import functional as F


def head_predictions(head,features):
    outputs=[];x_shifts=[];y_shifts=[];strides=[];scores=[]
    for level,(value,stride)in enumerate(zip(features,head.strides)):
        stem=head.stems[level](value)
        classification=head.cls_preds[level](head.cls_convs[level](stem))
        regression=head.reg_convs[level](stem)
        raw=torch.cat([head.reg_preds[level](regression),head.obj_preds[level](regression),classification],1)
        output,grid=head.get_output_and_grid(raw,level,stride,features[0].type())
        outputs.append(output);x_shifts.append(grid[:,:,0]);y_shifts.append(grid[:,:,1])
        strides.append(output.new_full((1,grid.shape[1]),stride))
        probability=output[:,:,4].sigmoid()*output[:,:,5].sigmoid()
        scores.append(probability.reshape(value.shape[0],1,*value.shape[-2:]))
    return torch.cat(outputs,1),dict(x_shifts=x_shifts,y_shifts=y_shifts,strides=strides,scores=scores)


def original_supervised_loss(head,predictions,metadata,targets,image_size):
    images=predictions.new_empty((len(predictions),0,*image_size))
    return head.get_losses(images,metadata['x_shifts'],metadata['y_shifts'],metadata['strides'],
                           targets,predictions,[],dtype=predictions.dtype)


def preserve_reference(predictions,reference,strides,reliable_only=False):
    reference=reference.detach()
    p=reference[:,:,4:].sigmoid()
    probability=p[:,:,0]*p[:,:,1]
    mask=(probability>=.6).to(probability)if reliable_only else torch.ones_like(probability)
    # Objectness is defined for every location; class/box predictions only
    # receive foreground-weighted preservation, avoiding arbitrary BG classes.
    log_difference=F.logsigmoid(reference[:,:,4:])-F.logsigmoid(predictions[:,:,4:])
    other_difference=F.logsigmoid(-reference[:,:,4:])-F.logsigmoid(-predictions[:,:,4:])
    divergence=p*log_difference+(1-p)*other_difference
    denominator=(probability*mask).sum().clamp_min(1.)
    classification=(divergence[:,:,0]*mask+divergence[:,:,1]*probability*mask).sum()/denominator
    scale=torch.cat(strides,1).to(predictions)[:,:,None]
    distance=F.smooth_l1_loss(predictions[:,:,:4]/scale,reference[:,:,:4]/scale,reduction='none').mean(-1)
    geometry=(distance*probability*mask).sum()/denominator
    return classification.clamp_min(0)+geometry
