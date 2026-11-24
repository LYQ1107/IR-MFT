"""Ordered-clip optimization for TaskUtilityFusion, no launch or data selection."""
import torch

from tools.native_pyramid_fit import learning_rate_factor
from tools.task_utility_fusion import TaskUtilityFusion
from tools.task_utility_objectives import task_supervision
from yolox.models.frozen_head_repair import head_predictions,original_supervised_loss,preserve_reference


LOSS_FIELDS=('loss','full_YOLOX','box','objectness','classification','reliable_RGB','foreground','utility')
COUNT_FIELDS=('available_frames','missing_frames','box_foreground_cells','valid_task_cells',
    'helpful_teacher_cells','harmful_teacher_cells','equal_teacher_cells','utility_label_mass')


def predict_clip(module,head,features,metadata,batch):
    """One current-frame cue per address; no temporal or GT argument to model."""
    count=len(batch.address.frames)
    if not 1<=count<=4 or len(batch.rgb)!=count or len(batch.cue)!=count or len(batch.available)!=count:
        raise ValueError('One to four corresponding current observations required')
    values=[];proposals=[];diagnostics=[]
    for index in range(count):
        current=tuple(v[index:index+1] for v in features)
        output,state,information=module(current,batch.cue[index:index+1],
            tuple(v[index:index+1] for v in metadata['scores']),valid=batch.valid[index:index+1],
            available=batch.available[index],return_proposals=True)
        if state is not None:raise ValueError('Task selector must be single-frame')
        values.append(output)
        if information['missing_exact_RGB']:
            proposals.append(current)
            diagnostics.append(dict(levels=[dict(foreground_logits=v[:,:1].new_zeros(v[:,:1].shape),
                utility_logits=v[:,:1].new_zeros(v[:,:1].shape),valid=torch.zeros_like(v[:,:1],dtype=torch.bool)) for v in current]))
        else:
            proposals.append(information.pop('detached_proposal_features'));diagnostics.append(information)
    joined=tuple(torch.cat([v[level] for v in values]) for level in range(3))
    prediction,_=head_predictions(head,joined)
    with torch.no_grad():
        proposal,_=head_predictions(head,tuple(torch.cat([v[level] for v in proposals]) for level in range(3)))
    information=dict(levels=[{key:torch.cat([r['levels'][level][key] for r in diagnostics])
        for key in ['foreground_logits','utility_logits','valid']} for level in range(3)])
    return prediction,proposal,information


def _direct_gradient(loss,parameters):
    if not loss.requires_grad:return 0.
    return sum(float(g.detach().abs().sum()) for g in torch.autograd.grad(loss,tuple(parameters),retain_graph=True,allow_unused=True) if g is not None)


def optimizer_step(module,encoder,head,batch,optimizer,config,position,state=None,previous_gray=None):
    if not isinstance(module,TaskUtilityFusion) or state is not None or previous_gray is not None:
        raise ValueError('Only current-frame TaskUtilityFusion inputs are supported')
    if encoder.training or head.training or any(p.requires_grad for p in (*encoder.parameters(),*head.parameters())):
        raise ValueError('Original RGB encoder/head must be frozen and in evaluation mode')
    supervision=config['task_supervision'];module.train()
    with torch.no_grad():
        features=tuple(v.float() for v in encoder(batch.rgb.to(dtype=next(encoder.parameters()).dtype)))
        reference,metadata=head_predictions(head,features)
    prediction,proposal,diagnostics=predict_clip(module,head,features,metadata,batch)
    initial_exact=torch.equal(prediction,reference);initial_proposal_exact=torch.equal(proposal,reference)
    if config.get('geometry_fixed',False):
        prediction=torch.cat((reference[:,:,:4],prediction[:,:,4:]),-1)
        proposal=torch.cat((reference[:,:,:4],proposal[:,:,4:]),-1)
    foreground,utility,labels=task_supervision(head,reference,proposal,metadata,batch.targets,diagnostics,
        config['training_input_hw'],temperature=supervision['utility_temperature'])
    original=original_supervised_loss(head,prediction,metadata,batch.targets,config['training_input_hw'])
    preservation=preserve_reference(prediction,reference,metadata['strides'],reliable_only=True)
    fg_weight=supervision['lambda_foreground'] if supervision['foreground'] else 0.
    utility_weight=supervision['lambda_utility'] if supervision['utility'] else 0.
    loss=original[0]+config['fit']['lambda_reliable_RGB']*preservation+fg_weight*foreground+utility_weight*utility
    if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite task-utility loss')
    if position==0:
        if not initial_exact or not initial_proposal_exact or any(r['utility_label_mass']!=0 for r in labels):
            raise RuntimeError('Initial current/proposal must exactly match fresh RGB with zero utility mass')
    factor=learning_rate_factor(position,config['fit']['horizon_updates'],config['fit']['warmup_updates'],config['fit']['minimum_lr_fraction'])
    for group in optimizer.param_groups:group['lr']=config['fit']['learning_rate']*factor
    direct_fg=direct_utility=None
    if config.get('audit_supervision_gradients',False):
        direct_fg=_direct_gradient(fg_weight*foreground,(p for t in module.task for p in t.foreground.parameters()))
        direct_utility=_direct_gradient(utility_weight*utility,(p for t in module.task for p in t.utility.parameters()))
    optimizer.zero_grad(set_to_none=True);updated=loss.requires_grad;norm=stem=0.
    if updated:
        loss.backward();g=module.encoder.blocks[0][0].weight.grad
        stem=float(g.detach().abs().sum()) if g is not None else 0.
        norm=float(torch.nn.utils.clip_grad_norm_(module.parameters(),config['fit']['gradient_clip_norm'],error_if_nonfinite=True));optimizer.step()
    if any(p.grad is not None or p.requires_grad for p in (*encoder.parameters(),*head.parameters())):
        raise RuntimeError('Original RGB parameters received training gradients')
    parts=dict(full_YOLOX=original[0],box=original[1],objectness=original[2],classification=original[3],reliable_RGB=preservation,
        foreground=foreground,utility=utility)
    row=dict(loss=float(loss.detach()),**{k:float(v.detach()) for k,v in parts.items()},
        optimizer_updated=updated,gradient_norm=norm,native_stem_gradient_L1=stem,
        initial_full_RGB_exact=initial_exact,initial_proposal_full_RGB_exact=initial_proposal_exact,
        foreground_supervision_direct_gradient_L1=direct_fg,utility_supervision_direct_gradient_L1=direct_utility,
        learning_rate=config['fit']['learning_rate']*factor,frames=len(batch.address.frames),
        available_frames=sum(bool(v) for v in batch.available),missing_frames=sum(not bool(v) for v in batch.available),
        decoded_box_max_change=float((prediction[:,:,:4]-reference[:,:,:4]).detach().abs().max()),
        box_foreground_cells=sum(r['box_foreground_cells'] for r in labels),valid_task_cells=sum(r['valid_cells'] for r in labels),
        helpful_teacher_cells=sum(r['helpful_cells'] for r in labels),harmful_teacher_cells=sum(r['harmful_cells'] for r in labels),
        equal_teacher_cells=sum(r['equal_cells'] for r in labels),utility_label_mass=sum(r['utility_label_mass'] for r in labels))
    return row,None,None
