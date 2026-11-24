"""Trainable native encoder, full detector geometry loss and online memory.

These functions do not launch a GPU, read Dev/Test, select a checkpoint or
change the older pinned structural pipeline.
"""
import math

import torch

from tools.native_pyramid_train_data import current_to_previous_flow, detach_state
from yolox.models.frozen_head_repair import head_predictions, original_supervised_loss, preserve_reference


def learning_rate_factor(position, horizon, warmup=500, floor=.03):
    if not 0 <= position < horizon or not 0 < warmup < horizon:
        raise ValueError('Position must be inside the fixed full-study horizon')
    if position < warmup:
        return (position + 1) / warmup
    return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * (position - warmup) / (horizon - warmup)))


def predict_clip(module, head, features, reference_metadata, batch, state=None, previous_gray=None):
    """Stop gradient through each past state; retain forward memory across clips."""
    outputs, previous = [], previous_gray
    for index, frame in enumerate(batch.address.frames):
        if state is not None and state.sequence != batch.address.sequence:
            state, previous = None, None
        flow = None
        if module.mode == 'temporal' and module.use_memory and module.use_motion and state is not None:
            flow = current_to_previous_flow(batch.flow_gray[index], previous)
            if flow is None:
                raise ValueError('Previous temporal state requires its actual previous RGB observation')
        result, state, _ = module(tuple(v[index:index+1] for v in features),
            batch.cue[index:index+1], tuple(v[index:index+1] for v in reference_metadata['scores']),
            valid=batch.valid[index:index+1], available=batch.available[index],
            state=detach_state(state), sequence=batch.address.sequence, frame=frame,
            flow_to_previous=flow)
        state = detach_state(state)
        previous = batch.flow_gray[index]
        outputs.append(result)
    joined = tuple(torch.cat([v[level] for v in outputs]) for level in range(3))
    predictions, _ = head_predictions(head, joined)
    return predictions, state, previous


def training_loss(head, predictions, reference, metadata, batch, config):
    if config.get('geometry_fixed', False):
        predictions = torch.cat((reference[:, :, :4], predictions[:, :, 4:]), -1)
    original = original_supervised_loss(head, predictions, metadata, batch.targets, config['training_input_hw'])
    preservation = preserve_reference(predictions, reference, metadata['strides'], reliable_only=True)
    loss = original[0] + config['fit']['lambda_reliable_RGB'] * preservation
    return loss, dict(full_YOLOX=original[0], box=original[1], objectness=original[2],
                      classification=original[3], reliable_RGB=preservation), predictions


def optimizer_step(module, encoder, head, batch, optimizer, config, position,
                   state=None, previous_gray=None):
    """One chronological clip, up to four distinct frames; no dataset padding."""
    module.train()
    with torch.no_grad():
        dtype = next(encoder.parameters()).dtype
        features = tuple(v.float() for v in encoder(batch.rgb.to(dtype=dtype)))
        reference, metadata = head_predictions(head, features)
    predictions, state, previous_gray = predict_clip(module, head, features, metadata, batch, state, previous_gray)
    initial_exact = bool(torch.equal(predictions, reference))
    loss, parts, predictions = training_loss(head, predictions, reference, metadata, batch, config)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite native-pyramid full detector loss')
    factor = learning_rate_factor(position, config['fit']['horizon_updates'],
                                  config['fit']['warmup_updates'], config['fit']['minimum_lr_fraction'])
    for group in optimizer.param_groups:
        group['lr'] = config['fit']['learning_rate'] * factor
    optimizer.zero_grad(set_to_none=True)
    # A wholly unavailable clip has no auxiliary gradient and no optimizer step.
    # It is still a delivered Train clip, never a fabricated successful update.
    updated = loss.requires_grad
    gradient, encoder_gradient = 0., 0.
    if updated:
        loss.backward()
        stem = module.encoder.blocks[0][0].weight.grad
        encoder_gradient = float(stem.abs().sum()) if stem is not None else 0.
        gradient = float(torch.nn.utils.clip_grad_norm_(module.parameters(),
                         config['fit']['gradient_clip_norm'], error_if_nonfinite=True))
        optimizer.step()
    if any(p.grad is not None or p.requires_grad for p in (*encoder.parameters(), *head.parameters())):
        raise RuntimeError('Original RGB encoder or head was updated')
    row = dict(loss=float(loss.detach()), **{k: float(v.detach()) for k, v in parts.items()},
               gradient_norm=gradient, native_stem_gradient_L1=encoder_gradient,
               optimizer_updated=updated, learning_rate=config['fit']['learning_rate'] * factor,
               initial_full_RGB_exact=initial_exact, frames=len(batch.address.frames),
               decoded_box_max_change=float((predictions[:, :, :4] - reference[:, :, :4]).detach().abs().max()))
    return row, state, previous_gray
