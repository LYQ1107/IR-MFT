"""Semantic-only preservation; shared foreground never validates regression."""
import torch
from torch.nn import functional as F


def preserve_semantics(predictions, reference, validity=None, reliable_only=False):
    if predictions.shape != reference.shape or predictions.shape[-1] != 6:
        raise ValueError('Corresponding one-class YOLOX logits required')
    reference = reference.detach()
    p = reference[:, :, 4:].sigmoid()
    foreground = p[:, :, 0]*p[:, :, 1]
    reliable = (foreground >= .6).to(foreground) if reliable_only else torch.ones_like(foreground)
    weight = reliable
    if validity is not None:
        if validity.shape != predictions.shape[:2]:
            raise ValueError('One semantic-validity flag per original anchor')
        weight = weight*validity.to(weight)
    kl = p*(F.logsigmoid(reference[:, :, 4:])-F.logsigmoid(predictions[:, :, 4:]))
    kl += (1-p)*(F.logsigmoid(-reference[:, :, 4:])-F.logsigmoid(-predictions[:, :, 4:]))
    # Keep the original reference mass after excluding shared foreground; do not
    # amplify background gradients when few valid locations remain.
    denominator = (foreground*reliable).sum().clamp_min(1.)
    return (((kl[:, :, 0]+foreground*kl[:, :, 1])*weight).sum()/denominator).clamp_min(0)
