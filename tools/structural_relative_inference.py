"""Keep original RGB decoding exact when the relative semantic influence is zero."""
import torch

from yolox.models.structural_relative_repair import relative_semantics


def decoded_relative_response(current, null, teacher_raw, teacher_decoded, centered=True, bound=2.):
    response = relative_semantics(current.float(), null.float(), teacher_raw.float(), centered, bound)
    delta = response[:, :, 4:]-teacher_raw[:, :, 4:].float()
    output = teacher_decoded.clone()
    mapped = response[:, :, 4:].sigmoid().to(output.dtype)
    # The original head applies sigmoid before concatenation/decoding. Keep
    # that exact value for zero influence rather than round-trip it via logits.
    output[:, :, 4:] = torch.where(delta == 0, teacher_decoded[:, :, 4:], mapped)
    return output
