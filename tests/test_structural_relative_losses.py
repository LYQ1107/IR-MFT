import torch

from yolox.models.structural_relative_losses import preserve_semantics


def test_semantic_preservation_ignores_geometry_and_masks_shared_foreground():
    teacher = torch.zeros(1, 3, 6)
    prediction = teacher.clone(); prediction[:, :, :4] = 100
    assert preserve_semantics(prediction, teacher) == 0
    prediction[:, 0, 4:] = 2
    prediction.requires_grad_()
    validity = torch.tensor([[False, True, True]])
    masked = preserve_semantics(prediction, teacher, validity)
    assert masked == 0
    assert preserve_semantics(prediction, teacher) > 0
    masked.backward()
    assert prediction.grad.count_nonzero() == 0


def test_semantic_preservation_does_not_amplify_remaining_locations():
    teacher = torch.ones(1, 4, 6)
    prediction = teacher.clone(); prediction[:, :, 4:] -= 1
    full = preserve_semantics(prediction, teacher)
    half = preserve_semantics(prediction, teacher, torch.tensor([[True, False, True, False]]))
    torch.testing.assert_close(half, full/2)
