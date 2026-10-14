import torch
from cfp.fish_pixel_reconstruction import patchify, pixel_reconstruction_loss


def test_patch_targets_match_row_major_image_and_rgb_channel_layout():
    image = torch.arange(3*126*252).reshape(1, 3, 126, 252).float()
    tokens = patchify(image)
    assert tokens.shape == (1, 162, 588)
    assert tokens[0, 3*18+7, (5*14+10)*3+2] == image[0, 2, 3*14+5, 7*14+10]


def test_reconstruction_loss_cannot_use_padding_or_unqueried_patch_targets():
    prediction = torch.zeros(1, 162, 588, requires_grad=True)
    target = torch.ones_like(prediction)
    weight = torch.ones_like(prediction)
    weight[:, 0, :294] = 0
    queries = torch.zeros(1, 162, dtype=torch.bool)
    queries[:, 0] = True
    observed = dict(prediction=prediction, target=target, pixel_weights=weight,
                    target_mask=queries, patch_fraction=torch.ones(1, 162))
    loss, count = pixel_reconstruction_loss(observed)
    changed = target.clone(); changed[:, 0, :294] += 100; changed[:, 1:] += 100
    other, _ = pixel_reconstruction_loss(dict(observed, target=changed))
    assert count == 1 and torch.equal(loss, other)
    loss.backward()
    assert prediction.grad[:, 0, 294:].abs().sum() > 0
    assert prediction.grad[:, 0, :294].abs().sum() == 0
    assert prediction.grad[:, 1:].abs().sum() == 0
