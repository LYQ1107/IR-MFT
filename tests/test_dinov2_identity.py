import pytest
import torch

from cfp.dinov2_identity import resized_observations, spatial_observations


def test_padding_cannot_change_pooled_identity_observations():
    tokens = torch.randn(2, 162, 6)
    visibility = torch.ones(2, 1, 9, 18)
    visibility[:, :, :, 9:] = 0
    baseline = spatial_observations(tokens, visibility)
    changed = tokens.reshape(2, 9, 18, 6).clone()
    changed[:, :, 9:] = 10000
    changed = spatial_observations(changed.flatten(1, 2), visibility)
    for name in baseline:
        torch.testing.assert_close(changed[name], baseline[name])
    assert baseline['local8'].shape == (2, 8, 6)


def test_missing_content_is_finite_zero_and_resize_preserves_fraction():
    images = torch.ones(2, 3, 128, 256)
    masks = torch.ones(2, 1, 128, 256)
    masks[1] = 0
    resized, visible = resized_observations(images, masks)
    assert resized.shape == (2, 3, 126, 252)
    torch.testing.assert_close(visible[0], torch.ones(1, 9, 18))
    observations = spatial_observations(torch.ones(2, 162, 6), visible)
    for value in observations.values():
        assert torch.isfinite(value).all()
        assert torch.count_nonzero(value[1]) == 0


def test_reject_misaligned_tokens_and_nonpatch_resize():
    with pytest.raises(ValueError):
        spatial_observations(torch.ones(2, 163, 6), torch.ones(2, 1, 9, 18))
    with pytest.raises(ValueError):
        resized_observations(torch.ones(1, 3, 128, 256), torch.ones(1, 1, 128, 256), (128, 256))
