import numpy as np
import torch

from cfp.dinov2_fish_data import augment_observations


def test_flip_preserves_observed_pixel_support_and_does_not_mutate_input():
    image = torch.arange(24.).reshape(1, 3, 2, 4)
    mask = torch.tensor([[[[1., 1., 0., 0.], [1., 1., 0., 0.]]]])
    original = image.clone()
    transformed, support = augment_observations(image, mask, np.random.default_rng(0),
        {'horizontal_flip_probability': 1., 'erase_probability': 0., 'erase_area_fraction': [.1, .2]})
    torch.testing.assert_close(transformed, original.flip(-1))
    torch.testing.assert_close(support, mask.flip(-1))
    torch.testing.assert_close(image, original)


def test_declared_erasure_removes_observed_support_and_pixels_together():
    image = torch.ones(2, 3, 32, 64)
    mask = torch.ones(2, 1, 32, 64)
    transformed, support = augment_observations(image, mask, np.random.default_rng(0),
        {'horizontal_flip_probability': 0., 'erase_probability': 1., 'erase_area_fraction': [.2, .2]})
    assert (support == 0).any() and (support == 1).any()
    torch.testing.assert_close(transformed[:, :1], support)
    assert torch.equal(image, torch.ones_like(image))
