import numpy as np
import pytest
import torch

from cfp.crop_views import long_axis_crop
from cfp.identity_data import crop_to_tensor


def test_horizontal_view_is_exact_original_and_does_not_mutate_source():
    image = np.random.default_rng(0).integers(0, 256, (50, 100, 3), dtype=np.uint8)
    before = image.copy()
    box = [5, 8, 70, 25]
    assert torch.equal(long_axis_crop(image, box), crop_to_tensor(image, box))
    assert np.array_equal(image, before)


def test_vertical_rotation_preserves_color_pair_alignment_and_increases_active_area():
    rgb = np.full((80, 20, 3), 255, np.uint8)
    rgb[:40, :, :2] = 0  # red top, white bottom in BGR.
    paired = rgb.copy()
    original = crop_to_tensor(rgb, [0, 0, 20, 80], margin=0)
    aligned = long_axis_crop(rgb, [0, 0, 20, 80], margin=0)
    assert torch.equal(aligned, long_axis_crop(paired, [0, 0, 20, 80], margin=0))
    assert (aligned.abs().sum(0) > 0).sum() > (original.abs().sum(0) > 0).sum()
    assert aligned[0, :, 192:].max() > 2 and aligned[2, :, 192:].max() <= 0
    with pytest.raises(ValueError):
        long_axis_crop(rgb, [100, 100, 2, 2])
