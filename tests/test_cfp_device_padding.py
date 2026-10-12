import numpy as np
import torch

from cfp.detector_runtime_device_pad import current_device_padding
from cfp.online import proposal_crop_arrays


def test_padding_conversion_is_same_contiguous_tensor_without_repeated_host_transfer():
    source = np.linspace(-3, 3, 6 * 17 * 19, dtype=np.float32).reshape(6, 17, 19)
    expected = torch.from_numpy(source)[None].repeat(8, 1, 1, 1).half()
    actual = current_device_padding(source, 8, device='cpu')
    assert actual.is_contiguous() and torch.equal(actual, expected)


def test_skip_unused_ir_does_not_change_any_rgb_crop_or_validity():
    image = np.random.default_rng(11).integers(0, 256, (45, 67, 3), dtype=np.uint8)
    ir = image[::-1].copy()
    boxes = np.array([[0, 0, 30, 20], [25, 10, 60, 40], [-5, -5, -1, -1]], float)
    original, valid = proposal_crop_arrays(image, ir, boxes)
    fast, fast_valid = proposal_crop_arrays(image, ir, boxes, rgb_only=True)
    assert torch.equal(original[0], fast[0]) and np.array_equal(valid[0], fast_valid[0])
    assert len(fast[1]) == len(fast_valid[1]) == 0

