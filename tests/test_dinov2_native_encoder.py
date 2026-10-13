import numpy as np
import torch

from cfp.dinov2_native_encoder import current_proposal_crops
from cfp.identity_data import crop_to_tensor
from cfp.local_identity_encoder import letterbox_visibility


def test_actual_current_proposal_pixels_and_masks_match_registered_crop_path():
    image = np.random.default_rng(8).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    boxes = np.array([[2, 3, 24, 21], [-3, 0, 12, 30], [60, 60, 70, 70]], np.float32)
    original = image.copy()
    crops, masks, valid = current_proposal_crops(image, boxes)
    assert valid.tolist()==[True, True, False]
    for i in range(2):
        x1,y1,x2,y2=boxes[i];xywh=[x1,y1,x2-x1,y2-y1]
        torch.testing.assert_close(crops[i], crop_to_tensor(image, xywh).half().float(), rtol=0, atol=0)
        torch.testing.assert_close(masks[i], letterbox_visibility(image.shape[:2], xywh), rtol=0, atol=0)
    assert torch.count_nonzero(crops[2])==torch.count_nonzero(masks[2])==0
    assert np.array_equal(image, original)


def test_zero_proposals_preserve_empty_current_frame():
    crops, masks, valid = current_proposal_crops(np.zeros((16, 16, 3), np.uint8), np.empty((0, 4), np.float32))
    assert crops.shape==(0, 3, 128, 256) and masks.shape==(0, 1, 128, 256) and valid.shape==(0,)
