import torch

from cfp.raw_identity_descriptor import raw_rgb_descriptor


def test_raw_descriptor_is_finite_unit_norm_and_padding_invariant():
    torch.manual_seed(7)
    crops = torch.randn(2, 3, 128, 256)
    masks = torch.zeros(2, 1, 128, 256)
    masks[:, :, 12:116, 24:232] = 1
    out = raw_rgb_descriptor(crops, masks)
    assert out.shape == (2, 384)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.norm(dim=1), torch.ones(2), atol=1e-5)


def test_raw_descriptor_invalid_crop_is_zero_and_no_label_input_exists():
    crops = torch.ones(1, 3, 128, 256)
    masks = torch.zeros(1, 1, 128, 256)
    out = raw_rgb_descriptor(crops, masks)
    assert torch.equal(out, torch.zeros_like(out))
