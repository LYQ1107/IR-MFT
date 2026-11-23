import torch

from tools.structural_relative_inference import decoded_relative_response


def test_zero_influence_preserves_original_half_decoding_and_all_dense_boxes():
    raw = torch.tensor([[[8., 12., 10., 20., .1234, 7.], [2., 5., 3., 4., -.76, 2.]]], dtype=torch.float16)
    original = torch.cat([raw[:, :, :4], raw[:, :, 4:].sigmoid()], -1)
    original[:, :, 4] = torch.nextafter(original[:, :, 4], torch.ones_like(original[:, :, 4]))
    current = raw+1
    identity = decoded_relative_response(current, current, raw, original)
    torch.testing.assert_close(identity, original, rtol=0, atol=0)
    changed = decoded_relative_response(current, raw, raw, original)
    torch.testing.assert_close(changed[:, :, :4], original[:, :, :4], rtol=0, atol=0)
    assert not torch.equal(changed[:, :, 4:], original[:, :, 4:])
