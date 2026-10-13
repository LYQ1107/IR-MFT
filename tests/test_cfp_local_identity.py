import numpy as np
import pytest
import torch

from cfp.local_identity_encoder import LocalIdentityAdapter, letterbox_visibility, strip_features


def test_equal_strip_raw_mean_preserves_global_spatial_mean():
    spatial = torch.arange(2 * 3 * 4 * 8, dtype=torch.float32).reshape(2, 3, 4, 8)
    for count in (4, 8):
        torch.testing.assert_close(strip_features(spatial, count).mean(1), spatial.mean((2, 3)))
    with pytest.raises(ValueError):
        strip_features(spatial, 5)


def test_letterbox_visibility_accounts_for_real_pixels_and_clipped_boundary():
    wide = letterbox_visibility((100, 200, 3), [0, 0, 200, 100])
    assert wide.shape == (1, 128, 256) and wide.mean() == 1
    tall = letterbox_visibility((200, 100, 3), [0, 0, 100, 200])
    assert tall.sum() == 128 * 64 and (tall[:, :, :96] == 0).all()
    with pytest.raises(ValueError, match='outside'):
        letterbox_visibility((20, 20, 3), [100, 100, 2, 2])


@pytest.mark.parametrize('mode', ['global', 'dual_global', 'simple', 'learned'])
def test_adapter_stops_source_gradients_and_handles_no_visible_tokens(mode):
    torch.manual_seed(9)
    model = LocalIdentityAdapter(3, dimension=16, mode=mode)
    raw = torch.randn(2, 2048, requires_grad=True)
    local = torch.randn(2, 8, 2048, requires_grad=True)
    visible = torch.tensor([[1., 1., .5, 0., 0., 0., 0., 0.], [0.] * 8])
    output = model(raw, local, visible)
    assert torch.isfinite(output['embedding']).all() and (output['weights'][:, 3:] == 0).all()
    assert (output['weights'][1] == 0).all()
    output['logits'].square().mean().backward()
    assert raw.grad is None and local.grad is None
    assert model.global_projection.weight.grad is not None


@pytest.mark.parametrize('mode', ['global', 'dual_global'])
def test_global_control_cannot_read_local_identity_or_visibility(mode):
    model = LocalIdentityAdapter(3, dimension=16, mode=mode).eval()
    raw, local, visibility = torch.randn(2, 2048), torch.randn(2, 8, 2048), torch.ones(2, 8)
    torch.testing.assert_close(model(raw, local, visibility)['embedding'], model(raw, -local, visibility * 0)['embedding'], rtol=0, atol=0)


def test_trainable_visual_route_keeps_real_spatial_gradient():
    from cfp.trainable_local_identity import TrainableLocalFishReID
    model = TrainableLocalFishReID(3, dimension=16, mode='learned')
    spatial = torch.randn(2, 2048, 4, 8, requires_grad=True)
    output = model.forward_spatial(spatial, torch.ones(2, 8))
    output['logits'].square().mean().backward()
    assert spatial.grad is not None and torch.isfinite(spatial.grad).all() and spatial.grad.abs().sum() > 0
