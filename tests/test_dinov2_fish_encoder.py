import torch
from torch.nn import functional as F

from cfp.dinov2_fish_encoder import PatchIdentityHead


def observations():
    return torch.randn(2, 16), torch.randn(2, 162, 16), torch.ones(2, 1, 9, 18)


def test_equal_capacity_control_cannot_read_patch_image_content():
    torch.manual_seed(3)
    control = PatchIdentityHead(3, 'global_capacity', dimension=16, heads=4).eval()
    spatial = PatchIdentityHead(3, 'patch_attention', dimension=16, heads=4).eval()
    spatial.load_state_dict(control.state_dict())
    assert sum(p.numel() for p in control.parameters() if p.requires_grad) == sum(p.numel() for p in spatial.parameters() if p.requires_grad)
    cls, patches, visible = observations()
    changed = patches+torch.randn_like(patches)*5
    torch.testing.assert_close(control(cls, patches, visible)['embedding'], control(cls, changed, visible)['embedding'])
    assert not torch.allclose(spatial(cls, patches, visible)['embedding'], spatial(cls, changed, visible)['embedding'])


def test_absent_regions_and_whole_crops_are_finite_and_invalid():
    cls, patches, visible = observations()
    visible[0, :, :, 9:] = 0
    visible[1] = 0
    for mode in ('global', 'simple', 'global_capacity', 'patch_attention'):
        output = PatchIdentityHead(3, mode, dimension=16, heads=4)(cls, patches, visible)
        assert all(torch.isfinite(x).all() for x in output.values())
        assert not output['crop_valid'][1]
        assert output['embedding'][1].count_nonzero() == 0
        assert output['local_tokens'][1].count_nonzero() == 0
        assert not output['local_valid'][0, 2:].any()


def test_identity_gradient_reaches_local_queries_and_observed_patches():
    cls, patches, visible = observations()
    patches.requires_grad_()
    head = PatchIdentityHead(3, 'patch_attention', dimension=16, heads=4)
    output = head(cls, patches, visible)
    F.cross_entropy(output['logits'], torch.tensor([0, 1])).backward()
    assert patches.grad.norm() > 0
    assert head.queries.grad.norm() > 0
    assert head.attention.in_proj_weight.grad.norm() > 0


def test_current_embedding_does_not_read_other_batch_images():
    cls, patches, visible = observations()
    head = PatchIdentityHead(3, 'patch_attention', dimension=16, heads=4).eval()
    single = head(cls[:1], patches[:1], visible[:1])['embedding']
    together = head(cls, patches, visible)['embedding'][:1]
    torch.testing.assert_close(single, together, atol=1e-6, rtol=1e-6)
