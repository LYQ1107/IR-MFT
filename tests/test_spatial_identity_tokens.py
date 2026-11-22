import numpy as np
import torch

from yolox.tracker.spatial_identity_tokens import spatial_weights,mask_axis,pool_identity_tokens


def test_RGB_spatial_controls_do_not_consume_IR_and_preserve_letterbox_coordinates():
    image=np.zeros((120,40,3),np.uint8);image[20:100,10:30]=[40,150,220]
    first=spatial_weights(image)
    second=spatial_weights(image,np.random.default_rng(3).integers(0,256,(120,40),dtype=np.uint8))
    for mode in ['box','rgb']:
        np.testing.assert_array_equal(first[mode]['weights'],second[mode]['weights'])
        np.testing.assert_array_equal(first[mode]['quality'],second[mode]['quality'])
        assert np.all(first[mode]['weights'][1:,:,0]==0)
        assert np.all(first[mode]['weights'][1:,:,-1]==0)
    assert np.all(first['box']['weights'][0]==1)


def test_principal_axis_uses_physical_crop_aspect_and_not_resized_mask_aspect():
    center,axis,coherence,scale=mask_axis(np.ones((32,64)),(120,40,3))
    np.testing.assert_allclose(center,[20,60],atol=1e-6)
    assert abs(axis[1])>.99 and coherence>.7 and scale>1


def test_pooling_preserves_constant_RGB_texture_without_pixel_masking():
    torch.manual_seed(2);projection=torch.nn.Linear(8,5,bias=False)
    feature=torch.randn(2,8,1,1).expand(-1,-1,4,8)
    weights=torch.rand(2,3,4,8)
    tokens=pool_identity_tokens(feature,weights,projection)
    expected=torch.nn.functional.normalize(projection(feature[:,:,0,0]),dim=-1)
    torch.testing.assert_close(tokens,expected[:,None].expand(-1,3,-1),rtol=1e-5,atol=1e-6)
