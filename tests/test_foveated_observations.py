import numpy as np

from yolox.tracker.foveated_observations import attention_windows,coarse_attention,integrate_crops


def test_zero_crop_preserves_every_original_coordinate_and_confidence_exactly():
    base=np.array([[-2,5,20,30,.85],[30,20,70,60,.4]],np.float32)
    np.testing.assert_array_equal(integrate_crops(base,[]),base)


def test_attention_uses_only_current_cue_and_selects_a_salient_half_image():
    heat=np.zeros((18,32),np.float32);heat[10:16,22:29]=1
    window=attention_windows(heat,(1080,1920),count=1)[0]
    assert window[0]<=1320<window[2] and window[1]<=720<window[3]
    assert window[2]-window[0]==960 and window[3]-window[1]==540


def test_cross_view_resolution_refinement_keeps_two_distinct_same_view_objects():
    base=np.array([[0,0,20,20,.9],[30,0,50,20,.8]],np.float32)
    crop=np.array([[2,0,22,20,.9],[32,0,52,20,.8]],np.float32)
    result=integrate_crops(base,[(crop,2.)])
    assert len(result)==2
    np.testing.assert_allclose(result[:,:4],[[1.6,0,21.6,20],[31.6,0,51.6,20]],atol=1e-5)
    np.testing.assert_array_equal(result[:,4],base[:,4])


def test_scalar_new_IR_and_RGB_attention_share_geometry_without_changing_pixels():
    rgb=np.zeros((108,192,3),np.uint8);rgb[20:40,80:110]=200
    ir=rgb[:,:,0].copy();saved=ir.copy()
    a=coarse_attention(rgb,np.empty((0,4)),[],rgb.shape)
    b=coarse_attention(ir,np.empty((0,4)),[],rgb.shape)
    assert a.shape==b.shape==(180,320)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    np.testing.assert_array_equal(ir,saved)
