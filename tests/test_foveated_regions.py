import numpy as np

from yolox.data.data_augment import preproc
from yolox.tracker.foveated_regions import exact_half_preproc,working_region,coarse_attention


def test_lookup_input_is_bit_exact_to_original_float64_then_float32_then_half():
    generator=np.random.default_rng(42)
    for shape in [(31,45),(100,217),(270,480)]:
        image=generator.integers(0,256,(*shape,3),dtype=np.uint8)
        original,ratio=preproc(image,(160,288),(.485,.456,.406),(.229,.224,.225))
        actual,other=exact_half_preproc(image,(160,288),(.485,.456,.406),(.229,.224,.225))
        assert ratio==other
        np.testing.assert_array_equal(actual,original.astype(np.float16))


def test_RGB_working_region_excludes_a_bright_fixture_without_dropping_known_fish():
    rgb=np.full((180,320,3),[140,85,40],np.uint8)
    rgb[:,:35]=255;rgb[:,-35:]=255
    boxes=np.array([[2,80,20,100]],np.float32)
    support=working_region(rgb,boxes,[.9])
    assert support[90,160]==1 and support[10,10]==0
    assert support[90,10]==1
    ir=np.full((180,320),20,np.uint8);ir[:,:35]=250;ir[:,-35:]=250;ir[70:90,150:170]=100
    heat=coarse_attention(ir,boxes,[.9],rgb.shape,support=support)
    assert heat[80,160]>heat[10,10]
