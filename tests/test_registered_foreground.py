import cv2
import numpy as np

from yolox.tracker.foreground_transport import region_masks
from yolox.tracker.registered_foreground_transport import registered_masks


POLICY=dict(minimum_response=.15,minimum_correlation_gain=.02,maximum_box_shift=.2,minimum_box_shift=.01)


def test_registration_corrects_known_IR_displacement_and_preserves_RGB_mask():
    rng=np.random.default_rng(2);gray=np.full((100,160),20,np.uint8)
    texture=rng.integers(30,230,(26,60),dtype=np.uint8);gray[35:61,50:110]=texture
    rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    ir=cv2.warpAffine(gray,np.float32([[1,0,5],[0,1,-2]]),(160,100))
    box=np.array([40,25,120,75],np.float32)
    raw,q=region_masks(rgb,ir,box);registered,rq,diagnostic=registered_masks(rgb,ir,box,POLICY)
    np.testing.assert_array_equal(raw['rgb'],registered['rgb'])
    assert q[0]==rq[0] and diagnostic[4]==1
    np.testing.assert_allclose(diagnostic[:2],[5/80,-2/50],atol=.02)
    target,_=region_masks(rgb,gray,box)
    assert np.mean((registered['ir']-target['ir'])**2)<np.mean((raw['ir']-target['ir'])**2)


def test_flat_or_rejected_registration_leaves_original_geometry_exact():
    rgb=np.full((80,120,3),60,np.uint8);ir=np.full((80,120),90,np.uint8);box=np.array([20,20,70,50],np.float32)
    first,q=region_masks(rgb,ir,box);second,rq,diagnostic=registered_masks(rgb,ir,box,POLICY)
    for mode in first:np.testing.assert_array_equal(first[mode],second[mode])
    np.testing.assert_array_equal(q,rq);assert not diagnostic[4]
