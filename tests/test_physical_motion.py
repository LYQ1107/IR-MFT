import cv2
import numpy as np

from yolox.tracker.physical_motion import padded_motion_view,backward_rgb_motion


def test_flow_canvas_preserves_detector_padding_instead_of_stretching_image():
    image=np.zeros((8,12,3),np.uint8);view=padded_motion_view(image,(16,28),4)
    assert view.shape==(4,7)
    np.testing.assert_array_equal(view[:,:6],0)
    np.testing.assert_array_equal(view[:,-1],114)


def test_current_to_previous_flow_has_correct_translation_sign_and_fb_support():
    rng=np.random.default_rng(41);previous=rng.integers(0,256,(96,144),dtype=np.uint8)
    previous=cv2.GaussianBlur(previous,(3,3),.5)
    current=cv2.warpAffine(previous,np.float32([[1,0,2],[0,1,1]]),(144,96),borderValue=114)
    backward,valid=backward_rgb_motion(previous,current)
    central=np.s_[20:-20,20:-20]
    np.testing.assert_allclose(np.median(backward[central].reshape(-1,2),axis=0),[-2,-1],atol=.25)
    assert valid[central].mean()>.9
