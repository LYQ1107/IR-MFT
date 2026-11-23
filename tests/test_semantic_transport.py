import cv2
import numpy as np

from yolox.tracker.foreground_transport import point_groups,frame_correspondences
from yolox.tracker.semantic_transport import semantic_groups,semantic_votes


def test_IR_contour_recovers_motion_when_RGB_is_textureless():
    rgb=np.full((100,160,3),100,np.uint8)
    ir=np.zeros((100,160),np.uint8)
    cv2.rectangle(ir,(45,35),(70,55),200,-1)
    shifted=cv2.warpAffine(ir,np.array([[1,0,3],[0,1,0]],np.float32),(160,100))
    boxes=np.array([[35,25,80,65]],np.float32)
    past=semantic_groups(rgb,ir,boxes)
    target=semantic_groups(rgb,shifted,boxes+np.array([3,0,3,0]))
    assert len(past['semantic_joint']['points'])>=4
    votes=semantic_votes(rgb,rgb,ir,shifted,past,target)
    assert votes['semantic_rgb']['matrix'][0,0]==0
    assert votes['semantic_safe']['matrix'][0,0]>.1
    assert votes['semantic_safe']['IR_fallback_points']>0


def test_flat_IR_preserves_original_point_packets_exactly():
    rng=np.random.default_rng(4)
    rgb=rng.integers(0,256,(80,120,3),dtype=np.uint8)
    ir=np.zeros((80,120),np.uint8)
    boxes=np.array([[20,15,80,65]],np.float32)
    original=point_groups(rgb,ir,boxes)
    added=semantic_groups(rgb,ir,boxes)
    for mode in original['groups']:
        for field in original['groups'][mode]:
            np.testing.assert_array_equal(original['groups'][mode][field],added['groups'][mode][field])
    reference=frame_correspondences(rgb,rgb,{'x':original},{'x':original})[('x','joint','rgb')][0]
    output=semantic_votes(rgb,rgb,ir,ir,added,added)
    for variant in output:
        np.testing.assert_array_equal(output[variant]['matrix'],reference)
