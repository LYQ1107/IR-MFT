import cv2
import numpy as np

from yolox.tracker.foreground_transport import point_groups,frame_correspondences,region_masks


def translated_pair():
    rng=np.random.default_rng(2)
    first=np.full((120,160,3),[30,60,100],np.uint8)
    scalar=np.full((120,160),40,np.uint8)
    mask=np.zeros((120,160),np.uint8)
    cv2.ellipse(mask,(50,40),(24,14),10,0,360,255,-1)
    texture=rng.integers(60,220,(120,160,3),dtype=np.uint8)
    first[mask>0]=texture[mask>0];scalar[mask>0]=190
    affine=np.float32([[1,0,6],[0,1,4]])
    second=cv2.warpAffine(first,affine,(160,120),borderValue=(30,60,100))
    infrared=cv2.warpAffine(scalar,affine,(160,120),borderValue=40)
    return first,second,scalar,infrared


def test_colour_body_points_vote_for_the_translated_object():
    first,second,ir0,ir1=translated_pair()
    a=point_groups(first,ir0,np.array([[20,20,80,60]],np.float32))
    b=point_groups(second,ir1,np.array([[26,24,86,64],[95,20,150,60]],np.float32))
    result=frame_correspondences(first,second,{'new':a},{'new':b})
    votes,good=result[('new','joint','joint')]
    assert good[0]>=4
    assert votes[0,0]>.25 and votes[0,0]>votes[0,1]+.2


def test_unsupported_IR_falls_back_to_RGB_geometry():
    first,second,_,_=translated_pair()
    missing=np.full(first.shape[:2],80,np.uint8)
    masks,q=region_masks(first,missing,np.array([20,20,80,60],np.float32))
    assert q[1]==0
    np.testing.assert_array_equal(masks['ir'],masks['rgb'])
    a=point_groups(first,missing,np.array([[20,20,80,60]],np.float32))
    b=point_groups(second,missing,np.array([[26,24,86,64]],np.float32))
    result=frame_correspondences(first,second,{'new':a},{'new':b})
    np.testing.assert_allclose(result[('new','ir','ir')][0],result[('new','rgb','rgb')][0],atol=1e-6)


def test_empty_proposals_produce_finite_empty_correspondences():
    first,second,ir0,ir1=translated_pair()
    a=point_groups(first,ir0,np.empty((0,4),np.float32))
    b=point_groups(second,ir1,np.empty((0,4),np.float32))
    result=frame_correspondences(first,second,{'new':a},{'new':b})
    assert all(v.shape==(0,0) and len(g)==0 for v,g in result.values())
