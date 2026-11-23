import numpy as np

from yolox.tracker.priority_observations import priority_union


def test_high_confidence_global_boxes_and_scores_are_immutable():
    base=np.array([[0,0,20,20,.8],[3,3,23,23,.9]],np.float32)
    crop=np.array([[1,1,21,21,.99],[4,4,24,24,.99]],np.float32)
    result,index=priority_union(base,[(crop,4)])
    np.testing.assert_array_equal(result,base)
    np.testing.assert_array_equal(index,[0,1])


def test_weak_confirmation_preserves_global_box_and_new_object_is_appended():
    base=np.array([[0,0,20,20,.2]],np.float32)
    crop=np.array([[1,1,21,21,.9],[50,0,70,20,.8]],np.float32)
    result,index=priority_union(base,[(crop,4)])
    np.testing.assert_array_equal(result[0,:4],base[0,:4])
    assert result[0,4]==crop[0,4]
    np.testing.assert_array_equal(result[1],crop[1])
    np.testing.assert_array_equal(index,[0,-1])


def test_partial_duplicate_does_not_displace_distinct_original_observations():
    base=np.array([[0,0,20,20,.8],[15,0,35,20,.85]],np.float32)
    crop=np.array([[5,5,15,15,.99],[0,0,35,20,.99]],np.float32)
    result,index=priority_union(base,[(crop,4)])
    np.testing.assert_array_equal(result,base)
    np.testing.assert_array_equal(index,[0,1])
