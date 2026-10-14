import numpy as np
import pytest
from cfp.fish_geometry import geometry_similarity,KINDS


def test_every_control_preserves_empty_rows_columns_finite_bounded_values():
    boxes=np.array([[0,0,10,20],[4,8,20,21],[7,5,7,9]],float)
    for mode in KINDS[1:]:
        a=geometry_similarity(boxes,boxes,mode);assert a.shape==(3,3) and np.isfinite(a).all() and (a>=0).all() and (a<=1).all()
        assert geometry_similarity(np.empty((0,4)),boxes,mode).shape==(0,3)
        assert geometry_similarity(boxes,np.empty((0,4)),mode).shape==(3,0)
        assert np.array_equal(a[2],np.zeros(3))


def test_source_central_defect_is_not_silently_erased():
    a=np.array([[0,0,100,40]],float);b=np.array([[30,0,130,40]],float)
    original=geometry_similarity(a,b,'fish_source');repaired=geometry_similarity(a,b,'fish_central_repaired')
    assert original[0,0]>repaired[0,0]
    assert abs(original[0,0]-repaired[0,0])==pytest.approx(.05,abs=1e-8)


def test_iou_coordinate_convention_and_fixed_offset_controls_are_explicit():
    a=np.array([[0,0,2,2]],float);b=np.array([[2,0,4,2]],float)
    assert geometry_similarity(a,b,'continuous_iou')[0,0]==0
    assert geometry_similarity(a,b,'iou_offset005')[0,0]==.05
    assert geometry_similarity(a,b,'fish_affine_control')[0,0]==.3


def test_source_geometry_translation_and_diou_scale_invariance():
    a=np.array([[0,0,10,4],[5,7,15,18]],float);b=np.array([[3,1,14,8],[6,9,16,20]],float)
    for mode in KINDS[1:]:np.testing.assert_allclose(geometry_similarity(a+123,b+123,mode),geometry_similarity(a,b,mode),atol=1e-14)
    np.testing.assert_allclose(geometry_similarity(a*5,b*5,'diou'),geometry_similarity(a,b,'diou'),atol=1e-14)
    assert not np.allclose(geometry_similarity(a*5,b*5,'agc_clipped'),geometry_similarity(a,b,'agc_clipped'))


def test_agc_zero_distance_has_disclosed_finite_clipped_limit():
    a=np.array([[0,0,10,10]],float);b=np.array([[-1,-1,11,11]],float)
    assert geometry_similarity(a,b,'agc_clipped')[0,0]==1
    with pytest.raises(ValueError):geometry_similarity(a*np.nan,b,'diou')


def test_geometry_adapter_preserves_identity_cost_and_original_appearance_graph():
    from types import SimpleNamespace
    from collections import deque
    from yolox.tracker import matching
    from cfp.fish_geometry import install_geometry_control
    rgb=np.ones(4,dtype=np.float32)/2
    tracks=[SimpleNamespace(tlbr=np.array([0.,0.,20.,10.]),identity_history=deque([(1,rgb,None)]),initial_identity_rgb=rgb,initial_identity_frame=1,initial_identity_ir=None),SimpleNamespace(tlbr=np.array([25.,0.,45.,10.]),identity_history=deque(),initial_identity_rgb=None,initial_identity_frame=0,initial_identity_ir=None)]
    detections=[SimpleNamespace(tlbr=np.array([2.,0.,22.,10.]),score=.9,current_rgb=rgb),SimpleNamespace(tlbr=np.array([26.,0.,46.,10.]),score=.8,current_rgb=None)]
    raw=matching.iou_distance(tracks,detections);old=matching.fuse_score(raw,detections);eligible=(raw[0]<=.9)&np.array([True,False]);factor=np.ones_like(old);factor[0,eligible]=.7
    baseline=old.copy();baseline[0,eligible]=.7*old[0,eligible]+.3*.2
    class Base:
        frame_id=0;memory_mode='predictor';history_view='anchor_recent';appearance_weight=.3;proximity_iou=.1;args=SimpleNamespace(mot20=False,match_thresh=.97)
        def association_cost(self,tracks,detections):return baseline.copy()
    t=install_geometry_control(Base(),'diou');t.frame_id=2
    result=t.association_cost(tracks,detections);new=matching.fuse_score(1-geometry_similarity([x.tlbr for x in tracks],[x.tlbr for x in detections],'diou'),detections)
    np.testing.assert_allclose(result,baseline+factor*(new-old),atol=1e-14)
    assert t.geometry_counts['edges']==4
    assert len(tracks[0].identity_history)==1 and not tracks[1].identity_history
