import numpy as np

from yolox.tracker.association_calibrator import AssociationCalibrator
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.missing_point_tracker import MissingPointEvidenceTracker
from yolox.tracker.strong_rgb_tracker import StrongRGBForegroundTracker
from test_calibrated_tracker import data_with_descriptors
from test_foreground_tracker import with_transport,assert_same


def missing_data():
    data=with_transport([[10,20,40,40],[60,20,90,40],[110,20,140,40],[160,20,190,40]], [.95]*4,[0,2,4],zero=True)
    for key in ['embedding','rgb_mask_embedding','ir_mask_embedding','rgb_hog','ir_hog','color']:
        data[key]=np.tile(np.eye(2,dtype=np.float32),(2,1))
    data['quality']=np.ones((4,2),np.float32)
    return data


def owner(**config):
    return MissingPointEvidenceTracker(config,model=AssociationCalibrator(base_dimensions=26),frozen_RGB=AssociationCalibrator(),cue='ir')


def test_strict_current_appearance_can_open_a_missing_point_link_without_changing_past_output():
    data=missing_data()
    for opening in [False,True]:
        BaseTrack._count=0;tracker=owner(semantic_missing_points=True,semantic_opening=opening)
        tracker.appearance_scores=lambda frame,context,data:(np.array([.995,.001]) if context['index']==0 else np.array([.001,.995]),np.zeros(2))
        first=tracker.update(1,data);saved=first[0][1].copy()
        result=tracker.update(2,data)
        assert [row[2] for row in result]==([1,2] if opening else [])
        np.testing.assert_array_equal(first[0][1],saved)
        assert bool(tracker.statistics['semantic_opened_edges'])==opening


def test_competing_old_identities_abstain_instead_of_claiming_the_same_current_object():
    data=missing_data();data['embedding'][:]=np.array([1,0],np.float32)
    BaseTrack._count=0;tracker=owner(semantic_missing_points=True,semantic_opening=True)
    tracker.appearance_scores=lambda frame,context,data:(np.array([.995,.001]),np.zeros(2))
    tracker.update(1,data);assert tracker.update(2,data)==[]
    assert tracker.statistics['semantic_rows']==0


def test_disabling_missing_point_repair_preserves_the_entire_parent_lifecycle():
    data=data_with_descriptors();RGB=AssociationCalibrator();residual=AssociationCalibrator(base_dimensions=26)
    BaseTrack._count=0;parent=StrongRGBForegroundTracker({},model=residual,frozen_RGB=RGB,cue='ir')
    expected=[parent.update(frame,data) for frame in range(1,4)]
    BaseTrack._count=0;tracker=owner(semantic_missing_points=False)
    assert_same(expected,[tracker.update(frame,data) for frame in range(1,4)])
