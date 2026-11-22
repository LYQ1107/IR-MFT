import numpy as np

from yolox.tracker.association_calibrator import AssociationCalibrator
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.calibrated_foreground_tracker import CalibratedForegroundTracker
from yolox.tracker.counterfactual_query_tracker import CounterfactualQueryTracker
from test_foreground_tracker import assert_same
from test_native_query_tracker import example, IdentityContrast


def owner(**config):
    return CounterfactualQueryTracker(dict(native_cue='ir', **config), model=AssociationCalibrator(),
                                     native_model=IdentityContrast(),prior=dict(t=0,a=0,b=0,c=0))


def test_zero_cue_exactly_preserves_the_point_matcher_even_with_retained_ID_memory():
    data = example()
    del data['native_reference'],data['native_auxiliary']
    BaseTrack._count=0
    parent=CalibratedForegroundTracker(dict(preserve_distinct_objects=True),model=AssociationCalibrator(),cue='rgb')
    expected=[parent.update(f,data) for f in range(1,4)]
    BaseTrack._count=0
    tracker=owner(disable_native_increment=True,prebirth_recovery=True,identity_retention=128,preserve_distinct_objects=True)
    assert_same(expected,[tracker.update(f,data) for f in range(1,4)])
    assert tracker.statistics['counterfactual_rows']==tracker.statistics['prebirth_recoveries']==0


def test_undoing_only_native_increment_retains_RGB_real_and_null_probabilities():
    tracker=owner();tracker.data=example()
    packet=tracker.query_posterior(2,dict(frame=1,index=0))
    real,null=tracker.counterfactual_probabilities(packet)
    np.testing.assert_allclose(real,np.full((2,2),1/3),rtol=1e-7,atol=1e-8)
    np.testing.assert_allclose(null,np.full(2,1/3),rtol=1e-7,atol=1e-8)


def test_native_cue_must_improve_the_same_RGB_query_before_opening_a_crossing():
    data=example()
    for key,value in data['foreground_transport'].items():
        if key.startswith('vote_') and 'offsets' not in key:value[:]=0
    BaseTrack._count=0
    tracker=owner(association_weight=0,preserve_distinct_objects=True)
    first=tracker.update(1,data);second=tracker.update(2,data)
    centers={row[2]:row[1][0]+row[1][2]/2 for row in second}
    assert centers[1]>centers[2]
    assert tracker.statistics['counterfactual_rows']>=2


def test_clear_RGB_body_transport_is_preserved_when_auxiliary_identity_is_mismatched():
    data=example();data['native_auxiliary'][2:4]=data['native_auxiliary'][2:4][::-1]
    BaseTrack._count=0
    tracker=owner(preserve_distinct_objects=True)
    tracker.update(1,data);second=tracker.update(2,data)
    centers={row[2]:row[1][0]+row[1][2]/2 for row in second}
    assert centers[1]>centers[2]
    assert tracker.statistics['counterfactual_clear_preserved']>=2
