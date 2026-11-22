import numpy as np

from yolox.tracker.association_calibrator import AssociationCalibrator
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.ordered_identity_memory import OrderedIdentityMemoryTracker
from test_foreground_tracker import assert_same
from test_native_query_tracker import IdentityContrast
from test_prebirth_identity import observations_at


def owner(**config):
    return OrderedIdentityMemoryTracker(dict(native_cue='ir',query_ranking=False,
        prebirth_recovery=True,identity_retention=25,counterfactual_recovery_maximum_lag=8,**config),
        model=AssociationCalibrator(),native_model=IdentityContrast(),prior=dict(t=0,a=15,b=0,c=0))


def event():
    return observations_at({1:[[20,20,50,40],[80,20,110,40]],3:[[220,20,250,40]]},3)


def test_native_memory_cannot_take_over_an_ID_already_confirmed_by_RGB_memory():
    data=event();BaseTrack._count=0;tracker=owner()
    tracker.reference_memory_scores=lambda f,c,d:(np.array([.99 if c['index']==0 else .2]),np.array([0.]))
    tracker.recovery_scores=lambda f,c,d:(np.array([.999 if c['index']==1 else .2]),np.array([1.]))
    tracker.update(1,data);tracker.update(2,data);rows=tracker.update(3,data)
    assert [row[2] for row in rows]==[1]
    assert tracker.statistics['RGB_memory_recoveries']==1
    assert tracker.statistics['native_memory_recoveries']==0


def test_native_phase_can_consult_an_identity_when_the_RGB_phase_abstains():
    data=event();BaseTrack._count=0;tracker=owner()
    tracker.reference_memory_scores=lambda f,c,d:(np.array([.2]),np.array([0.]))
    tracker.recovery_scores=lambda f,c,d:(np.array([.99 if c['index']==1 else .2]),np.array([1.]))
    tracker.update(1,data);tracker.update(2,data);rows=tracker.update(3,data)
    assert [row[2] for row in rows]==[2]
    assert tracker.statistics['RGB_memory_recoveries']==0
    assert tracker.statistics['native_memory_recoveries']==1


def test_exact_RGB_memory_outputs_need_no_auxiliary_feature_when_native_is_zero():
    data=event();data['native_reference']=np.array([[1,0],[0,1],[1,0]],np.float32)
    data['native_auxiliary']=np.array([[1,0],[0,1],[0,1]],np.float32)
    BaseTrack._count=0;full=owner(native_memory=False)
    expected=[full.update(f,data) for f in range(1,4)]
    del data['native_auxiliary'];BaseTrack._count=0;zero=owner(disable_native_increment=True)
    actual=[zero.update(f,data) for f in range(1,4)]
    assert_same(expected,actual)
    assert actual[-1][0][2]==1
    assert zero.statistics['native_memory_recoveries']==0
