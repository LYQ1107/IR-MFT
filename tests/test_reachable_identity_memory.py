from types import SimpleNamespace

import numpy as np

from yolox.tracker.association_calibrator import AssociationCalibrator
from yolox.tracker.reachable_identity_memory import ReachableIdentityMemoryTracker
from test_native_query_tracker import IdentityContrast
from test_prebirth_identity import observations_at


def owner():
    return ReachableIdentityMemoryTracker(dict(native_cue='ir'),model=AssociationCalibrator(),
        native_model=IdentityContrast(),prior=dict(t=0,a=0,b=0,c=0),
        motion_statistics=dict(speed_floor=.02,position_slack_body_diagonals=.5,acceleration_upper=.02))


def test_reachability_uses_actual_past_positions_and_is_independent_of_future_boxes():
    data=observations_at({1:[[20,20,50,40]],2:[[23,20,53,40]],
        3:[[26,20,56,40],[300,20,330,40]],4:[[30,20,60,40]]},4)
    tracker=owner();history=[dict(frame=1,index=0),dict(frame=2,index=1)]
    tracker.identity_archive={1:dict(latest=history[-1],track=SimpleNamespace(object_observations=history))}
    first=tracker.reachable(3,history[-1],data)
    np.testing.assert_array_equal(first,[True,False])
    data['boxes'][4:]=1e6;tracker.reachability_cache={}
    np.testing.assert_array_equal(tracker.reachable(3,history[-1],data),first)


def test_single_observation_or_a_future_source_does_not_supply_physical_identity_proof():
    data=observations_at({1:[[20,20,50,40]],2:[[23,20,53,40]]},2)
    tracker=owner();context=dict(frame=1,index=0)
    tracker.identity_archive={1:dict(latest=context,track=SimpleNamespace(object_observations=[context]))}
    assert not tracker.reachable(2,context,data).any()
    try:tracker.reachable(2,dict(frame=2,index=1),data)
    except ValueError:pass
    else:raise AssertionError('A current/future source entered the reachability bound')
