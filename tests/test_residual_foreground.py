from types import SimpleNamespace

import numpy as np

from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.residual_foreground_tracker import ResidualForegroundTracker
from test_foreground_tracker import crossing_data,run_points,assert_same


def test_disabled_or_uninformative_IR_preserves_RGB_point_tracker_exactly():
    data=crossing_data()
    _,reference=run_points(data,3,source_mode='rgb',target_mode='rgb',point_weight=1)
    for enabled in [False,True]:
        BaseTrack._count=0
        owner=ResidualForegroundTracker(dict(source_mode='rgb',target_mode='rgb',IR_increment=enabled,point_weight=1))
        actual=[owner.update(f,data) for f in range(1,4)]
        assert_same(reference,actual)
        assert owner.statistics['IR_increment_rows']==0
    # A genuine RGB control must also work with every auxiliary vote absent.
    for key in list(data['foreground_transport']):
        if key.startswith('vote_') and any(x in key for x in ['_ir','_joint']):
            del data['foreground_transport'][key]
    BaseTrack._count=0
    owner=ResidualForegroundTracker(dict(source_mode='rgb',target_mode='rgb',IR_increment=False,point_weight=1))
    assert_same(reference,[owner.update(f,data) for f in range(1,4)])


def ambiguity_owner(anchors):
    owner=ResidualForegroundTracker(dict(source_mode='rgb',target_mode='rgb',minimum_IR_anchors=anchors))
    context1=dict(frame=1,index=0,quality=np.ones(2))
    context2=dict(frame=2,index=1,quality=np.ones(2))
    track=SimpleNamespace(track_id=1,evidence=context2,object_observations=[context1,context2])
    observations=[SimpleNamespace(score=.95,evidence=dict(index=i,quality=np.ones(2))) for i in [2,3]]
    owner.current_start=2
    good=np.array([20]);total=good.copy()
    owner.lookup={1:(1,np.array([[.48,.52]]),good,total)}
    owner.auxiliary_lookup={1:(1,np.array([[1.,0.]]),good,total),2:(0,np.array([[1.,0.]]),good,total)}
    return owner,[track],observations


def test_IR_resolves_RGB_ambiguity_only_after_cross_time_confirmation():
    base=np.array([[.4,.3]])
    owner,tracks,observations=ambiguity_owner(2)
    revised=owner.rank_cost(tracks,observations,base,3,'primary')
    np.testing.assert_array_equal(base,[[.4,.3]])
    assert revised[0,0]<revised[0,1]
    assert owner.statistics['IR_increment_rows']==1
    owner,tracks,observations=ambiguity_owner(3)
    rejected=owner.rank_cost(tracks,observations,base,3,'primary')
    np.testing.assert_array_equal(rejected,base)
    assert owner.statistics['IR_temporally_rejected']==1


def test_disagreeing_IR_history_cannot_accumulate_false_confidence():
    owner,tracks,observations=ambiguity_owner(2)
    start,_,good,total=owner.auxiliary_lookup[2]
    owner.auxiliary_lookup[2]=(start,np.array([[0.,1.]]),good,total)
    base=np.array([[.5,.2]])
    result=owner.rank_cost(tracks,observations,base,3,'primary')
    np.testing.assert_array_equal(result,base)
    assert owner.statistics['IR_increment_rows']==0


def test_clear_motion_match_is_not_overruled_by_uncertain_point_competition():
    owner,tracks,observations=ambiguity_owner(2)
    owner.config['preserve_clear_motion']=True
    base=np.array([[.7,.2]])
    np.testing.assert_array_equal(owner.rank_cost(tracks,observations,base,3,'primary'),base)
    assert owner.statistics['motion_preserved_rows']==1
    assert owner.statistics['IR_applied_rows']==0
