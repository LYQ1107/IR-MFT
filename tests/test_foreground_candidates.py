import numpy as np

from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.candidate_foreground_tracker import CandidateForegroundTracker
from test_foreground_tracker import with_transport


def low_score_data(zero=False):
    return with_transport([[20,20,50,40]]*4,[.65]*4,[0,1,2,3,4],zero=zero)


def run(data,**config):
    BaseTrack._count=0
    owner=CandidateForegroundTracker(config)
    return owner,[owner.update(f,data) for f in range(1,5)]


def test_repeated_low_score_body_becomes_an_unconfirmed_then_confirmed_identity():
    plain,original=run(low_score_data(),candidate_birth=False)
    supported,output=run(low_score_data(),candidate_birth=True)
    assert all(not rows for rows in original)
    assert not output[0] and not output[1]
    assert [r[2] for r in output[2]]==[1]
    assert [r[2] for r in output[3]]==[1]
    assert supported.statistics['candidate_births']==1
    assert supported.statistics['IR_supported_births']==1


def test_unsupported_candidate_is_not_promoted_by_simply_reducing_a_threshold():
    owner,output=run(low_score_data(zero=True),candidate_birth=True)
    assert all(not rows for rows in output)
    assert owner.statistics['candidate_births']==0
    global_threshold,unconditional=run(low_score_data(zero=True),global_birth_threshold=.6)
    assert all(rows for rows in unconditional)
    assert global_threshold.statistics['candidate_births']==0


def test_known_parent_does_not_spawn_a_second_identity_and_missing_IR_falls_back():
    data=low_score_data();data['scores'][0]=.95
    owner,output=run(data,candidate_birth=True)
    assert all([r[2] for r in rows]==[1] for rows in output)
    assert owner.statistics['candidate_births']==0
    data=low_score_data();data['foreground_transport']['quality'][:,1]=0
    owner,output=run(data,candidate_birth=True)
    assert owner.statistics['candidate_births']==1
    assert owner.statistics['IR_supported_births']==0
    assert len(output[2])==1
