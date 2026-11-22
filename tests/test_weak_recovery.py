import numpy as np

from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.weak_recovery_tracker import RecoverableForegroundTracker
from test_foreground_tracker import with_transport


def run(data,**config):
    BaseTrack._count=0
    owner=RecoverableForegroundTracker(dict(source_mode='rgb',target_mode='rgb',IR_increment=False,weak_recovery_geometry='rgb',**config))
    return owner,[owner.update(f,data) for f in range(1,5)]


def test_lost_identity_is_recovered_from_weak_current_body_without_a_new_ID():
    data=with_transport([[20,20,50,40]]*3,[.95,.45,.45],[0,1,1,2,3])
    plain,reference=run(data,weak_recovery=False)
    assert len(reference[0])==1 and all(not rows for rows in reference[1:])
    owner,output=run(data,weak_recovery=True)
    assert not output[1]
    assert [r[2] for r in output[2]]==[1] and [r[2] for r in output[3]]==[1]
    assert owner.statistics['weak_recoveries']==1
    assert owner.statistics['candidate_births']==0


def test_two_lost_objects_recover_one_to_one_after_weak_crossing():
    first=[[20,20,50,40],[70,20,100,40]]
    crossed=[[70,20,100,40],[20,20,50,40]]
    data=with_transport(first+crossed+crossed,[.95,.95]+[.45]*4,[0,2,2,4,6])
    owner,output=run(data,weak_recovery=True)
    assert len({r[2] for r in output[2]})==2
    centers={r[2]:r[1][0]+r[1][2]/2 for r in output[2]}
    assert centers[1]>centers[2]
    assert owner.statistics['weak_recoveries']==2


def test_conflicting_weak_ownership_does_not_reactivate_arbitrary_old_identity():
    first=[[20,20,50,40],[70,20,100,40]]
    data=with_transport(first+[[40,20,70,40]]*2,[.95,.95,.45,.45],[0,2,2,3,4])
    point=data['foreground_transport']
    point['vote_2_rgb_rgb'][:]=1
    owner,output=run(data,weak_recovery=True)
    assert all(not rows for rows in output[1:])
    assert owner.statistics['weak_recoveries']==0
