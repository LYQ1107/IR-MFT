from types import SimpleNamespace

import numpy as np

from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.cycle_candidate_tracker import CycleForegroundTracker
from test_foreground_tracker import with_transport,assert_same


def run(data,**config):
    BaseTrack._count=0
    owner=CycleForegroundTracker(dict(source_mode='rgb',target_mode='rgb',IR_increment=False,**config))
    return owner,[owner.update(f,data) for f in range(1,5)]


def test_weak_body_is_confirmed_after_three_images_without_rewriting_earlier_outputs():
    data=with_transport([[20,20,50,40]]*4,[.45]*4,[0,1,2,3,4])
    owner,output=run(data,cycle_birth=True,cycle_foreground='joint')
    assert not output[0] and not output[1]
    assert [r[2] for r in output[2]]==[1] and [r[2] for r in output[3]]==[1]
    assert owner.statistics['cycle_confirmations']==1
    assert owner.statistics['IR_supported_births']==1


def test_direct_cycle_disagreement_blocks_a_premature_identity():
    data=with_transport([[20,20,50,40]]*4,[.45]*4,[0,1,2,3,4])
    data['foreground_transport']['vote_2_rgb_joint'][:]=0
    owner,output=run(data,cycle_birth=True,cycle_foreground='joint')
    assert all(not rows for rows in output)
    assert owner.statistics['cycle_rejections']==2
    owner,unverified=run(data,cycle_birth=True,cycle_foreground='joint',candidate_cycle_check=False)
    assert len(unverified[2])==1


def test_ordinary_low_threshold_control_keeps_original_ByteTrack_output_exact():
    data=with_transport([[20,20,50,40]]*4,[.45]*4,[0,1,2,3,4])
    BaseTrack._count=0
    plain=BYTETracker(SimpleNamespace(track_thresh=.3,track_buffer=30,match_thresh=.9,mot20=False),frame_rate=25)
    reference=[]
    for frame in range(1,5):
        a,b=data['offsets'][frame-1:frame+1];dets=np.c_[data['boxes'][a:b],data['scores'][a:b]]
        reference.append([(frame,t.tlwh.copy(),t.track_id,t.score) for t in plain.update(dets,(1,1),(1,1))])
    owner,actual=run(data,low_track_threshold=.3,point_weight=0,cycle_birth=False)
    assert_same(reference,actual)
