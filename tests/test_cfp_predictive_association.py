from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cfp.association import IdentityTrack
from cfp.predictive import LatentIdentityPredictor
from cfp.predictive_association import PredictiveAssociationTracker
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker import matching

ARGS=SimpleNamespace(track_thresh=.58,track_buffer=45,match_thresh=.97,mot20=False)


def trace(memory,future=False):
    BaseTrack._count=0;model=LatentIdentityPredictor(2,'gru',hidden=8) if memory=='predictor' else None
    tracker=PredictiveAssociationTracker(ARGS,predictor=model,memory=memory)
    result=[]
    for frame in range(1,5):
        boxes=np.array([[frame,0,30+frame,20,.9],[70-frame,0,100-frame,20,.8]],np.float32)
        features=np.eye(2,dtype=np.float32)
        if future and frame==4:features=features[::-1].copy()
        tracks=tracker.update(boxes,(1,1),(1,1),features)
        result.append([(t.track_id,*t.tlwh.tolist()) for t in tracks])
        assert all(t.last_identity_frame==frame for t in tracks)
    return result


def test_initial_predictor_matches_last_memory_trace_and_future_changes_preserve_prefix():
    assert trace('predictor')==trace('last')
    assert trace('predictor',True)[:3]==trace('predictor')[:3]


def test_shared_graph_includes_weak_appearance_and_missing_crop_keeps_geometry():
    track=IdentityTrack([0,0,30,20],.9,np.array([1.,0.]),None)
    track.write_identity(1,.9)
    detections=[IdentityTrack([0,0,30,20],.9,np.array([0.,1.]),None),IdentityTrack([1,0,30,20],.9,None,None)]
    raw=matching.iou_distance([track],detections);geometry=matching.fuse_score(raw,detections)
    for memory in ['last','ema','fifo','all_history_ema','constant_zero','constant_half']:
        tracker=PredictiveAssociationTracker(ARGS,memory=memory);tracker.frame_id=2
        costs=tracker.association_cost([track],detections)
        assert costs[0,1]==geometry[0,1] and costs[0,0]!=geometry[0,0]
        assert tracker.prediction_counts['modified_edges']==1  # .5 cosine edge remains in every matched graph.
    tracker.frame_id=1
    with pytest.raises(ValueError,match='Predict before'):tracker.association_cost([track],detections)
