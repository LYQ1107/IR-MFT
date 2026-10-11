import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from yolox.tracker.basetrack import BaseTrack

spec = importlib.util.spec_from_file_location('cfp_utility_queries_cli', Path(__file__).parents[1] / 'tools/cfp_build_utility_queries.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_supervised_identity_labels_never_change_online_outputs_and_memory_is_past_only(tmp_path):
    args = SimpleNamespace(track_thresh=.58, track_buffer=45, match_thresh=.97, mot20=False)
    config = dict(query_frame_stride=1, teacher_temperature=.1, teacher_ir_weight=.5, null_distance=.35)
    frame = np.array([[0,0,30,20,.9]], np.float32)
    features = np.array([[1.,0.]], np.float32)
    traces, trackers = [], []
    for labels in ([3,3,3], [3,7,3]):
        BaseTrack._count = 0
        tracker = module.QueryRecorder(args, np.array(labels), config, np.array([1,2,3]))
        trace = []
        for index in range(3):
            tracks = tracker.update(frame.copy(),(1,1),(1,1),features,features, np.array([index]))
            trace.append([(t.track_id,*t.tlwh.tolist()) for t in tracks])
        traces.append(trace);trackers.append(tracker)
    assert traces[0] == traces[1]
    assert trackers[0].rows[0]['target'] == 0
    assert trackers[1].rows[0]['target'] == 1  # NULL, original identity is absent.
    assert trackers[1].rows[1]['anchor_id'] == 3
    assert trackers[1].rows[1]['history'] == [(1,0),(2,1)]  # Includes actual contaminated observation.
    for row in trackers[1].rows:
        assert all(t < row['frame'] for t,_ in row['history'])
        assert not set(row['candidate_indices']) & {index for _,index in row['history']}
    path = tmp_path/'queries.npz'
    module.store_queries(path, trackers[1].rows)
    bank = np.load(path)
    assert bank['target'].tolist() == [1,0] and bank['history_indices'].shape == (2,8)
    assert bank['offsets'].tolist() == [0,1,2]


def test_ambiguous_current_identity_is_excluded_without_dropping_tracker_input():
    args = SimpleNamespace(track_thresh=.58, track_buffer=45, match_thresh=.97, mot20=False)
    config = dict(query_frame_stride=1, teacher_temperature=.1, teacher_ir_weight=.5, null_distance=.35)
    tracker = module.QueryRecorder(args,np.array([3,3]),config,np.array([1,2]))
    frame = np.array([[0,0,30,20,.9]],np.float32);features = np.array([[1.,0.]],np.float32)
    tracker.update(frame.copy(),(1,1),(1,1),features,features,np.array([0]))
    tracker.current_ambiguous={3}
    tracks=tracker.update(frame.copy(),(1,1),(1,1),features,features,np.array([1]))
    assert len(tracks)==1 and not tracker.rows
    assert tracker.exclusions['ambiguous_current_identity']==1
    assert list(tracks[0].identity_proposals)==[(1,0),(2,1)]
