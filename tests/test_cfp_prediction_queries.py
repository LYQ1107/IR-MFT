import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from yolox.tracker.basetrack import BaseTrack

spec=importlib.util.spec_from_file_location('cfp_prediction_query_cli',Path(__file__).parents[1]/'tools/cfp_build_prediction_queries.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def run(labels,ambiguous=None):
    BaseTrack._count=0
    tracker=module.PredictionRecorder(SimpleNamespace(track_thresh=.58,track_buffer=45,match_thresh=.97,mot20=False),
          np.array(labels),np.repeat(np.arange(1,5),2),dict(history_length=8,gap_training_candidates=[1],query_frame_stride=1),0)
    boxes=np.array([[0,0,30,20,.9],[70,0,100,20,.9]],np.float32)
    features=np.eye(2,dtype=np.float32);trace=[]
    for frame in range(1,5):
        tracker.ambiguous=set(ambiguous or [])
        tracks=tracker.update(boxes.copy(),(1,1),(1,1),features,features,np.arange(2*(frame-1),2*frame))
        trace.append([(t.track_id,*t.tlwh.tolist()) for t in tracks])
        tracker.collect_writes(frame,2*(frame-1),2*frame,np.ones(8,bool))
    return tracker,trace


def test_physical_supervision_never_changes_outputs_and_keeps_contaminated_past(tmp_path):
    clean,reference=run([3,7]*4)
    contaminated,trace=run([3,7,7,3,3,7,3,7])
    assert trace==reference
    row=next(r for r in contaminated.rows if r['frame']==3 and r['anchor_id']==3)
    assert row['actual']==[(1,0),(2,2)]
    assert row['clean']==[(1,0),(2,3)]
    assert row['purity']==.5
    assert row['candidate_indices'][row['positive_index']]==4
    for r in contaminated.rows:
        assert all(t<=r['frame']-r['requested_gap'] and t<r['frame'] for t,_ in r['actual']+r['clean'])
        assert not set(r['candidate_indices']) & {i for _,i in r['actual']+r['clean']}
    path=tmp_path/'queries.npz';module.save_rows(path,contaminated.rows,8)
    with np.load(path) as data:
        assert data['actual_history_indices'].shape==(6,8)
        assert data['candidate_offsets'].tolist()==list(range(0,13,2))
        assert np.all(data['actual_history_frames']<data['frame'][:,None])


def test_ambiguous_identity_does_not_become_anchor_or_negative():
    tracker,trace=run([3,7]*4,ambiguous=[7])
    assert all(len(frame)==2 for frame in trace)
    assert set(tracker.anchors.values())=={3}
    assert not tracker.rows  # One usable fish cannot supply a distinct hard negative.
    assert not tracker.clean[7]
