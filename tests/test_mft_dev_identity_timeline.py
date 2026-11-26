import inspect
import sys
import numpy as np
import pytest
from tools.mft_dev_identity_timeline import observe_official_timeline,identity_events


def metric_and_data():
    from pathlib import Path
    from tools.native_pyramid_experiment import read,ROOT
    plan=read(ROOT/'configs/perceptual_mft_20261122.json')
    sys.path.insert(0,str(Path(plan['evaluation']['TrackEval_root'])))
    if 'float'not in np.__dict__:np.float=float
    if 'int'not in np.__dict__:np.int=int
    from trackeval.metrics.hota import HOTA
    data=dict(num_timesteps=5,num_gt_ids=2,num_tracker_ids=4,num_gt_dets=10,num_tracker_dets=9,
        gt_ids=[np.asarray([0,1])for _ in range(5)],
        tracker_ids=[np.asarray(x)for x in [[0,1],[0,1],[1,2],[1],[1,3]]],
        similarity_scores=[np.asarray(x,float)for x in [
            [[.9,0],[0,.9]],[[.9,0],[0,.9]],[[0,.9],[.9,0]],[[0],[.9]],[[0,.9],[.9,0]]]])
    return HOTA(),data,inspect.getsourcefile(HOTA)


def test_trace_captures_actual_official_time_matches_and_restores_trace():
    metric,data,path=metric_and_data();previous=sys.gettrace()
    expected=metric.eval_sequence(data)
    actual,captured,rows=observe_official_timeline(metric,data,path)
    assert sys.gettrace()is previous
    for k in metric.fields:np.testing.assert_array_equal(actual[k],expected[k])
    assert len(rows)==9 and (3,0,2)in rows and (5,0,3)in rows
    assert not any(f==4 and g==0 for f,g,p in rows)


def test_transition_events_distinguish_gap_and_existing_output_lifetimes():
    rows=[(1,7,21),(2,7,21),(3,7,23),(5,7,24)]+[(f,8,22)for f in range(1,6)]
    rows=sorted(rows)
    gt={7:set(range(1,6)),8:set(range(1,6))};pred={21:{1,2},22:set(range(1,6)),23:{3},24:{5}}
    summaries,events=identity_events('artificial','one',rows,gt,pred)
    assert len(summaries)==2 and len(events)==2
    assert events[0]['gap_category']=='continuous'and events[1]['gap_category']=='short_1_to_25'
    assert events[1]['GT_present_unmatched_intermediate_frames']==1
    assert summaries[0]['matched_ID_transitions']==2 and summaries[1]['matched_ID_transitions']==0


def test_observer_exception_restores_existing_python_trace():
    metric,data,path=metric_and_data();previous=sys.gettrace();data=dict(data);data['similarity_scores']=[]
    with pytest.raises(IndexError):observe_official_timeline(metric,data,path)
    assert sys.gettrace()is previous
