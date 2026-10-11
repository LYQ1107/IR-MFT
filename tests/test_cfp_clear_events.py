import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('cfp_clear_trace_cli', Path(__file__).parents[1] / 'tools/cfp_trace_clear_events.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exact_clear_switch_trace_handles_empty_frames_and_long_match_gaps():
    root = Path(__file__).resolve().parents[2] / 'CFP-FishMOT_references/TrackEval'
    path = root / 'trackeval/metrics/clear.py'
    if not path.exists():
        pytest.skip('Pinned official TrackEval checkout required')
    sys.path.insert(0, str(root))
    from trackeval.metrics import CLEAR
    data = dict(num_tracker_dets=2, num_gt_dets=3, num_gt_ids=1, num_tracker_ids=2, num_timesteps=4,
                gt_ids=[np.array([0]), np.array([0]), np.array([], dtype=int), np.array([0])],
                tracker_ids=[np.array([0]), np.array([], dtype=int), np.array([], dtype=int), np.array([1])],
                gt_dets=[np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10]]), np.zeros((0, 4)), np.array([[0, 0, 10, 10]])],
                tracker_dets=[np.array([[0, 0, 10, 10]]), np.zeros((0, 4)), np.zeros((0, 4)), np.array([[0, 0, 10, 10]])],
                similarity_scores=[np.array([[.9]]), np.zeros((1, 0)), np.zeros((0, 0)), np.array([[.9]])])
    metric = CLEAR({'PRINT_CONFIG': False})
    matches = []
    result, events = module.capture_clear(metric, data, path,
        lambda t, values: matches.extend((t + 1, int(gt), int(track)) for gt, track in
                                        zip(values['matched_gt_ids'], values['matched_tracker_ids'])))
    assert result['IDSW'] == 1 and result['CLR_FN'] == 1
    assert len(events) == 1 and events[0]['frame'] == 4
    assert events[0]['previous_tracker_id_relabelled'] == 0 and events[0]['new_tracker_id_relabelled'] == 1
    assert events[0]['gap_since_previous_gt_match'] == 3
    assert matches == [(1, 0, 0), (4, 0, 1)]
    assert len(matches) == result['CLR_TP']
    assert sys.gettrace() is None
