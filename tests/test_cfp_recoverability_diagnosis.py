"""Test distinctions that would otherwise invalidate scientific error counts."""
import numpy as np
import pytest

from cfp.recoverability_diagnosis import EarlyReference, assess_candidate_row, accumulate
from collections import Counter


def example(labels=(7, 7, 7, 7, 8), geometry=(.95, .2), costs=(.94, .2), match=1):
    bank = dict(labels=np.array(labels), frame_of_proposal=np.array([1, 2, 3, 4, 4]),
        offsets=np.array([0, 1, 2, 3, 5]), boxes=np.tile([0, 0, 10, 10], (5, 1)),
        gt={4: [(7, [0, 0, 10, 10]), (8, [10, 0, 20, 10])]})
    track = dict(track_id=1, history=[(1, 0), (2, 1), (3, 2)],
        initial_proposal=0, initial_frame=1, box=[0, 0, 10, 10], state=1)
    row = dict(frame=4, tracks=[track], proposals=[3, 4], costs=[list(costs)],
        geometry=[list(geometry)], threshold=.97, matches=[] if match is None else [[0, match]], scores=[.9, .9])
    features = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [0, 1]], np.float32)
    return row, bank, features


def test_geometry_exclusion_is_not_hard_candidate_absence():
    row, bank, features = example()
    a = assess_candidate_row(row, 0, bank, features, 7)
    assert a['wrong'] and a['stage_coverage'] and not a['geometry_appearance_coverage']
    assert a['cost_feasible_coverage'] and a['opportunity'] == 'local_reranking_cost_feasible'


def test_absent_stage_but_present_detector_candidate():
    row, bank, features = example()
    row.update(proposals=[4], costs=[[.2]], geometry=[[.2]], matches=[[0, 0]], scores=[.9])
    a = assess_candidate_row(row, 0, bank, features, 7)
    assert a['detection_coverage'] and not a['stage_coverage']
    assert a['opportunity'] == 'current_stage_coverage_change'


def test_unknown_candidate_and_history_are_not_wrong():
    row, bank, features = example(labels=(7, -1, -1, 7, -1))
    a = assess_candidate_row(row, 0, bank, features, 7)
    assert not a['wrong'] and a['unknown_write']
    assert a['correct_history'] == 1 and a['unknown_history'] == 2
    row['tracks'][0]['history'] = [(2, 1), (3, 2)]
    row['tracks'][0]['initial_proposal'] = None
    a = assess_candidate_row(row, 0, bank, features, 7)
    assert a['cue_state'] == 'unverified_cue_availability'


def test_multiframe_reference_not_first_token_and_stays_frozen():
    row, bank, _ = example(labels=(8, 7, 7, 8, 7))
    ref = EarlyReference()
    assert ref.update(row['tracks'][0], bank['labels']) == 7
    row['tracks'][0]['history'] = [(1, 0), (4, 3)]
    assert ref.update(row['tracks'][0], bank['labels']) == 7


def test_unmatched_is_not_wrong_accepted_write():
    row, bank, features = example(match=None)
    a = assess_candidate_row(row, 0, bank, features, 7)
    assert a['unmatched'] and not a['accepted_memory_write_wrong']
    counts = Counter()
    accumulate(counts, a)
    assert counts['unmatched_visible_stage_covered'] == 1 and counts['wrong'] == 0


def test_empty_pool_and_future_history():
    row, bank, features = example(match=None)
    row.update(proposals=[], costs=[[]], geometry=[[]], scores=[])
    assert not assess_candidate_row(row, 0, bank, features, 7)['stage_coverage']
    row['tracks'][0]['history'].append((4, 3))
    with pytest.raises(ValueError, match='timestamp'):
        assess_candidate_row(row, 0, bank, features, 7)

