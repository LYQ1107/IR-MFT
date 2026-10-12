import numpy as np

from cfp.recovery_supervision import modal_identity, history_view, classify_row


def test_history_reference_does_not_permanently_trust_a_wrong_first_anchor():
    labels = np.array([2, 1, 1, 1, -1])
    track = dict(history=[(10, 1), (20, 2), (40, 3)], initial_frame=1, initial_proposal=0)
    history = history_view(track)
    target, consensus = modal_identity([index for _, index in history], labels)
    assert target == 1 and consensus == .75
    assert modal_identity([0, 1], labels)[0] is None
    assert modal_identity([4], labels)[0] is None


def test_unknown_current_label_is_not_classified_as_wrong_match():
    labels = np.array([1, 1, -1])
    row = dict(tracks=[dict(track_id=7, state=1, history=[(1, 0), (40, 1)], initial_frame=1, initial_proposal=0)],
               proposals=[2], costs=[[.1]], geometry=[[.1]], threshold=.97, scores=[.99], boxes=[[0, 0, 10, 10]], matches=[[0, 0]])
    result = classify_row(row, 0, labels, 'gru')
    assert result['correct_match'] is None and not result['categories']


def test_geometric_exclusion_and_confident_wrong_reactivation_are_distinct():
    labels = np.array([1, 1, 1, 2])
    row = dict(tracks=[dict(track_id=7, state=2, history=[(1, 0), (40, 1)], initial_frame=1, initial_proposal=0)],
               proposals=[2, 3], costs=[[.99, .1]], geometry=[[.95, .1]], threshold=.97, scores=[.9, .95],
               boxes=[[0, 0, 10, 10], [30, 0, 40, 10]], matches=[[0, 1]])
    result = classify_row(row, 0, labels, 'gru')
    assert result['geometry_excluded_columns'] == [0]
    assert {'true_candidate_geometry_excluded', 'wrong_reactivation', 'high_confidence_wrong_match'} <= set(result['categories'])
    assert result['row_margin'] > .8

