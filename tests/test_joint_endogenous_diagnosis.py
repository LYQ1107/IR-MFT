import numpy as np
from cfp.joint_endogenous_diagnosis import classify_history, classify_selection


def test_mixed_history_retains_unknown_and_distinguishes_recent_loss_of_reference():
    labels = np.array([4, 4, 5, -1], dtype=np.int64)
    track = dict(initial_frame=1, initial_proposal=0, history=[(3, 2), (4, 3)])
    result = classify_history(track, labels, 4)
    assert result['known_correct'] == 1 and result['known_wrong'] == 1 and result['unknown'] == 1
    assert result['state'] == 'mixed_known_identity' and result['recent_known_correct'] == 0
    assert not result['latest_disagrees']  # Latest unknown must never become a known wrong identity.
    assert not result['anchor_disagrees']


def test_unknown_reference_and_nonunique_target_are_never_null_ground_truth():
    track = dict(initial_frame=1, initial_proposal=0, history=[(3, 1)])
    result = classify_history(track, np.array([-1, 7]), None)
    assert result['state'] == 'unknown_early_reference' and result['known_wrong'] is None
    assert classify_selection(-1, -1, 3) == 'unsupervised_unknown_or_nonunique'
    assert classify_selection(0, -1, 3) == 'unsupervised_unknown_or_nonunique'
    assert classify_selection(-1, 1, 3) == 'false_null_for_known_existing'
