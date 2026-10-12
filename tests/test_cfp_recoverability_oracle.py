import numpy as np

from cfp.recoverability_oracle import joint_candidate_oracle


def test_competing_tracks_cannot_both_take_one_correct_candidate():
    rows = [dict(target=1, chosen=0, candidates=[0, 1]), dict(target=1, chosen=1, candidates=[0, 1])]
    assigned, counts = joint_candidate_oracle(rows, [0, 1], np.array([1, 2]))
    assert assigned == [0, None]
    assert counts['oracle_correct'] == 1 and counts['N01'] == 0 and counts['N10'] == 0
    assert counts['wrong_to_abstention'] == 1


def test_expand_only_existing_proposals_and_preserve_unknown_reference():
    rows = [dict(target=1, chosen=1, candidates=[1]), dict(target=None, chosen=2, candidates=[2])]
    labels = np.array([1, 2, -1])
    stage, _ = joint_candidate_oracle(rows, [0, 1, 2], labels)
    full, counts = joint_candidate_oracle(rows, [0, 1, 2], labels, expand=True)
    assert stage == [None, 2] and full == [0, 2] and counts['N01'] == 1

