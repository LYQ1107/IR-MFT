import numpy as np
import pytest

from cfp.explanation_diagnosis import explanations


def test_retained_explanations_use_observed_birth_and_keep_unknown_separate():
    labels = np.array([10, 20, 20, 10, -1])
    frames = np.array([1, 1, 2, 2, 2])
    birth = [1, 1, 0]
    event = dict(frame=2, winner=0, hypotheses=[
        dict(branch=0, current=[dict(proposal=2, birth=birth)]),
        dict(branch=1, current=[dict(proposal=3, birth=birth), dict(proposal=4, birth=[2, 2, 4])]),
    ])
    result = explanations(event, labels, frames)
    assert result[0][tuple(birth)] == dict(known=True, correct=False)
    assert result[1][tuple(birth)] == dict(known=True, correct=True)
    assert result[1][(2, 2, 4)] == dict(known=False, correct=False)


def test_diagnosis_rejects_current_proposal_from_another_frame():
    event = dict(frame=2, winner=0, hypotheses=[dict(branch=0, current=[dict(proposal=0, birth=[1, 1, 0])])])
    with pytest.raises(ValueError, match='timestamp drift'):
        explanations(event, np.array([10]), np.array([1]))
