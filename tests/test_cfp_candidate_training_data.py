import numpy as np

from cfp.candidate_training_data import candidate_batch
from cfp.recoverability import FEATURE_NAMES


def sample(positive=0, null_known=True):
    features = np.zeros((6, 512), np.float32)
    features[:, 0] = np.arange(1, 7)
    bank = dict(features=features, labels=np.array([1, 2, 1, -1, 1, 2]),
        frames=np.array([1, 1, 2, 2, 8, 8]), boxes=np.tile([0, 0, 10, 10], (6, 1)), offsets=np.array([0, 2, 4]))
    row = dict(sequence='v', frame=2, identity=1, indices=np.array([0] + [-1] * 7),
        times=np.array([1] + [0] * 7), candidates=np.array([2, 3]),
        evidence=np.zeros((2, len(FEATURE_NAMES)), np.float32), edge_labels=np.array([0, -1]),
        positive=positive, null_known=null_known)
    return row, {'v': bank}


def test_unknown_is_not_a_classification_negative_and_future_teacher_is_separate():
    row, banks = sample()
    now = candidate_batch([row], banks, 'cpu')
    future = candidate_batch([row], banks, 'cpu', target_time='future')
    assert now[7].tolist() == [[True, False, True]]
    assert now[8].tolist() == [0]
    for i in range(9):
        assert np.array_equal(now[i].numpy(), future[i].numpy())
    assert now[9][0, 0] == 3 and future[9][0, 0] == 5


def test_absent_positive_maps_to_global_null_and_unknown_null_is_not_labelled():
    row, banks = sample(positive=2, null_known=False)
    result = candidate_batch([row], banks, 'cpu')
    assert result[8].tolist() == [2] and not result[7][0, -1]
    assert not result[10].any()

