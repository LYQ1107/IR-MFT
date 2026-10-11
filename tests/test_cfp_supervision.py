import numpy as np
import pytest

from cfp.supervision import independent_proposal_labels


def test_labels_depend_only_on_current_physical_boxes_and_allow_unmatched():
    gt = np.array([[0, 0, 10, 10], [30, 0, 40, 10]])
    boxes = np.array([[30, 0, 40, 10], [0, 0, 10, 10], [100, 100, 110, 110]])
    labels, ambiguous = independent_proposal_labels(boxes, gt, [7, 3])
    assert labels.tolist() == [3, 7, -1] and not ambiguous
    reordered, _ = independent_proposal_labels(boxes[::-1], gt, [7, 3])
    assert np.array_equal(labels[::-1], reordered)
    empty, ambiguous = independent_proposal_labels(np.zeros((0, 4)), gt, [7, 3])
    assert len(empty) == 0 and not ambiguous


def test_ambiguous_crop_and_duplicate_predictions_cannot_anchor_identity():
    labels, ambiguous = independent_proposal_labels([[0, 0, 10, 10]], [[0, 0, 10, 10], [.5, 0, 10.5, 10]], [3, 7])
    assert labels.tolist() == [-1] and ambiguous == {3, 7}
    labels, ambiguous = independent_proposal_labels([[0, 0, 10, 10], [0, 0, 10, 10]], [[0, 0, 10, 10]], [3])
    assert labels.tolist() == [-1, -1] and ambiguous == {3}
    with pytest.raises(ValueError):
        independent_proposal_labels([[0, 0, 10, 10]], [[0, 0, 10, 10], [30, 0, 40, 10]], [3, 3])
