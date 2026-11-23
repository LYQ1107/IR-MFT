import torch

from tools.indexed_postprocess import indexed_postprocess, corner_boxes
from yolox.utils import postprocess


def test_indexed_selection_exactly_matches_existing_postprocess_and_keeps_inputs():
    torch.manual_seed(18)
    for classes in [1, 3]:
        value = torch.rand(4, 64, 5+classes)
        value[:, :, :2] *= 400; value[:, :, 2:4] = value[:, :, 2:4]*100+10
        original = value.clone()
        expected = postprocess(value.clone(), classes, .1, .7)
        current, anchors = indexed_postprocess(value, classes, .1, .7)
        torch.testing.assert_close(value, original, rtol=0, atol=0)
        corners = corner_boxes(original)
        for i, (a, b, index) in enumerate(zip(expected, current, anchors)):
            if a is None:
                assert b is None and index is None
            else:
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(b[:, :4], corners[i, index], rtol=0, atol=0)
                torch.testing.assert_close(b[:, 4], original[i, index, 4], rtol=0, atol=0)


def test_empty_low_confidence_and_tied_duplicate_proposals_keep_provenance():
    empty = torch.empty(2, 0, 6)
    result, index = indexed_postprocess(empty, 1)
    assert result == [None, None] and index == [None, None]
    value = torch.tensor([[[20., 20., 10., 10., .9, .9], [20., 20., 10., 10., .9, .9],
                           [80., 20., 10., 10., .01, .9]]])
    old = postprocess(value.clone(), 1, .1, .7)
    new, index = indexed_postprocess(value, 1)
    torch.testing.assert_close(old[0], new[0], rtol=0, atol=0)
    assert len(index[0]) == 1 and int(index[0][0]) in [0, 1]
