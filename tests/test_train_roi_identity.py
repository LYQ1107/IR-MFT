import numpy as np
import pytest
import torch

from tools.train_roi_identity import identities_and_boxes, normalized, retrieval, geometry_scores, p3_branch
from yolox.models.structural_factorial_repair import StructuralFactorial


def test_reordered_gallery_uses_identity_instead_of_row_position():
    q = np.eye(2, dtype=np.float32)
    g = q[::-1].copy()
    result, correct, known = retrieval(q @ g.T, [10, 20], [20, 10], [True, True], [True, True])
    assert result['queries'] == 2 and result['correct'] == 2
    assert correct.tolist() == [True, True] and known.tolist() == [0, 1]


def test_invalid_query_and_positive_gallery_are_misses_not_excluded():
    values, valid = normalized(np.array([[0., 0.], [0., 1.]], np.float32))
    stats, correct, _ = retrieval(values @ np.eye(2, dtype=np.float32), [1, 2], [1, 2], valid, [True, False])
    assert stats['queries'] == 2 and stats['correct'] == 0
    assert stats['invalid_query'] == 1 and stats['invalid_positive_gallery'] == 1
    assert not correct.any()


def test_unseen_queries_are_reported_and_empty_gallery_is_well_defined():
    stats, correct, known = retrieval(np.empty((2, 0), np.float32), [1, 2], [], [True, True], [])
    assert stats['queries'] == 0 and stats['unseen_identity_queries'] == 2
    assert len(correct) == len(known) == 0


def test_geometry_reference_uses_original_boxes_and_known_labels():
    q = np.array([[1., 0., 11., 10.], [21., 0., 31., 10.]], np.float32)
    g = np.array([[20., 0., 30., 10.], [0., 0., 10., 10.]], np.float32)
    stats, correct, _ = retrieval(geometry_scores(q, g), [10, 20], [20, 10], [True, True], [True, True])
    assert stats['correct'] == 2 and correct.all()


def test_GT_box_and_cached_target_alignment_rejects_a_changed_label_source():
    gt = np.array([[3, 7, 10, 20, 30, 40, 1, 1, 1], [3, 8, 50, 60, 20, 10, 1, 1, 1]], np.float64)
    targets = np.zeros((4, 5), np.float32)
    expected = gt[:, 2:6].astype(np.float32).copy()
    expected[:, :2] += expected[:, 2:] / 2
    expected *= .5
    targets[:2, 1:] = expected
    ids, boxes = identities_and_boxes(gt, 3, .5, targets)
    assert ids.tolist() == [7, 8]
    np.testing.assert_array_equal(boxes, [[10, 20, 40, 60], [50, 60, 70, 70]])
    targets[0, 1] += 1
    with pytest.raises(ValueError, match='cached targets'):
        identities_and_boxes(gt, 3, .5, targets)


@pytest.mark.parametrize('bits', ['111', '011', '101', '110'])
@pytest.mark.parametrize('null', [False, True])
def test_partial_P3_analysis_exact_to_actual_full_fusion_with_nonzero_residual(bits, null):
    torch.manual_seed(0)
    module = StructuralFactorial(latent=4, radius=1, temperature=.1, bound=.5,
        risk_threshold=.6, components=bits, tolerance=1).eval().requires_grad_(False)
    with torch.no_grad():
        for layer in module.levels:
            layer.output.weight.fill_(.02)
            layer.output.bias.fill_(.03)
        rgb = tuple(torch.linspace(.1, 1., c*h*w).reshape(1, c, h, w)
            for c, h, w in [(320, 4, 8), (640, 2, 4), (1280, 1, 2)])
        auxiliary = tuple(value.flip(-1) + .2 for value in rgb)
        scores = tuple(torch.full((1, 1, *value.shape[-2:]), .9) for value in rgb)
        query = torch.linspace(.1, 1., 12*8*16).reshape(1, 12, 8, 16)
        key = query.flip(-1)
        valid = torch.ones((1, 1, 8, 16))
        full = module(rgb, auxiliary, scores, query, key, valid, valid, True, null)[0]
        partial = p3_branch(module, rgb[0], auxiliary[0], scores[0], query, key, valid, valid, True, null)
        assert torch.equal(full, partial)
