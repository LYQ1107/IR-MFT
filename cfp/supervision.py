"""Retrospective current-proposal labels, never used by the online tracker."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def independent_proposal_labels(boxes, gt_boxes, gt_ids, minimum_iou=.5, minimum_gap=.15):
    """Physical-label audit: no track state, continuity bonus or future frames.

    Ambiguous crop/GT competition is removed before independent assignment.
    Return IDs -1 for unavailable labels and GT IDs with ambiguous overlap.
    """
    boxes, gt_boxes = [np.asarray(x, dtype=np.float64).reshape(-1, 4) for x in (boxes, gt_boxes)]
    gt_ids = np.asarray(gt_ids, dtype=np.int64)
    if len(gt_ids) != len(gt_boxes) or len(np.unique(gt_ids)) != len(gt_ids):
        raise ValueError('Unique current-frame GT identities required')
    if not np.isfinite(boxes).all() or not np.isfinite(gt_boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]) or np.any(gt_boxes[:, 2:] <= gt_boxes[:, :2]):
        raise ValueError('Invalid supervision boxes')
    labels = np.full(len(boxes), -1, dtype=np.int64)
    if not len(boxes) or not len(gt_boxes):
        return labels, set()
    left = np.maximum(boxes[:, None, :2], gt_boxes[None, :, :2])
    right = np.minimum(boxes[:, None, 2:], gt_boxes[None, :, 2:])
    intersection = np.maximum(right - left, 0).prod(-1)
    areas = (boxes[:, 2:] - boxes[:, :2]).prod(-1)
    gt_areas = (gt_boxes[:, 2:] - gt_boxes[:, :2]).prod(-1)
    iou = intersection / np.maximum(areas[:, None] + gt_areas[None] - intersection, 1e-12)
    def unambiguous(values, axis):
        ordered = np.sort(values, axis=axis)
        first = np.take(ordered, -1, axis=axis)
        second = np.take(ordered, -2, axis=axis) if values.shape[axis] > 1 else np.zeros_like(first)
        return (first >= minimum_iou) & (first - second >= minimum_gap)
    row_ok, col_ok = unambiguous(iou, 1), unambiguous(iou, 0)
    allowed = row_ok[:, None] & col_ok[None, :] & (iou >= minimum_iou)
    row, col = linear_sum_assignment(np.where(allowed, -iou, 1.))
    for r, c in zip(row, col):
        if allowed[r, c]:
            labels[r] = gt_ids[c]
    labelled = set(labels[labels >= 0].tolist())
    overlapping = {int(gt_ids[c]) for c in np.flatnonzero((iou >= minimum_iou).any(0))}
    return labels, overlapping - labelled
