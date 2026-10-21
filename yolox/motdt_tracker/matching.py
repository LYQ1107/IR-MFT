import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    import lap
except ImportError:  # optional acceleration dependency
    lap = None
from scipy.spatial.distance import cdist

try:
    from cython_bbox import bbox_overlaps as bbox_ious
except ImportError:  # pure NumPy fallback keeps evaluator importable
    def bbox_ious(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.size == 0 or b.size == 0:
            return np.zeros((len(a), len(b)), dtype=np.float64)
        top_left = np.maximum(a[:, None, :2], b[None, :, :2])
        bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
        wh = np.clip(bottom_right - top_left, 0.0, None)
        intersection = wh[..., 0] * wh[..., 1]
        area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
        area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
        union = area_a[:, None] + area_b[None, :] - intersection
        return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
from yolox.motdt_tracker import kalman_filter

if "float" not in np.__dict__:
    np.float = float


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    if lap is not None:
        matches, unmatched_a, unmatched_b = [], [], []
        cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
        return np.asarray(matches, dtype=int).reshape(-1, 2), unmatched_a, unmatched_b
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
    finite = np.isfinite(cost_matrix)
    values = cost_matrix[finite]
    scale = max(float(np.max(np.abs(values))) if values.size else 1.0, abs(float(thresh)), 1.0)
    safe = np.where(finite, cost_matrix, scale * 1e6)
    rows, cols = linear_sum_assignment(safe)
    accepted = [(int(row), int(col)) for row, col in zip(rows, cols) if finite[row, col] and cost_matrix[row, col] <= thresh]
    matches = np.asarray(accepted, dtype=int).reshape(-1, 2)
    matched_a = {row for row, _ in accepted}
    matched_b = {col for _, col in accepted}
    unmatched_a = np.asarray([row for row in range(cost_matrix.shape[0]) if row not in matched_a], dtype=int)
    unmatched_b = np.asarray([col for col in range(cost_matrix.shape[1]) if col not in matched_b], dtype=int)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray
    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]
    :rtype cost_matrix np.ndarray
    """
    atlbrs = [track.tlbr for track in atracks]
    btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def nearest_reid_distance(tracks, detections, metric='cosine'):
    """
    Compute cost based on ReID features
    :type tracks: list[STrack]
    :type detections: list[BaseTrack]
    :rtype cost_matrix np.ndarray
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix

    det_features = np.asarray([track.curr_feature for track in detections], dtype=np.float32)
    for i, track in enumerate(tracks):
        cost_matrix[i, :] = np.maximum(0.0, cdist(track.features, det_features, metric).min(axis=0))

    return cost_matrix


def mean_reid_distance(tracks, detections, metric='cosine'):
    """
    Compute cost based on ReID features
    :type tracks: list[STrack]
    :type detections: list[BaseTrack]
    :type metric: str
    :rtype cost_matrix np.ndarray
    """
    cost_matrix = np.empty((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix

    track_features = np.asarray([track.curr_feature for track in tracks], dtype=np.float32)
    det_features = np.asarray([track.curr_feature for track in detections], dtype=np.float32)
    cost_matrix = cdist(track_features, det_features, metric)

    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix
