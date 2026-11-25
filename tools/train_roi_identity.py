"""Train-only oracle-ROI diagnostics; no tracker decision or parameter update."""
import numpy as np
import torch


def identities_and_boxes(gt, frame, ratio, cached_targets):
    rows = gt[(gt[:, 0] == frame) & (gt[:, 6] > 0) & (gt[:, 7] == 1)]
    ids = rows[:, 1].astype(np.int64)
    if not np.array_equal(ids, rows[:, 1]) or len(set(ids)) != len(ids) or np.any(ids <= 0):
        raise ValueError('Unambiguous positive Train identities required in each frame')
    expected = rows[:, 2:6].astype(np.float32).copy()
    expected[:, :2] += expected[:, 2:] / 2
    expected *= ratio
    target = cached_targets[(cached_targets[:, 3] > 0) & (cached_targets[:, 4] > 0)]
    if target.shape != (len(rows), 5) or not np.array_equal(target[:, 1:], expected):
        raise ValueError('Original cached targets and current canonical Train GT disagree')
    boxes = rows[:, 2:6].astype(np.float32).copy()
    boxes[:, 2:] += boxes[:, :2]
    return ids, boxes


@torch.no_grad()
def p3_branch(module, rgb, auxiliary, score, query, key, query_valid, key_valid, available, null):
    # The parent StructuralFactorial applies this intervention before calling
    # its independent per-level modules. Preserve it in this P3-only analysis.
    effective_score = score if module.feature_risk else torch.zeros_like(score)
    return module.levels[0](rgb, auxiliary, effective_score, query, key,
        query_valid, key_valid, available, null)


def normalized(values):
    x = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1)
    valid = norms > 1e-12
    result = np.zeros_like(x)
    result[valid] = x[valid] / norms[valid, None]
    return result, valid


def geometry_scores(query_boxes, gallery_boxes):
    q, g = np.asarray(query_boxes), np.asarray(gallery_boxes)
    centers_q = (q[:, :2] + q[:, 2:]) / 2
    centers_g = (g[:, :2] + g[:, 2:]) / 2
    scale = np.maximum(q[:, 2:] - q[:, :2], 1e-6)
    return -(((centers_q[:, None] - centers_g[None]) / scale[:, None]) ** 2).sum(-1)


def retrieval(scores, query_ids, gallery_ids, valid_query, valid_gallery):
    """Count invalid embeddings as misses; retain every known-identity query."""
    scores = np.array(scores, dtype=np.float32, copy=True)
    q, g = np.asarray(query_ids), np.asarray(gallery_ids)
    if scores.shape != (len(q), len(g)) or len(set(q)) != len(q) or len(set(g)) != len(g):
        raise ValueError('One oracle ROI per identity and a complete score matrix required')
    if not np.isfinite(scores).all():
        raise ValueError('Only finite raw descriptor scores are accepted')
    known = np.flatnonzero(np.isin(q, g))
    gallery_index = {int(identity): i for i, identity in enumerate(g)}
    truth = np.array([gallery_index[int(q[i])] for i in known], dtype=np.int64)
    filtered = scores[known]
    valid_q = np.asarray(valid_query, bool)[known]
    valid_g = np.asarray(valid_gallery, bool)
    filtered[~valid_q] = -np.inf
    filtered[:, ~valid_g] = -np.inf
    if len(g):
        prediction = filtered.argmax(1)
        any_valid = np.isfinite(filtered).any(1)
        correct = any_valid & (prediction == truth)
        positive = filtered[np.arange(len(known)), truth]
        negative_scores = filtered.copy()
        negative_scores[np.arange(len(known)), truth] = -np.inf
        negative = negative_scores.max(1)
        finite = np.isfinite(positive) & np.isfinite(negative)
        margin = positive[finite] - negative[finite]
        top = filtered.max(1)
        difference = np.full_like(filtered, np.inf)
        np.subtract(filtered, top[:, None], out=difference, where=np.isfinite(filtered) & any_valid[:, None])
        ties = ((np.abs(difference) <= 1e-8).sum(1) > 1) & any_valid
    else:
        correct = np.zeros(len(known), bool)
        margin = np.empty(0, np.float32)
        ties = np.zeros(len(known), bool)
    stats = dict(queries=len(known), unseen_identity_queries=len(q)-len(known),
        correct=int(correct.sum()), invalid_query=int((~valid_q).sum()),
        invalid_positive_gallery=int((~valid_g[truth]).sum()) if len(known) else 0,
        tied_top1=int(ties.sum()), margin_count=len(margin), margin_sum=float(margin.astype(np.float64).sum()),
        margin_nonpositive=int((margin <= 0).sum()))
    return stats, correct, known


def add_stats(target, values):
    for key, value in values.items():
        target[key] = target.get(key, 0) + value


def summary(values):
    result = dict(values)
    result['top1_percent'] = None if not result.get('queries') else 100 * result['correct'] / result['queries']
    result['mean_hard_negative_margin'] = None if not result.get('margin_count') else result['margin_sum'] / result['margin_count']
    result['net_RGB_corrections'] = result.get('corrected_RGB_errors', 0) - result.get('reversed_RGB_correct', 0)
    return result
