"""Offline known-label identity metrics; never imported by online association."""
import numpy as np


def physical_identity_key(video, identity):
    if not isinstance(video, str) or not video or int(identity) != identity or identity < 0:
        raise ValueError('Known nonnegative physical identity and explicit video required')
    return video, int(identity)


def pair_discrimination(positive, negative):
    """Pooled-edge AUC/EER; ties are half successes and not arbitrarily ordered."""
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError('Finite identity similarities required')
    counts = dict(positive_edges=int(positive.size), negative_edges=int(negative.size))
    if not positive.size or not negative.size:
        return dict(**counts, auc=None, eer=None, scope='No valid two-class sample')
    sorted_negative = np.sort(negative)
    lower = np.searchsorted(sorted_negative, positive, side='left')
    upper = np.searchsorted(sorted_negative, positive, side='right')
    auc = float(np.mean((lower + .5*(upper-lower))/negative.size))
    scores = np.r_[positive, negative]
    is_positive = np.r_[np.ones(positive.size, bool), np.zeros(negative.size, bool)]
    order = np.argsort(-scores, kind='stable')
    values = scores[order]
    ends = np.r_[np.flatnonzero(values[:-1] != values[1:]), len(values)-1]
    tp = np.r_[0., np.cumsum(is_positive[order])[ends]/positive.size]
    fp = np.r_[0., np.cumsum(~is_positive[order])[ends]/negative.size]
    fn = 1-tp
    difference = fp-fn
    crossing = int(np.flatnonzero(difference >= 0)[0])
    if crossing == 0 or difference[crossing] == 0:
        eer = float((fp[crossing]+fn[crossing])/2)
    else:
        a, b = crossing-1, crossing
        weight = -difference[a]/(difference[b]-difference[a])
        eer = float(fp[a]+weight*(fp[b]-fp[a]))
    return dict(**counts, auc=auc, eer=eer,
                positive_cosine_mean=float(positive.mean()), negative_cosine_mean=float(negative.mean()),
                positive_quantiles=np.quantile(positive, [.1,.5,.9]).tolist(),
                negative_quantiles=np.quantile(negative, [.1,.5,.9]).tolist(),
                scope='Pooled known-positive/known-negative edges; repeated fish/frames are not independent samples. Diagnostic EER, no deployed threshold.')


def query_retrieval(scores, labels):
    """Known-positive query ranking on the actual pool, with unknowns explicit.

    Edge labels: 0=same known fish, 1=different known fish, -1=unknown.
    Unknowns remain in actual inference pool; known-only ranking is supplementary.
    """
    scores, labels = np.asarray(scores, float), np.asarray(labels)
    if scores.ndim != 1 or labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError('Finite aligned score and label vectors required')
    if not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError('Unknowns cannot be relabeled as hard negatives')
    result = dict(candidates=int(labels.size), positive_count=int((labels==0).sum()),
                  negative_count=int((labels==1).sum()), unknown_count=int((labels<0).sum()))
    if not (labels==0).any():
        return dict(**result, target_present=False, rank1=None, rank5=None, average_precision=None,
                    unknown_selected=None, known_only_rank1=None, hard_negative_margin=None)
    order = np.argsort(-scores, kind='stable')
    ordered = labels[order]
    relevant = ordered == 0
    precisions = np.cumsum(relevant)/np.arange(1, len(ordered)+1)
    known = labels >= 0
    selected_known = int(np.argmax(scores[known]))
    negative = scores[labels==1]
    return dict(**result, target_present=True, rank1=bool(ordered[0]==0),
                rank5=bool(relevant[:5].any()), average_precision=float(precisions[relevant].mean()),
                unknown_selected=bool(ordered[0]<0), known_only_rank1=bool(labels[known][selected_known]==0),
                hard_negative_margin=float(scores[labels==0].max()-negative.max()) if negative.size else None,
                tie_policy='Stable proposal order for query ranks; pooled pair AUC handles ties explicitly')
