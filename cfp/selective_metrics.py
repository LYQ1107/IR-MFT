"""Finite selective-risk diagnostics; class labels=-1 always remain excluded."""
import numpy as np
from scipy.stats import rankdata


def risk_metrics(probabilities, labels):
    p, y = np.asarray(probabilities, float), np.asarray(labels)
    known = y >= 0
    p, y = p[known], y[known].astype(int)
    if not len(y):
        return dict(known=0, auc=None, brier=None, ece=None, risk_coverage=[])
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or np.any((y != 0) & (y != 1)):
        raise ValueError('Finite probability and known binary labels required')
    pos, neg = int(y.sum()), int((1 - y).sum())
    auc = float((rankdata(p)[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)) if pos and neg else None
    ece, bins = 0., []
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        chosen = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if chosen.any():
            gap = abs(float(p[chosen].mean()) - float(y[chosen].mean()))
            ece += chosen.mean() * gap
            bins.append(dict(lower=float(lo), upper=float(hi), count=int(chosen.sum()),
                predicted_risk=float(p[chosen].mean()), observed_risk=float(y[chosen].mean())))
    order = np.argsort(p, kind='stable')
    cumulative = np.cumsum(y[order])
    curve = []
    for coverage in (.05, .1, .2, .4, .6, .8, 1.):
        count = max(1, int(np.ceil(len(y) * coverage)))
        curve.append(dict(coverage=count / len(y), accepted=count,
            wrong=int(cumulative[count - 1]), risk=float(cumulative[count - 1] / count)))
    gates = {}
    for threshold in (.05, .1, .2, .3, .5):
        accepted = p <= threshold
        gates[str(threshold)] = dict(coverage=float(accepted.mean()),
            correct_accepted=int(np.sum(accepted & (y == 0))), wrong_accepted=int(np.sum(accepted & (y == 1))),
            correct_rejected=int(np.sum(~accepted & (y == 0))), wrong_rejected=int(np.sum(~accepted & (y == 1))))
    return dict(known=len(y), unknown_excluded=int((~known).sum()), wrong=pos, correct=neg,
        auc=auc, brier=float(np.mean((p - y)**2)), ece=float(ece), bins=bins,
        risk_coverage=curve, fixed_risk_thresholds=gates)

