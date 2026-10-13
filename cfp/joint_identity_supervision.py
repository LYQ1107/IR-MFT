"""Training-only physical labels for full-frame ID/NULL supervision.

Unknown identities are ignored, never fabricated hard negatives. Physical IDs
are local to one explicit video; model inference never imports this module.
"""
import torch
from torch.nn import functional as F


def joint_identity_targets(video, track_identities, detection_identities):
    if not isinstance(video, str) or not video:
        raise ValueError('Explicit physical-identity video required')
    if (track_identities.ndim != 1 or detection_identities.ndim != 1
            or track_identities.dtype != torch.int64 or detection_identities.dtype != torch.int64
            or (track_identities < -1).any() or (detection_identities < -1).any()):
        raise ValueError('Known nonnegative fish IDs or -1 unknown required')
    n, m = len(track_identities), len(detection_identities)
    target = torch.full((m,), -1, dtype=torch.int64, device=detection_identities.device)
    classes = torch.zeros(m, n + 1, dtype=torch.bool, device=target.device)
    reason = []
    for j, identity in enumerate(detection_identities):
        if identity < 0:
            reason.append('unknown_detection')
            continue
        # Duplicate proposals or duplicate tracks have no unique one-to-one ID
        # target. Leave them in model inputs, exclude them from this ID loss.
        matching = (track_identities == identity).nonzero(as_tuple=True)[0]
        if (detection_identities == identity).sum() != 1 or len(matching) > 1:
            reason.append('nonunique_physical_assignment')
            continue
        if len(matching) == 1:
            target[j] = matching[0]
            classes[j, :n] = track_identities >= 0
            classes[j, n] = True
            reason.append('known_existing')
        elif (track_identities >= 0).all():
            target[j] = n
            classes[j] = True
            reason.append('known_no_old_match_not_verified_birth')
        else:
            reason.append('old_identity_coverage_unknown')
    return dict(target=target, classes=classes, reason=reason, video=video)


def joint_identity_loss(logits, supervision, legal, margin=.2,
                        ranking_weight=.1, consistency_weight=.1):
    target, classes = supervision['target'], supervision['classes']
    if (logits.ndim != 2 or classes.shape != logits.shape or target.shape != (len(logits),)
            or legal.shape != (logits.shape[1] - 1, len(logits))):
        raise ValueError('Aligned whole-frame ID supervision required')
    if min(margin, ranking_weight, consistency_weight) < 0:
        raise ValueError('Nonnegative declared ID loss weights required')
    allowed = classes.clone()
    allowed[:, :-1] &= legal.T
    selected = target >= 0
    rows = torch.arange(len(logits), device=logits.device)
    selected &= allowed[rows, target.clamp_min(0)]
    selected &= torch.isfinite(logits[rows, target.clamp_min(0)])
    if not selected.any():
        # Keep a differentiable finite zero even when masked logits are -inf.
        zero = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits)).sum() * 0
        return zero, dict(supervised_detections=0, existing=0, null=0,
                          classification=0., ranking=0., consistency=0.)
    active = logits[selected].masked_fill(~allowed[selected], -torch.inf)
    labels = target[selected]
    classification = F.cross_entropy(active, labels)
    probabilities = F.softmax(active, dim=-1)
    one_to_one = F.relu(probabilities[:, :-1].sum(0) - 1).square().sum() / max(len(labels), 1)
    existing = labels < logits.shape[1] - 1
    ranking = classification * 0
    if existing.any():
        values = active[existing]
        correct = values.gather(1, labels[existing, None])
        negatives = torch.isfinite(values[:, :-1])
        negatives.scatter_(1, labels[existing, None], False)
        violations = F.relu(margin + values[:, :-1] - correct)
        if negatives.any():
            ranking = violations[negatives].mean()
    loss = classification + ranking_weight * ranking + consistency_weight * one_to_one
    return loss, dict(supervised_detections=int(selected.sum()), existing=int(existing.sum()),
                      null=int((~existing).sum()), classification=float(classification.detach()),
                      ranking=float(ranking.detach()), consistency=float(one_to_one.detach()))
