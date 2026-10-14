"""Training-only set-valued known identity targets; never feed labels online."""
import torch
from torch.nn import functional as F


def consistent_past_identity(indices, labels, minimum_known=3):
    """Only entirely known, same-identity recent histories are supervised."""
    values = labels[indices]
    if len(values) < minimum_known or (values < 0).any() or len(set(values.tolist())) != 1:
        return -1
    return int(values[0])


def joint_identity_target_sets(track_ids, current_ids):
    if track_ids.dtype != torch.int64 or current_ids.dtype != torch.int64 or track_ids.ndim != 1 or current_ids.ndim != 1 or (track_ids < -1).any() or (current_ids < -1).any():
        raise ValueError('Known local physical IDs or -1unknown required')
    n, m = len(track_ids), len(current_ids)
    positives = torch.zeros(m, n+1, dtype=torch.bool, device=current_ids.device)
    allowed = torch.zeros_like(positives)
    reason = []
    for j, identity in enumerate(current_ids):
        if identity < 0:
            reason.append('unknown_current'); continue
        if (current_ids == identity).sum() != 1:
            reason.append('duplicate_current_physical_identity'); continue
        known = track_ids >= 0
        matching = track_ids == identity
        if matching.any():
            positives[j, :-1] = matching
            allowed[j, :-1] = known
            allowed[j, -1] = True
            reason.append('known_existing_multiple_slots' if matching.sum() > 1 else 'known_existing_single_slot')
        elif known.all():
            positives[j, -1] = True
            allowed[j] = True
            reason.append('known_no_supplied_match_not_verified_birth')
        else:
            reason.append('old_identity_coverage_unknown')
    return dict(positives=positives, allowed=allowed, reason=reason)


def balanced_set_identity_loss(logits, targets, legal, consistency_weight=.1):
    positives, allowed = targets['positives'].clone(), targets['allowed'].clone()
    if positives.shape != logits.shape or legal.shape != (logits.shape[1]-1, len(logits)):
        raise ValueError('Aligned existing-slot/NULL target sets required')
    allowed[:, :-1] &= legal.T
    positives &= allowed
    selected = positives.any(1)
    if not selected.any():
        zero = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits)).sum()*0
        return zero, dict(supervised=0, existing=0, null=0, multi_positive=0, classification=0., consistency=0.)
    active = logits[selected].masked_fill(~allowed[selected], -torch.inf)
    positive = positives[selected]
    ce = torch.logsumexp(active, dim=-1)-torch.logsumexp(active.masked_fill(~positive, -torch.inf), dim=-1)
    null = positive[:, -1]
    groups = [ce[mask].mean() for mask in (null, ~null) if mask.any()]
    classification = torch.stack(groups).mean()
    probabilities = F.softmax(active, dim=-1)
    consistency = F.relu(probabilities[:, :-1].sum(0)-1).square().sum()/max(len(active), 1)
    loss = classification+consistency_weight*consistency
    return loss, dict(supervised=int(selected.sum()), existing=int((~null).sum()), null=int(null.sum()),
                      multi_positive=int((positive[:, :-1].sum(1)>1).sum()), classification=float(classification.detach()), consistency=float(consistency.detach()))
