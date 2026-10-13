"""Training-only utility labels for actual retained tokens; never imported online."""
import numpy as np
import torch
from torch.nn import functional as F


def known_cue_targets(rows, banks, visible_times, visible_mask):
    times, valid = visible_times.detach().cpu().numpy(), visible_mask.detach().cpu().numpy()
    labels = np.full(times.shape, -1., np.float32)
    for i, row in enumerate(rows):
        if row['identity'] < 0:
            continue
        original = {int(time): int(index) for time, index in zip(row['times'], row['indices']) if index >= 0}
        physical = banks[row['sequence']]['labels']
        for j in np.flatnonzero(valid[i]):
            time = int(times[i, j])
            if time >= row['frame'] or time not in original:
                raise ValueError('Utility supervision requires an actual visible, strictly past token')
            identity = physical[original[time]]
            if identity >= 0:
                labels[i, j] = float(identity == row['identity'])
    return torch.tensor(labels, device=visible_times.device)


def cue_utility_loss(logits, labels, current_legal):
    known = (labels[:, None] >= 0) & current_legal[..., None]
    target = labels[:, None].expand_as(logits)
    positive, negative = known & (target == 1), known & (target == 0)
    losses = []
    for mask in [positive, negative]:
        if mask.any():
            losses.append(F.binary_cross_entropy_with_logits(logits[mask], target[mask]))
    loss = torch.stack(losses).mean() if losses else logits.sum() * 0
    return loss, dict(known_correct=int(positive.sum()), known_wrong=int(negative.sum()))
