"""Training-only masked/current-or-future clean targets in aligned frozen banks."""
from pathlib import Path

import numpy as np
import torch

from .recovery_training_data import make_batch


def teacher_index_maps(query_root, labels):
    maps = {}
    for seq, physical in labels.items():
        with np.load(Path(query_root) / (seq + '.npz'), allow_pickle=False) as archive:
            frames = archive['frame_of_proposal']
        identity_frame, identity_indices = {}, {}
        for identity in np.unique(physical):
            if identity < 0:
                continue
            indices = np.flatnonzero(physical == identity)
            identity_indices[int(identity)] = (frames[indices], indices)
            for index in indices:
                key = (int(identity), int(frames[index]))
                if key in identity_frame:
                    raise ValueError('Nonunique physical teacher target')
                identity_frame[key] = int(index)
        maps[seq] = (identity_frame, identity_indices)
    return maps


def make_jepa_batch(rows, features, labels, maps, device, rng, view='mixed', masking=False, corruption=False, future_gap=0, view_rng=None, mask_rng=None):
    base = make_batch(rows, features, labels, device, rng, view=view, perturb=corruption, masking=False, view_rng=view_rng)
    mask_rng = mask_rng or rng
    past, times, valid, query, candidates, candidate_valid, positive, correct, known = base
    context_mask = np.zeros(tuple(valid.shape), bool)
    target = np.zeros((len(rows), 512), np.float32)
    masked_target = np.zeros((len(rows), 8, 512), np.float32)
    masked_valid = np.zeros((len(rows), 8), bool)
    cpu_times = times.cpu().numpy()
    cpu_valid = valid.cpu().numpy()
    for i, row in enumerate(rows):
        seq, identity, frame = row['sequence'], row['identity'], row['frame']
        mapping, by_identity = maps[seq]
        index = int(row['candidates'][row['positive']])
        if future_gap:
            available_frames, indices = by_identity[identity]
            later = np.flatnonzero((available_frames >= frame + future_gap) & (available_frames <= frame + 30))
            if later.size:
                index = int(indices[later[0]])
        target[i] = features[seq][index]
        for step, time in enumerate(cpu_times[i]):
            if cpu_valid[i, step]:
                if time >= frame:
                    raise ValueError('Future context is never allowed even during JEPA training')
                clean_index = mapping.get((identity, int(time)))
                if clean_index is not None:
                    masked_target[i, step] = features[seq][clean_index]
                    masked_valid[i, step] = True
        length = int(cpu_valid[i].sum())
        if masking and length > 1:
            count = int(mask_rng.integers(1, min(4, length - 1) + 1))
            start = int(mask_rng.integers(0, length - count + 1))
            context_mask[i, start:start + count] = True
    extras = tuple(torch.from_numpy(value).to(device) for value in (target, masked_target, masked_valid, context_mask))
    return base + extras
