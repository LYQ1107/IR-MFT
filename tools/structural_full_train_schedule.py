"""All Train frames, paired within video for the existing counterfactual loss.

One logical Train traversal uses 24 macro passes of 288 batch4 updates. The
macro pass is an optimizer/checkpoint bookkeeping unit, not a full data epoch.
"""
import hashlib
import json

import numpy as np


def make_full_train_schedule(lengths, logical_epochs=3, macro_passes=24,
                             batches_per_macro=288, seed=0):
    if seed != 0 or not lengths or logical_epochs < 1 or macro_passes < 1 or batches_per_macro < 1:
        raise ValueError('Require fixed seed0 and positive registered dimensions')
    sequences = sorted(lengths)
    if any(int(lengths[s]) != lengths[s] or lengths[s] < 2 for s in sequences):
        raise ValueError('Each Train video needs at least two frames')
    slots = macro_passes * batches_per_macro * 2  # Two within-video pairs per batch.
    minimum_pairs = sum((int(lengths[s]) + 1) // 2 for s in sequences)
    if minimum_pairs > slots:
        raise ValueError('Budget cannot visit every Train frame')
    traversals = []
    summaries = []
    for epoch in range(logical_epochs):
        rng = np.random.default_rng(20261123 + epoch)
        rows = []
        for index, sequence in enumerate(sequences):
            frames = rng.permutation(np.arange(1, int(lengths[sequence]) + 1, dtype=np.int32))
            if len(frames) % 2:
                frames = np.r_[frames, frames[:1]]
            pairs = np.empty((len(frames) // 2, 2, 2), np.int32)
            pairs[:, :, 0] = index
            pairs[:, :, 1] = frames.reshape(-1, 2)
            rows.append(pairs)
        pairs = np.concatenate(rows)
        extra = slots - len(pairs)
        if extra:
            padding = pairs[rng.choice(len(pairs), size=extra, replace=extra > len(pairs))]
            pairs = np.concatenate([pairs, padding])
        pairs = pairs[rng.permutation(len(pairs))]
        table = pairs.reshape(macro_passes, batches_per_macro, 4, 2)
        # The two members of each counterfactual pair are distinct frames of
        # one video. Both pairs in a batch may belong to the same video.
        paired = table.reshape(-1, 2, 2)
        assert np.all(paired[:, 0, 0] == paired[:, 1, 0])
        assert np.all(paired[:, 0, 1] != paired[:, 1, 1])
        coverage = {}
        for index, sequence in enumerate(sequences):
            frames = table[:, :, :, 1][table[:, :, :, 0] == index]
            np.testing.assert_array_equal(np.unique(frames), np.arange(1, lengths[sequence] + 1))
            coverage[sequence] = dict(exposures=len(frames), unique_frames=len(np.unique(frames)))
        summaries.append(dict(logical_Train_epoch=epoch + 1, unique_Train_frames=sum(lengths.values()),
            sample_exposures=int(table.size // 2), coverage=coverage,
            repeated_exposures=int(table.size // 2 - sum(lengths.values())),
            same_video_for_both_pairs_fraction=float(np.mean(table[:, :, 0, 0] == table[:, :, 2, 0]))))
        traversals.append(table)
    result = np.concatenate(traversals, axis=0)
    return sequences, result, summaries


def table_digest(sequences, table):
    digest = hashlib.sha256(json.dumps(sequences, separators=(',', ':')).encode())
    digest.update(np.asarray(table, dtype='<i4').tobytes())
    return digest.hexdigest()
