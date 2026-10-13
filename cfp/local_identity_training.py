"""Offline known-physical supervised batches on stopped ImageNet features."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .provenance import sha256
from .reid import temporal_batch_hard


def load_local_features(root, count=8):
    root = Path(root)
    meta = json.loads((root / 'manifest.json').read_text())
    if meta['status'] != 'COMPLETE_SPARSE_TWO_CROP_FEATURES' or count not in (4, 8):
        raise ValueError('Complete registered ImageNet crop feature source required')
    banks = {}
    for video, entry in meta['videos'].items():
        directory = root / video
        if sha256(directory / 'records.json') != entry['records_sha256']:
            raise ValueError('Physical crop record drift')
        arrays = {}
        # Never consume the fish-trained source columns for a heldout adapter.
        for name in ('imagenet_global', 'imagenet_local' + str(count), 'visibility' + str(count), 'valid'):
            path = directory / (name + '.npy')
            if sha256(path) != entry['artifacts'][name]['sha256']:
                raise ValueError('Stopped ImageNet feature drift')
            arrays[name] = np.load(path, mmap_mode='r')
        records = json.loads((directory / 'records.json').read_text())
        banks[video] = dict(records=records, arrays=arrays)
    return banks, meta


class SameFrameTemporalSampler:
    def __init__(self, banks, fit_videos, seed, identities_per_frame=8, observations_per_identity=4, minimum_gap=25):
        self.banks, self.videos = banks, list(fit_videos)
        self.p, self.k, self.gap = identities_per_frame, observations_per_identity, minimum_gap
        self.rng = np.random.default_rng(seed)
        self.labels, self.by_identity, self.by_frame, self.frames = {}, {}, {}, {}
        keys = []
        for video in self.videos:
            records = banks[video]['records']
            valid = banks[video]['arrays']['valid']
            by_identity, by_frame = defaultdict(list), defaultdict(list)
            for i, row in enumerate(records):
                if row['identity'] >= 0 and valid[i]:
                    by_identity[row['identity']].append(i)
                    if row['view'] == 'gt':
                        by_frame[row['frame']].append(i)
            self.by_identity[video], self.by_frame[video] = by_identity, by_frame
            self.frames[video] = [frame for frame, rows in by_frame.items() if len(rows) >= 2]
            if not self.frames[video]:
                raise ValueError('Each fit video needs real same-frame different-fish context')
            keys.extend((video, identity) for identity in by_identity)
        self.labels = {key: i for i, key in enumerate(sorted(keys))}

    def sample(self):
        video = self.videos[int(self.rng.integers(len(self.videos)))]
        frame = self.frames[video][int(self.rng.integers(len(self.frames[video])))]
        anchors = self.by_frame[video][frame]
        chosen = self.rng.choice(anchors, size=min(self.p, len(anchors)), replace=False)
        observations = []
        for anchor in chosen:
            row = self.banks[video]['records'][anchor]
            identity = row['identity']
            pool = [i for i in self.by_identity[video][identity]
                    if abs(self.banks[video]['records'][i]['frame'] - frame) >= self.gap]
            # The first observation for every identity shares one real frame.
            # For short histories, retain anchor without inventing far pairs;
            # loss excludes anchors lacking a valid temporal positive.
            extra = self.rng.choice(pool or [int(anchor)], self.k - 1, replace=len(pool) < self.k - 1)
            observations.extend((video, int(i)) for i in [anchor, *extra])
        return observations


def local_batch(observations, banks, label_map, video_map, device, count=8):
    global_rows, local_rows, visibility_rows, labels, frames, videos = [], [], [], [], [], []
    for video, i in observations:
        bank, row = banks[video]['arrays'], banks[video]['records'][i]
        global_rows.append(bank['imagenet_global'][i])
        local_rows.append(bank['imagenet_local' + str(count)][i])
        visibility_rows.append(bank['visibility' + str(count)][i])
        labels.append(label_map[(video, row['identity'])])
        frames.append(row['frame'])
        videos.append(video_map[video])
    arrays = [np.asarray(global_rows, np.float32), np.asarray(local_rows, np.float32), np.asarray(visibility_rows, np.float32),
              np.asarray(labels, np.int64), np.asarray(frames, np.int64), np.asarray(videos, np.int64)]
    return tuple(torch.from_numpy(a).to(device) for a in arrays)


def local_identity_objective(output, labels, frames, videos, minimum_gap=25, triplet_weight=1., hard_frame_weight=.2):
    embedding = output['embedding']
    ce = F.cross_entropy(output['logits'], labels, label_smoothing=.1)
    temporal, coverage = temporal_batch_hard(embedding, labels, frames, videos, minimum_gap=minimum_gap)
    distance = torch.cdist(embedding.float(), embedding.float())
    positive = (labels[:, None] == labels[None]) & ((frames[:, None] - frames[None]).abs() >= minimum_gap)
    negative = (labels[:, None] != labels[None]) & (frames[:, None] == frames[None]) & (videos[:, None] == videos[None])
    valid = positive.any(1) & negative.any(1)
    hard = embedding.sum() * 0
    if valid.any():
        hardest_positive = distance.masked_fill(~positive, -torch.inf).max(1).values
        hardest_negative = distance.masked_fill(~negative, torch.inf).min(1).values
        hard = F.relu(.3 + hardest_positive[valid] - hardest_negative[valid]).mean()
    return ce + triplet_weight * temporal + hard_frame_weight * hard, dict(ce=float(ce.detach()),
        temporal=float(temporal.detach()), same_frame=float(hard.detach()), temporal_coverage=coverage,
        same_frame_anchors=int(valid.sum()))
