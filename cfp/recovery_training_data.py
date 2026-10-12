"""Training-only index loading and realistic, current/past-preserving perturbations."""
import json
from pathlib import Path

import numpy as np
import torch

from .provenance import sha256


def load_recovery_data(query_root, seed):
    query_root = Path(query_root)
    manifest = json.loads((query_root / 'manifest.json').read_text())
    if manifest['status'] != 'COMPLETE' or manifest['partition'] != 'train':
        raise ValueError('Complete training-only recovery indices required')
    feature_root = Path('/data3/liuyeqiang/IR-MFT_assets/cfp/phase2') / f'causal_train_detection_features_postneck_seed{seed}'
    feature_manifest = json.loads((feature_root / 'manifest.json').read_text())
    if feature_manifest['status'] != 'COMPLETE' or feature_manifest['embedding_stage'] != 'post_neck' or feature_manifest['rgb_encoder']['seed'] != seed:
        raise ValueError('Matched frozen post-neck feature coordinates required')
    rows, features, labels = [], {}, {}
    for seq, entry in manifest['videos'].items():
        qp, fp = query_root / (seq + '.npz'), feature_root / (seq + '.npz')
        if sha256(qp) != entry['sha256'] or sha256(fp) != json.loads(fp.with_suffix('.provenance.json').read_text())['features_sha256']:
            raise ValueError('Query/feature bytes changed')
        with np.load(fp, allow_pickle=False) as archive:
            features[seq] = archive['rgb']
        with np.load(qp, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        labels[seq] = data['physical_labels']
        if len(features[seq]) != len(labels[seq]) or features[seq].shape[1] != 512:
            raise ValueError('Latent/proposal alignment drift')
        for i, frame in enumerate(data['frame']):
            lo, hi = data['candidate_offsets'][i:i + 2]
            row = dict(sequence=seq, frame=int(frame), identity=int(data['identity'][i]), positive=int(data['positive'][i]), candidates=data['candidate_indices'][lo:hi])
            for view in ('actual', 'clean'):
                for field in ('indices', 'frames', 'neighbors'):
                    row[view + '_' + field] = data[view + '_' + field][i]
                indices, times = row[view + '_indices'], row[view + '_frames']
                valid = indices >= 0
                if not np.array_equal(valid, np.arange(8) < valid.sum()) or not valid.any() or np.any(times[valid] >= frame) or np.any(times[valid] <= 0) or np.any(np.diff(times[valid]) <= 0):
                    raise ValueError('Noncausal or malformed training context')
                if np.any(data['frame_of_proposal'][indices[valid]] != times[valid]):
                    raise ValueError('History index timestamp mismatch')
                donors = row[view + '_neighbors']
                chosen = donors >= 0
                if np.any(chosen & ~valid) or np.any(data['frame_of_proposal'][donors[chosen]] != times[chosen]):
                    raise ValueError('Structured neighbor comes from a different/future frame')
            if np.any(data['frame_of_proposal'][row['candidates']] != frame) or labels[seq][row['candidates'][row['positive']]] != row['identity']:
                raise ValueError('Invalid current clean teacher/candidate label')
            rows.append(row)
    manifest = dict(manifest, feature_root=str(feature_root), feature_manifest_sha256=sha256(feature_root / 'manifest.json'), rgb_encoder=feature_manifest['rgb_encoder'])
    return rows, features, labels, manifest


def sampling_groups(rows, videos):
    result = []
    for video in videos:
        identities = sorted({row['identity'] for row in rows if row['sequence'] == video})
        if not identities:
            raise ValueError('Missing fit video queries')
        result.append([np.array([i for i, row in enumerate(rows) if row['sequence'] == video and row['identity'] == identity]) for identity in identities])
    return result


def sample_indices(groups, rng, size):
    values = []
    for _ in range(size):
        identities = groups[int(rng.integers(len(groups)))]
        positions = identities[int(rng.integers(len(identities)))]
        values.append(int(rng.choice(positions)))
    return values


def make_batch(rows, features, labels, device, rng=None, view='actual', perturb=False, masking=False, view_rng=None):
    width = max(len(row['candidates']) for row in rows)
    past = np.zeros((len(rows), 8, 512), np.float32)
    candidates = np.zeros((len(rows), width, 512), np.float32)
    times = np.zeros((len(rows), 8), np.int64)
    valid = np.zeros((len(rows), 8), bool)
    candidate_valid = np.zeros((len(rows), width), bool)
    correct = np.zeros((len(rows), 8), bool)
    known = np.zeros((len(rows), 8), bool)
    for i, row in enumerate(rows):
        selected_view = view
        if view == 'mixed':
            selected_view = 'clean' if (view_rng or rng).random() < .25 else 'actual'
        indices = row[selected_view + '_indices'].copy()
        history_times = row[selected_view + '_frames'].copy()
        length = int(np.sum(indices >= 0))
        if perturb and rng is not None and rng.random() < .25:
            donors = row[selected_view + '_neighbors']
            if rng.random() < 1/3:
                positions = [0]  # Incorrect first anchor, not privileged truth.
            else:
                count = int(rng.integers(1, min(length, 4) + 1))
                positions = range(length - count, length)
            for position in positions:
                if donors[position] >= 0:
                    indices[position] = donors[position]
        if masking and rng is not None and length > 1 and rng.random() < .25:
            # Contiguous block missing, strictly pack only remaining past tokens.
            count = int(rng.integers(1, min(length - 1, 4) + 1))
            start = int(rng.integers(0, length - count + 1))
            keep = np.r_[np.arange(start), np.arange(start + count, length)]
            reduced, reduced_times = indices[keep].copy(), history_times[keep].copy()
            indices.fill(-1)
            history_times.fill(0)
            indices[:len(reduced)] = reduced
            history_times[:len(reduced)] = reduced_times
        present = indices >= 0
        n = len(row['candidates'])
        past[i, present] = features[row['sequence']][indices[present]]
        times[i] = history_times
        valid[i] = present
        physical = labels[row['sequence']][indices[present]]
        correct[i, present] = physical == row['identity']
        known[i, present] = physical >= 0
        candidates[i, :n] = features[row['sequence']][row['candidates']]
        candidate_valid[i, :n] = True
    values = (past, times, valid, np.array([row['frame'] for row in rows]), candidates, candidate_valid,
              np.array([row['positive'] for row in rows]), correct, known)
    return tuple(torch.from_numpy(value).to(device) for value in values)
