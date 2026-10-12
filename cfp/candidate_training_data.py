"""Training-only raw candidate/history batches; no inference imports this file."""
import json
from pathlib import Path

import numpy as np
import torch

from .provenance import sha256
from .recoverability import FEATURE_NAMES


def load_candidate_training(root, seed):
    root = Path(root)
    meta = json.loads((root / 'manifest.json').read_text())
    if meta['status'] != 'COMPLETE' or meta['partition'] != 'train' or meta['seed'] != seed:
        raise ValueError('Complete seed-matched train candidate data required')
    assets = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    features_root = assets / 'phase2' / f'causal_train_detection_features_postneck_seed{seed}'
    physical_root = assets / 'phase5/recovery_train_queries_v1'
    physical_meta = json.loads((physical_root / 'manifest.json').read_text())
    rows, banks = [], {}
    for seq, entry in meta['videos'].items():
        qp, fp, pp = root / (seq + '.npz'), features_root / (seq + '.npz'), physical_root / (seq + '.npz')
        fpmeta = json.loads(fp.with_suffix('.provenance.json').read_text())
        if sha256(qp) != entry['sha256'] or sha256(fp) != fpmeta['features_sha256'] or sha256(pp) != physical_meta['videos'][seq]['sha256']:
            raise ValueError('Query/feature/physical supervision drift')
        with np.load(fp, allow_pickle=False) as archive:
            features = archive['rgb']
        with np.load(pp, allow_pickle=False) as archive:
            labels, frames = archive['physical_labels'], archive['frame_of_proposal']
        dp = assets / 'phase0/causal_v3_train_detections' / (seq + '.npz')
        dm = json.loads(dp.with_name('manifest.json').read_text())
        if sha256(dp) != dm['caches'][seq]['sha256']:
            raise ValueError('Proposal geometry drift')
        with np.load(dp, allow_pickle=False) as archive:
            boxes, offsets = archive['boxes'], archive['offsets']
        if len(features) != len(labels) or features.shape[1] != 512:
            raise ValueError('Physical/feature alignment differs')
        banks[seq] = dict(features=features, labels=labels, frames=frames, boxes=boxes, offsets=offsets)
        with np.load(qp, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        for i, frame in enumerate(arrays['frame']):
            lo, hi = arrays['candidate_offsets'][i:i + 2]
            indices = arrays['history_indices'][i].copy()
            times = arrays['history_frames'][i].copy()
            valid = indices >= 0
            candidates = arrays['candidates'][lo:hi].copy()
            if np.any(times[valid] >= frame) or np.any(frames[indices[valid]] != times[valid]) or np.any(frames[candidates] != frame):
                raise ValueError('Candidate history/current timestamp drift')
            rows.append(dict(sequence=seq, frame=int(frame), track_id=int(arrays['track_id'][i]),
                identity=int(arrays['target'][i]), indices=indices, times=times,
                candidates=candidates, evidence=arrays['features'][lo:hi].copy(),
                edge_labels=arrays['risk_labels'][lo:hi].copy(), assigned=int(arrays['assigned'][i]),
                positive=int(arrays['positive'][i]), null_known=bool(arrays['null_known'][i]),
                correct_history=int(arrays['correct_history'][i]), unknown_history=int(arrays['unknown_history'][i])))
    meta = dict(meta, feature_manifest_sha256=sha256(features_root / 'manifest.json'),
        rgb_encoder=json.loads((features_root / 'manifest.json').read_text())['rgb_encoder'],
        physical_manifest_sha256=sha256(physical_root / 'manifest.json'))
    return rows, banks, meta


def candidate_batch(rows, banks, device, perturb_rng=None, masking=False, corruption=False, target_time='current'):
    width = max(max(len(r['candidates']) for r in rows), 1)
    b = len(rows)
    past = np.zeros((b, 8, 512), np.float32)
    times = np.zeros((b, 8), np.int64)
    valid = np.zeros((b, 8), bool)
    candidates = np.zeros((b, width, 512), np.float32)
    legal = np.zeros((b, width), bool)
    supervised = np.zeros((b, width + 1), bool)
    evidence = np.zeros((b, width, len(FEATURE_NAMES)), np.float32)
    positive = np.full(b, width, np.int64)
    target = np.zeros((b, 512), np.float32)
    target_valid = np.zeros(b, bool)
    for i, row in enumerate(rows):
        bank = banks[row['sequence']]
        indices, history_times = row['indices'].copy(), row['times'].copy()
        length = int(np.sum(indices >= 0))
        if corruption and perturb_rng is not None and length and perturb_rng.random() < .25:
            count = int(perturb_rng.integers(1, min(length, 4) + 1))
            for j in range(length - count, length):
                index = indices[j]
                lo, hi = bank['offsets'][history_times[j] - 1:history_times[j] + 1]
                other = np.flatnonzero((bank['labels'][lo:hi] >= 0) & (bank['labels'][lo:hi] != row['identity'])) + lo
                if len(other):
                    center = (bank['boxes'][index, :2] + bank['boxes'][index, 2:]) / 2
                    centers = (bank['boxes'][other, :2] + bank['boxes'][other, 2:]) / 2
                    scale = max(float(np.linalg.norm(bank['boxes'][index, 2:] - bank['boxes'][index, :2])), 1.)
                    near = other[np.linalg.norm(centers - center, axis=1) <= 2 * scale]
                    if len(near):
                        indices[j] = near[np.argmax(bank['features'][near] @ bank['features'][index])]
        if masking and perturb_rng is not None and length > 1 and perturb_rng.random() < .25:
            count = int(perturb_rng.integers(1, min(length - 1, 4) + 1))
            start = int(perturb_rng.integers(0, length - count + 1))
            keep = np.r_[np.arange(start), np.arange(start + count, length)]
            saved_indices, saved_times = indices[keep].copy(), history_times[keep].copy()
            indices.fill(-1); history_times.fill(0)
            indices[:len(keep)], history_times[:len(keep)] = saved_indices, saved_times
        present = indices >= 0
        past[i, present], times[i], valid[i] = bank['features'][indices[present]], history_times, present
        n = len(row['candidates'])
        candidates[i, :n] = bank['features'][row['candidates']]
        legal[i, :n] = True
        supervised[i, :n] = row['edge_labels'] >= 0
        supervised[i, -1] = row['null_known']
        evidence[i, :n] = row['evidence']
        if row['positive'] < n:
            positive[i] = row['positive']
            index = row['candidates'][row['positive']]
            if target_time == 'future':
                future = np.flatnonzero((bank['labels'] == row['identity']) &
                    (bank['frames'] >= row['frame'] + 5) & (bank['frames'] <= row['frame'] + 30))
                if len(future):
                    index = future[0]
            elif target_time != 'current':
                raise ValueError('Declared current/future training target required')
            target[i], target_valid[i] = bank['features'][index], True
    arrays = (past, times, valid, np.array([r['frame'] for r in rows]), candidates, legal, evidence,
              supervised, positive, target, target_valid)
    return tuple(torch.from_numpy(v).to(device) for v in arrays)

