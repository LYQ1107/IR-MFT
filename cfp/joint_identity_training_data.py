"""Training-only full-frame snapshots; keeps physical supervision separate."""
import json
from pathlib import Path

import numpy as np
import torch

from .joint_identity_supervision import joint_identity_targets
from .provenance import sha256


def load_joint_training(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['status'] != 'COMPLETE' or manifest['seed'] != 0:
        raise ValueError('Complete seed0 full-set warm data required')
    assets = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    feature_root = assets / 'phase2/causal_train_detection_features_postneck_seed0'
    if sha256(feature_root / 'manifest.json') != manifest['feature_manifest_sha256']:
        raise ValueError('Frozen full-set feature manifest drift')
    banks, rows = {}, []
    for video, entry in manifest['videos'].items():
        qp, fp = root / (video + '.npz'), feature_root / (video + '.npz')
        if sha256(qp) != entry['sha256'] or sha256(fp) != entry['feature_sha256']:
            raise ValueError('Immutable query/feature drift')
        with np.load(qp, allow_pickle=False) as source:
            queries = {k: source[k] for k in source.files}
        with np.load(fp, allow_pickle=False) as source:
            features = source['rgb'].copy()
        frames = queries['frame']
        to, do, eo = (queries[k] for k in ('track_offsets', 'detection_offsets', 'edge_offsets'))
        if (any(len(v) != len(frames) + 1 or v[0] != 0 or np.any(np.diff(v) < 0) for v in (to, do, eo))
                or not np.array_equal(np.diff(eo), np.diff(to) * np.diff(do))
                or to[-1] != len(queries['track_identity']) or do[-1] != len(queries['candidates'])
                or eo[-1] != len(queries['geometry']) or eo[-1] != len(queries['appearance'])):
            raise ValueError('Whole-frame offsets are not aligned')
        banks[video] = dict(queries=queries, features=features)
        rows.extend(dict(video=video, position=i) for i in range(len(frames)))
    return rows, banks, manifest


def joint_frame_batch(record, banks, device, dictionary_rng, capacity=128):
    video, i = record['video'], record['position']
    bank = banks[video]
    q = bank['queries']
    tl, th = q['track_offsets'][i:i + 2]
    dl, dh = q['detection_offsets'][i:i + 2]
    el, eh = q['edge_offsets'][i:i + 2]
    n, m = int(th - tl), int(dh - dl)
    if n > capacity:
        raise ValueError('Track set exceeds declared capacity; never silently truncate')
    indices, times = q['indices'][tl:th], q['times'][tl:th]
    valid = indices >= 0
    if ((times >= q['frame'][i]) & valid).any():
        raise ValueError('Current/future observations cannot enter track memory')
    past = np.zeros((n, 8, 512), np.float32)
    past[valid] = bank['features'][indices[valid]]
    arrays = dict(history=past, times=times.copy(), valid=valid,
        frame=np.full(n, q['frame'][i], np.int64), detections=bank['features'][q['candidates'][dl:dh]],
        detection_valid=np.ones(m, bool), geometry=q['geometry'][el:eh].reshape(n, m).copy(),
        appearance=q['appearance'][el:eh].reshape(n, m).copy(), track_state=q['track_state'][tl:th].copy(),
        confidence=q['confidence'][dl:dh].copy(), codes=dictionary_rng.permutation(capacity)[:n].astype(np.int64))
    inputs = {name: torch.from_numpy(value).to(device) for name, value in arrays.items()}
    # Labels stay outside the model's observable input dictionary.
    supervision = joint_identity_targets(video,
        torch.from_numpy(q['track_identity'][tl:th].copy()).to(device),
        torch.from_numpy(q['detection_identity'][dl:dh].copy()).to(device))
    return inputs, supervision
