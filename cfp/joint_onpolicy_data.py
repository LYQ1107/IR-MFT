"""Frozen-coordinate current/past native inputs; physical labels kept separate."""
import json
from pathlib import Path
import numpy as np
import torch
from .association import optional_feature
from .native_local_identity import RecentIdentityControl
from .joint_set_supervision import joint_identity_target_sets
from .provenance import sha256


def load_native_source(root, detections, video):
    root, detections = Path(root), Path(detections)
    m = json.loads((root/'manifest.json').read_text())
    dp = detections/(video+'.npz'); dm = json.loads((detections/'manifest.json').read_text())
    if (m['status'] != 'COMPLETE_NATIVE_CURRENT_DINO_FEATURES' or m['video'] != video
            or m['encoder']['trained_encoder'] is not None or m['encoder']['coordinate_kind'] != 'frozen_DINO_CLS384'
            or sha256(dp) != m['detection_sha256'] or sha256(dp) != dm['caches'][video]['sha256']
            or m['detection_manifest_sha256'] != sha256(detections/'manifest.json')):
        raise ValueError('Identical complete frozen-coordinate/current-proposal source required')
    with np.load(dp, allow_pickle=False) as bank:
        arrays = {k: bank[k].copy() for k in ['frame_ids', 'offsets', 'boxes', 'scores']}
    bank = {}
    for name in ['embedding', 'valid']:
        p = root/(name+'.npy')
        if sha256(p) != m['artifacts'][name]['sha256']:
            raise ValueError('Native source feature drift')
        bank[name] = np.load(p, mmap_mode='r')
    for name in ['frame_ids', 'offsets']:
        p = root/(name+'.npy')
        if sha256(p) != m[name+'_sha256'] or not np.array_equal(arrays[name], np.load(p)):
            raise ValueError('Native frame/proposal alignment drift')
    if not np.array_equal(arrays['frame_ids'], np.arange(1, len(arrays['frame_ids'])+1)) or m['completed_frames'] != len(arrays['frame_ids']) or bank['embedding'].shape != (len(arrays['boxes']), 384):
        raise ValueError('All native frames and proposals required')
    return dict(arrays, **bank, frame_of_proposal=np.repeat(arrays['frame_ids'], np.diff(arrays['offsets'])), metadata=m,
                manifest_sha256=sha256(root/'manifest.json'), detection_sha256=sha256(dp))


def frame_inputs(indices, times, current, state, confidence, geometry, bank, frame, codes, length=4):
    valid = indices >= 0
    past = np.zeros((len(indices), length, 384), np.float32)
    for row, col in zip(*np.nonzero(valid)):
        index = indices[row, col]
        if bank['frame_of_proposal'][index] != times[row, col] or times[row, col] >= frame:
            raise ValueError('Actual strictly-past history indices required')
        feature = optional_feature(bank['embedding'][index])
        if feature is None:
            raise ValueError('Unavailable observation cannot be accepted identity history')
        past[row, col] = feature
    current_values = np.zeros((len(current), 384), np.float32)
    for j, index in enumerate(current):
        if bank['frame_of_proposal'][index] != frame:
            raise ValueError('Only actual current proposal inputs permitted')
        feature = optional_feature(bank['embedding'][index])
        if feature is not None:
            current_values[j] = feature
    inputs = dict(history=torch.from_numpy(past), times=torch.from_numpy(times.copy()), valid=torch.from_numpy(valid),
                  frame=torch.full((len(indices),), int(frame), dtype=torch.int64), detections=torch.from_numpy(current_values),
                  detection_valid=torch.from_numpy(np.asarray(bank['valid'][current], bool)), geometry=torch.from_numpy(geometry.copy()),
                  track_state=torch.from_numpy(state.copy()), confidence=torch.from_numpy(confidence.copy()), codes=torch.from_numpy(codes))
    with torch.no_grad():
        prototype, _ = RecentIdentityControl('fifo', length, 1500)(inputs['history'], inputs['times'], inputs['valid'], inputs['frame'])
        inputs['appearance'] = ((1-prototype@inputs['detections'].T)/2).clamp(0, 1)
    return inputs


def load_query_dataset(root):
    root = Path(root); manifest = json.loads((root/'manifest.json').read_text())
    if manifest['status'] != 'COMPLETE_FIT_ONLY_NATIVE_DINO_SET_QUERIES':
        raise ValueError('Complete actual fit-only query corpus required')
    rows, banks = [], {}
    config = manifest['config']
    for video in config['fit_videos']:
        p = root/(video+'.npz')
        if sha256(p) != manifest['videos'][video]['query_sha256']:
            raise ValueError('Native query corpus changed')
        bank = load_native_source(Path(manifest['root'])/f'native_dino_frozen_{video}_k4_v1', manifest['detections'], video)
        if bank['manifest_sha256'] != manifest['videos'][video]['feature_manifest_sha256']:
            raise ValueError('Exact training coordinates changed')
        with np.load(p, allow_pickle=False) as a:
            bank['queries'] = {k: a[k].copy() for k in a.files}
        banks[video] = bank
        q = bank['queries']
        rows.extend(dict(video=video, position=i) for i in range(len(q['frame'])) if q['supervisable'][i])
    return rows, banks, manifest


def query_batch(row, banks, codes_rng, capacity):
    bank = banks[row['video']]; q = bank['queries']; i = row['position']
    lo, hi = q['track_offsets'][i:i+2]; dl, dh = q['detection_offsets'][i:i+2]; el, eh = q['edge_offsets'][i:i+2]
    n, m = int(hi-lo), int(dh-dl)
    if n > capacity:
        raise ValueError('No capacity truncation or silent fallback')
    inputs = frame_inputs(q['indices'][lo:hi], q['times'][lo:hi], q['current'][dl:dh], q['state'][lo:hi],
                          q['confidence'][dl:dh], q['geometry'][el:eh].reshape(n, m), bank, q['frame'][i], codes_rng.permutation(capacity)[:n].astype(np.int64))
    labels = torch.from_numpy(q['current_identity'][dl:dh].copy())
    labels[~inputs['detection_valid']] = -1
    targets = joint_identity_target_sets(torch.from_numpy(q['track_identity'][lo:hi].copy()), labels)
    return inputs, targets
