#!/usr/bin/env python3
"""Index clean and actual accepted histories from six frozen train videos."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import shutil

import numpy as np

from cfp.provenance import sha256, snapshot_sources, write_json
from cfp.recovery_supervision import physical_bank, history_view, modal_identity


def pad(values, length=8, fill=-1):
    result = np.full(length, fill, np.int64)
    result[:len(values)] = values
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Recovery training indices immutable')
    root = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    run = json.loads((args.run / 'manifest.json').read_text())
    if run['status'] != 'COMPLETE' or run['partition'] != 'train' or run['probe']:
        raise ValueError('Complete actual six-train-video histories required; no dev fitting')
    if shutil.disk_usage(root).free < 40 * 2**30:
        raise OSError('40GiB reserve')
    args.output.mkdir(parents=True)
    record = dict(status='RUNNING', partition='train', teacher_replay_seed=run['seed'],
                  actual_replay_manifest_sha256=sha256(args.run / 'manifest.json'), videos={},
                  source_snapshots=snapshot_sources(args.output, [__file__, 'cfp/recovery_supervision.py', 'cfp/supervision.py', 'configs/cfp/phase_v_protocol_v1.json']),
                  fit_videos=['BT-001', 'BT-003', 'MSK-002', 'PF-001', 'SN-001'], calibration_videos=['SN-015'],
                  scope='GT labels supervise offline only. Reference identity is >=3 known earliest accepted observations with>=.6consensus, never immutable first token. Actual context is the exact accepted anchor/recent8. Clean context is physical same-ID strictly past proposals. No new features/images, labels not inference inputs. Shared seed0 teacher proposal indices may be reused with aligned frozen encoder seeds.')
    write_json(args.output / 'manifest.json', record)
    for seq in run['videos']:
        bank = physical_bank(seq, 'train')
        labels, frames = bank['labels'], bank['frame_of_proposal']
        feature_path = root / 'phase2' / f'causal_train_detection_features_postneck_seed{run["seed"]}' / (seq + '.npz')
        with np.load(feature_path, allow_pickle=False) as archive:
            features = archive['rgb']
        if sha256(feature_path) != json.loads(feature_path.with_suffix('.provenance.json').read_text())['features_sha256']:
            raise ValueError('Feature drift')
        labelled_by_identity = {int(identity): np.flatnonzero(labels == identity) for identity in np.unique(labels) if identity >= 0}
        first_known, references = defaultdict(dict), {}
        rows, excluded = [], Counter()
        trace = args.run / (seq + '.decisions.jsonl.gz')
        if sha256(trace) != run['videos'][seq]['decisions_sha256']:
            raise ValueError('Trace drift')
        with gzip.open(trace, 'rt') as stream:
            for line in stream:
                decision = json.loads(line)
                if decision['stage'] != 'high':
                    continue
                frame = decision['frame']
                lo, hi = bank['offsets'][frame - 1:frame + 1]
                candidates = np.flatnonzero(labels[lo:hi] >= 0) + lo
                for track in decision['tracks']:
                    history = history_view(track)
                    identifier = track['track_id']
                    for time, index in history:
                        if labels[index] >= 0 and len(first_known[identifier]) < 5:
                            first_known[identifier][index] = time
                    if identifier not in references and len(first_known[identifier]) >= 3:
                        indices = sorted(first_known[identifier], key=lambda index: first_known[identifier][index])
                        reference, _ = modal_identity(indices, labels)
                        if reference is not None:
                            references[identifier] = reference
                    if (frame - 1) % 5:
                        continue
                    if identifier not in references:
                        excluded['unknown_early_multi_observation_reference'] += 1
                        continue
                    target = references[identifier]
                    positive = np.flatnonzero(labels[candidates] == target)
                    if not len(positive) or len(candidates) < 2:
                        excluded['target_absent_or_no_physical_negative'] += 1
                        continue
                    if len(positive) != 1:
                        raise ValueError('Nonunique physical positive')
                    if not history or any(time >= frame or frames[index] != time for time, index in history):
                        raise ValueError('Accepted context contains a current/future frame')
                    target_indices = labelled_by_identity[target]
                    prior = target_indices[frames[target_indices] < frame]
                    if not len(prior):
                        excluded['no_clean_past_context'] += 1
                        continue
                    # A clean comparator has the same8token budget and a first
                    # past cue plus recent7, with no accepted-history contamination.
                    clean = prior[-8:]
                    if prior[0] != clean[0]:
                        clean = np.r_[prior[0], clean[-7:]]
                    actual = np.array([index for _, index in history], np.int64)
                    def neighbors(indices):
                        result = []
                        for index in indices:
                            past_frame = frames[index]
                            start, end = bank['offsets'][past_frame - 1:past_frame + 1]
                            other = np.flatnonzero((labels[start:end] >= 0) & (labels[start:end] != target)) + start
                            if not len(other):
                                result.append(-1)
                                continue
                            centers = (bank['boxes'][other, :2] + bank['boxes'][other, 2:]) / 2
                            center = (bank['boxes'][index, :2] + bank['boxes'][index, 2:]) / 2
                            scale = max(float(np.linalg.norm(bank['boxes'][index, 2:] - bank['boxes'][index, :2])), 1.)
                            near = other[np.linalg.norm(centers - center, axis=1) / scale <= 2.]
                            if not len(near):
                                result.append(-1)
                            else:
                                result.append(int(near[np.argmax(features[near] @ features[index])]))
                        return pad(result)
                    rows.append(dict(frame=frame, track_id=identifier, identity=target, positive=int(positive[0]), candidates=candidates.copy(),
                                     actual_indices=pad(actual), actual_frames=pad(frames[actual], fill=0),
                                     clean_indices=pad(clean), clean_frames=pad(frames[clean], fill=0),
                                     actual_neighbors=neighbors(actual), clean_neighbors=neighbors(clean),
                                     actual_token_labels=pad(labels[actual]), clean_token_labels=pad(labels[clean]),
                                     first_wrong=bool(labels[actual[0]] >= 0 and labels[actual[0]] != target),
                                     retained_correct_tokens=int(np.sum(labels[actual] == target))))
        if not rows:
            raise ValueError('No supervised recovery queries: ' + seq)
        arrays = {key: np.array([row[key] for row in rows]) for key in ('frame', 'track_id', 'identity', 'positive', 'actual_indices', 'actual_frames', 'clean_indices', 'clean_frames', 'actual_neighbors', 'clean_neighbors', 'actual_token_labels', 'clean_token_labels', 'first_wrong', 'retained_correct_tokens')}
        arrays['candidate_offsets'] = np.r_[0, np.cumsum([len(row['candidates']) for row in rows])]
        arrays['candidate_indices'] = np.concatenate([row['candidates'] for row in rows])
        arrays['physical_labels'] = labels
        arrays['frame_of_proposal'] = frames
        path = args.output / (seq + '.npz')
        np.savez_compressed(path, **arrays)
        known_tokens = np.concatenate([row['actual_token_labels'][row['actual_token_labels'] >= 0] == row['identity'] for row in rows])
        record['videos'][seq] = dict(queries=len(rows), sha256=sha256(path), features_sha256=sha256(feature_path),
                                    trace_sha256=sha256(trace), detection_sha256=bank['detection_sha256'],
                                    first_wrong_fraction=float(np.mean(arrays['first_wrong'])), known_token_wrong_fraction=float(np.mean(~known_tokens)),
                                    no_retained_correct_token_fraction=float(np.mean(arrays['retained_correct_tokens'] == 0)), exclusions=dict(excluded))
        write_json(args.output / 'manifest.json', record)
        print(seq, record['videos'][seq], flush=True)
    record['status'] = 'COMPLETE'
    write_json(args.output / 'manifest.json', record)


if __name__ == '__main__':
    main()
