#!/usr/bin/env python3
"""Keep every track in each actual high-stage frame, including unknown refs.

Historical teacher snapshots are warm-start conditional training data, not OOF
online evidence. Physical supervision is saved separately from observable input.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

from cfp.candidate_training_data import load_candidate_training
from cfp.joint_identity_supervision import joint_identity_targets
from cfp.predictive import LatentIdentityPredictor
from cfp.provenance import sha256, snapshot_sources, write_json
from cfp.recoverability_diagnosis import EarlyReference
from cfp.recovery_supervision import history_view


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or args.stride < 1:
        raise ValueError('New immutable full-set query output and positive stride required')
    assets = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    if shutil.disk_usage(assets).free < 40 * 2**30:
        raise OSError('40GiB reserve')
    torch.set_num_threads(2)
    teacher = assets / 'phase5/cross_train_gru_cfp_s0_c1_v1'
    teacher_meta = json.loads((teacher / 'manifest.json').read_text())
    if teacher_meta['status'] != 'COMPLETE' or teacher_meta['partition'] != 'train' or teacher_meta['seed'] != 0:
        raise ValueError('Complete historical six-train-video teacher required')
    _, banks, feature_meta = load_candidate_training(assets / 'phase6/candidate_queries_warmstart_s0_v1', 0)
    checkpoint = assets / 'phase3/predictor_v3_gru_association_anchor_seed0/epoch8.pt'
    cm = json.loads(checkpoint.with_name('manifest.json').read_text())
    if sha256(checkpoint) != cm['checkpoint_sha256']:
        raise ValueError('Frozen ordinary GRU changed')
    model = LatentIdentityPredictor().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    args.output.mkdir(parents=True)
    report = dict(status='RUNNING', seed=0, stride=args.stride, videos={},
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        teacher_root=str(teacher), teacher_manifest_sha256=sha256(teacher / 'manifest.json'),
        feature_manifest_sha256=feature_meta['feature_manifest_sha256'], checkpoint_sha256=sha256(checkpoint),
        scope='All known/unknown tracks and all actual current high-stage proposals in each sampled teacher snapshot; low-stage/initial no-old-track frames are not covered by this warm source. No row-wise known-track filtering.',
        supervision='Separate known proposal labels and strictly-past EarlyReference consensus (>=3 observations, minimum0.6); this is a noisy diagnostic identity convention, not unconditional identity truth. Unknown or nonunique physical assignments ignored. No-old-match NULL is not a verified birth or full occlusion label.',
        exposure='RGB encoder all6train exposed; ordinary GRU fit5/calibrationSN015. This dataset cannot establish encoder-level leave-video-out generalization or endogenous new tracker performance.',
        source_snapshots=snapshot_sources(args.output / 'sources', [__file__, 'cfp/joint_identity_supervision.py',
            'cfp/recoverability_diagnosis.py', 'cfp/recovery_supervision.py', 'cfp/candidate_training_data.py', 'cfp/predictive.py']))
    write_json(args.output / 'manifest.json', report)
    with torch.inference_mode():
        for video, bank in banks.items():
            trace = teacher / (video + '.decisions.jsonl.gz')
            if sha256(trace) != teacher_meta['videos'][video]['decisions_sha256']:
                raise ValueError('Teacher trace drift')
            refs, frames, track_offsets, detection_offsets, edge_offsets = EarlyReference(), [], [0], [0], [0]
            values = {k: [] for k in ('indices', 'times', 'track_id', 'track_identity', 'track_state',
                'candidates', 'detection_identity', 'confidence', 'geometry', 'appearance', 'baseline_cost')}
            reasons = Counter()
            with gzip.open(trace, 'rt') as stream:
                for line in stream:
                    row = json.loads(line)
                    identities = [refs.update(track, bank['labels']) for track in row['tracks']]
                    if row['stage'] != 'high' or (row['frame'] - 1) % args.stride:
                        continue
                    n, m = len(row['tracks']), len(row['proposals'])
                    frame = int(row['frame'])
                    indices = np.full((n, 8), -1, np.int64)
                    times = np.zeros((n, 8), np.int64)
                    for i, track in enumerate(row['tracks']):
                        for j, (t, index) in enumerate(history_view(track)):
                            if t >= frame or bank['frames'][index] != t:
                                raise ValueError('Actual history is not strictly past')
                            indices[i, j], times[i, j] = index, t
                    valid = indices >= 0
                    past = np.zeros((n, 8, 512), np.float32)
                    past[valid] = bank['features'][indices[valid]]
                    proposal = np.asarray(row['proposals'], np.int64)
                    if (bank['frames'][proposal] != frame).any():
                        raise ValueError('Current proposal timestamp drift')
                    prototype = (model(torch.from_numpy(past), torch.from_numpy(times), torch.from_numpy(valid),
                                      torch.full((n,), frame, dtype=torch.int64))[0].numpy()
                                 if n else np.empty((0, 512), np.float32))
                    features = bank['features'][proposal]
                    cosine = prototype @ features.T
                    physical = np.asarray([-1 if identity is None else identity for identity in identities], np.int64)
                    labels = bank['labels'][proposal].copy()
                    supervision = joint_identity_targets(video, torch.from_numpy(physical), torch.from_numpy(labels))
                    reasons.update(supervision['reason'])
                    state = np.asarray([[np.log1p(frame - t['start_frame']) / np.log(15001),
                                         np.log1p(frame - t['end_frame']) / np.log(15001), float(t['state'] == 2)]
                                        for t in row['tracks']], np.float32).reshape(n, 3)
                    entries = dict(indices=indices, times=times, track_id=np.asarray([t['track_id'] for t in row['tracks']], np.int64),
                        track_identity=physical, track_state=state, candidates=proposal, detection_identity=labels,
                        confidence=np.asarray(row['scores'], np.float32),
                        geometry=np.asarray(row['geometry'], np.float32).ravel(), appearance=np.clip((1 - cosine) / 2, 0, 1).ravel(),
                        baseline_cost=np.asarray(row['costs'], np.float32).ravel())
                    for name, value in entries.items():
                        values[name].append(value)
                    frames.append(frame)
                    track_offsets.append(track_offsets[-1] + n)
                    detection_offsets.append(detection_offsets[-1] + m)
                    edge_offsets.append(edge_offsets[-1] + n * m)
            output = args.output / (video + '.npz')
            arrays = {name: np.concatenate(parts, axis=0) for name, parts in values.items()}
            np.savez_compressed(output, frame=np.asarray(frames, np.int64),
                track_offsets=np.asarray(track_offsets, np.int64), detection_offsets=np.asarray(detection_offsets, np.int64),
                edge_offsets=np.asarray(edge_offsets, np.int64), **arrays)
            report['videos'][video] = dict(frames=len(frames), track_slots=track_offsets[-1], detections=detection_offsets[-1],
                unknown_track_slots=int((arrays['track_identity'] < 0).sum()), supervision_reasons=dict(reasons),
                sha256=sha256(output), trace_sha256=sha256(trace), feature_sha256=feature_meta['videos'][video]['feature_sha256'])
            write_json(args.output / 'manifest.json', report)
            print(video, report['videos'][video], flush=True)
    report['status'] = 'COMPLETE'
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
