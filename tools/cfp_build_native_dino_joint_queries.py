#!/usr/bin/env python3
"""Actual four-fit-video DINO/FIFO teacher; labels loaded after output parity."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import numpy as np
import torch
from cfp.joint_onpolicy_data import load_native_source
from cfp.joint_set_supervision import consistent_past_identity, joint_identity_target_sets
from cfp.lifecycle_crossover import DecisionRecorder
from cfp.native_local_identity import RecentIdentityControl
from cfp.predictive_association import PredictiveAssociationTracker
from cfp.provenance import sha256, snapshot_sources, write_json
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker import matching


def main():
    ap = argparse.ArgumentParser()
    for name in ['root', 'config', 'detections', 'output']:
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args(); c = json.loads(a.config.read_text())
    if (a.output.exists() or a.output.resolve().is_relative_to(Path.cwd().resolve()) or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or c['state'] != 'FROZEN' or len(c['fit_videos']) != 4 or set(c['fit_videos']) & set(c['outer_videos']+c['inner_videos'])):
        raise ValueError('New CPU-only outsideGit four-fit-video corpus required')
    if shutil.disk_usage(a.root).free < 42*2**30:
        raise OSError('Reserve with parallel training margin')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    a.output.mkdir()
    sources = [__file__, str(a.config), 'cfp/joint_onpolicy_data.py', 'cfp/joint_set_supervision.py', 'cfp/association.py',
               'cfp/predictive_association.py', 'cfp/predictive.py', 'cfp/native_local_identity.py', 'cfp/lifecycle_crossover.py']
    report = dict(status='RUNNING_GT_FREE_TEACHER_OBSERVATION', pid=os.getpid(), root=str(a.root), detections=str(a.detections),
                  config=c, config_sha256=sha256(a.config), videos={},
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  sources=snapshot_sources(a.output/'sources', sources),
                  scope='All native frames/fixed V3/DINO384/FIFOteacher of fourfitting videos only. Actual recent4/maxage1500 accepted memories. Independent physical labels enter only after ALLfour frozenreference predictions byte-equal; no inner/outer labels/frames loaded. Source teacher histories, not learned head endogenous trajectories. Frozen foundation/private pretrain and inherited detector exposure remain explicit.')
    write_json(a.output/'manifest.json', report)
    for video in c['fit_videos']:
        bank = load_native_source(a.root/f'native_dino_frozen_{video}_k4_v1', a.detections, video)
        reference = a.root/f'native_identity_frozen_{video}_fifo_s0_v1'
        rm = json.loads((reference/'manifest.json').read_text())
        if rm['status'] != 'COMPLETE_NATIVE_CURRENT_IDENTITY_REPLAY' or rm['feature_manifest_sha256'] != bank['manifest_sha256'] or rm['memory'] != 'fifo':
            raise ValueError('Exact same-coordinate full frozenFIFO reference required')
        folder = a.output/video; folder.mkdir()
        BaseTrack._count = 0
        tracker = PredictiveAssociationTracker(SimpleNamespace(track_thresh=c['track_thresh'], track_buffer=c['track_buffer'],
            match_thresh=c['match_thresh'], mot20=c['mot20']), frame_rate=c['frame_rate'],
            predictor=RecentIdentityControl('fifo', c['history_length'], c['maximum_history_age_frames']),
            memory='predictor', history_view='recent', appearance_weight=c['appearance_weight'], proximity_iou=c['proximity_iou'], ema=.9)
        recorder = DecisionRecorder(folder/'decisions.jsonl.gz', tracker)
        prediction = folder/(video+'.txt')
        try:
            with prediction.open('w') as stream:
                for i, frame in enumerate(bank['frame_ids']):
                    lo, hi = bank['offsets'][i:i+2]
                    with recorder.capture([matching]):
                        output = tracker.update(np.c_[bank['boxes'][lo:hi].copy(), bank['scores'][lo:hi]], (1, 1), (1, 1),
                                                np.array(bank['embedding'][lo:hi], copy=True), proposal_indices=np.arange(lo, hi))
                    for track in output:
                        if np.prod(track.tlwh[2:]) <= 100:
                            continue
                        index = track.current_proposal_index
                        if track.frame_id != frame or not lo <= index < hi:
                            raise ValueError('Teacher output lost exact current measurement')
                        measured = bank['boxes'][index].copy(); measured[2:] -= measured[:2]
                        stream.write(f'{frame},{int(track.track_id)},{measured[0]:.3f},{measured[1]:.3f},{measured[2]:.3f},{measured[3]:.3f},{float(track.score):.6f},-1,-1,-1\n')
        finally:
            recorder.close()
        if sha256(prediction) != rm['prediction_sha256'] or prediction.read_bytes() != (reference/(video+'.txt')).read_bytes():
            raise ValueError('Read-only teacher observation changed full native output')
        report['videos'][video] = dict(frames=len(bank['frame_ids']), teacher_prediction_byte_equal=True,
            prediction_sha256=sha256(prediction), trace_sha256=sha256(folder/'decisions.jsonl.gz'),
            feature_manifest_sha256=bank['manifest_sha256'], coordinate_sha256=bank['metadata']['encoder']['coordinate_sha256'],
            online_physical_labels_loaded=False)
        write_json(a.output/'manifest.json', report); print(json.dumps(dict(video=video, frames=len(bank['frame_ids']), prediction_parity=True)), flush=True)
    report['status'] = 'ALL_FOUR_GT_FREE_TEACHER_PARITIES_VERIFIED'; write_json(a.output/'manifest.json', report)
    from cfp.recovery_supervision import physical_bank
    coordinates = {v['coordinate_sha256'] for v in report['videos'].values()}
    if len(coordinates) != 1:
        raise ValueError('No mixing different encoders with identical width')
    for video in c['fit_videos']:
        physical = physical_bank(video, 'train')
        bank = load_native_source(a.root/f'native_dino_frozen_{video}_k4_v1', a.detections, video)
        if physical['detection_sha256'] != bank['detection_sha256']:
            raise ValueError('Offline labels must refer to identical actual proposals')
        parts = {k: [] for k in ['indices', 'times', 'track_identity', 'current', 'current_identity', 'confidence', 'geometry', 'baseline_cost', 'state']}
        frames, to, do, eo, supervisable = [], [0], [0], [0], []
        counts = Counter(); maximum_tracks = 0
        with gzip.open(a.output/video/'decisions.jsonl.gz', 'rt') as stream:
            for row in map(json.loads, stream):
                if row['stage'] != 'high' or (row['frame']-1) % c['query_stride']:
                    continue
                frame = row['frame']; n, m = len(row['tracks']), len(row['proposals'])
                if not n or not m:
                    counts['no_joint_conditional_set'] += 1; continue
                if n > c['capacity']:
                    raise ValueError('Native set exceeds capacity; no silent truncation')
                maximum_tracks = max(maximum_tracks, n)
                indices = np.full((n, c['history_length']), -1, np.int64); times = np.zeros_like(indices)
                identities = np.full(n, -1, np.int64)
                for j, track in enumerate(row['tracks']):
                    history = [(int(t), int(index)) for t, index in track['history'] if index is not None and frame-c['maximum_history_age_frames'] <= t][-c['history_length']:]
                    for k, (t, index) in enumerate(history):
                        if t >= frame or physical['frame_of_proposal'][index] != t:
                            raise ValueError('Actual teacher history contains current/future row')
                        indices[j, k], times[j, k] = index, t
                    ids = indices[j][indices[j] >= 0]
                    identities[j] = consistent_past_identity(ids, physical['labels'], c['minimum_known_past'])
                current = np.asarray(row['proposals'], np.int64)
                current_ids = physical['labels'][current].copy(); current_ids[~bank['valid'][current]] = -1
                targets = joint_identity_target_sets(torch.from_numpy(identities), torch.from_numpy(current_ids))
                supervised = bool(targets['positives'].any()); counts.update(targets['reason'])
                counts['frames'] += 1; counts['supervisable_frames'] += int(supervised); counts['track_slots'] += n
                counts['unknown_mixed_or_short_track_slots'] += int((identities < 0).sum())
                states = np.array([[np.log1p(frame-t['start_frame'])/np.log(15001), np.log1p(frame-t['end_frame'])/np.log(15001), float(t['state']==2)] for t in row['tracks']], np.float32)
                entry = dict(indices=indices, times=times, track_identity=identities, current=current, current_identity=current_ids,
                    confidence=np.asarray(row['scores'], np.float32), geometry=np.asarray(row['geometry'], np.float32).ravel(),
                    baseline_cost=np.asarray(row['costs'], np.float32).ravel(), state=states)
                for key, value in entry.items():
                    parts[key].append(value)
                frames.append(frame); to.append(to[-1]+n); do.append(do[-1]+m); eo.append(eo[-1]+n*m); supervisable.append(supervised)
        if not counts['supervisable_frames']:
            raise ValueError('No verified fitting supervision; do not fabricate labels')
        path = a.output/(video+'.npz')
        np.savez_compressed(path, frame=np.array(frames, np.int64), track_offsets=np.array(to, np.int64),
            detection_offsets=np.array(do, np.int64), edge_offsets=np.array(eo, np.int64), supervisable=np.array(supervisable, bool),
            **{k: np.concatenate(v, axis=0) for k, v in parts.items()})
        report['videos'][video].update(query_sha256=sha256(path), annotation_sha256=physical['annotation_sha256'], counts=dict(counts), maximum_tracks=maximum_tracks)
        write_json(a.output/'manifest.json', report); print(json.dumps(dict(video=video, counts=dict(counts), maximum_tracks=maximum_tracks)), flush=True)
    report['status'] = 'COMPLETE_FIT_ONLY_NATIVE_DINO_SET_QUERIES'; write_json(a.output/'manifest.json', report)


if __name__ == '__main__':
    main()
