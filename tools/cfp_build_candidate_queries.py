#!/usr/bin/env python3
"""Index actual accepted-history graphs, including absent candidates and NULL.

Historical teacher traces initialize exploration, never presented as genuine OOF
tracker states. No GT repair is applied to histories, boxes, candidates or scores.
"""
import argparse
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from cfp.candidate_graph import CandidateGraph
from cfp.predictive import LatentIdentityPredictor
from cfp.provenance import sha256, snapshot_sources, write_json
from cfp.recoverability import observable_candidate_features, FEATURE_NAMES
from cfp.recoverability_diagnosis import EarlyReference
from cfp.recovery_supervision import physical_bank, history_view


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stride', type=int, default=5)
    args = p.parse_args()
    if args.output.exists() or args.stride < 1:
        raise ValueError('New immutable output and positive stride required')
    run = json.loads((args.run / 'manifest.json').read_text())
    if run['status'] != 'COMPLETE' or run['partition'] != 'train' or run['association'] != 'gru' or run['probe']:
        raise ValueError('Complete actual GRU train trace required')
    root = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    if shutil.disk_usage(root).free < 40 * 2**30:
        raise OSError('40 GiB reserve')
    args.output.mkdir(parents=True)
    seed = run['seed']
    checkpoint = root / 'phase3' / f'predictor_v3_gru_association_anchor_seed{seed}/epoch8.pt'
    metadata = json.loads(checkpoint.with_name('manifest.json').read_text())
    if sha256(checkpoint) != metadata['checkpoint_sha256']:
        raise ValueError('Frozen GRU weight changed')
    torch.set_num_threads(2)
    model = LatentIdentityPredictor().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    feature_root = root / 'phase2' / f'causal_train_detection_features_postneck_seed{seed}'
    report = dict(status='RUNNING', partition='train', seed=seed, stride=args.stride,
        historical_trace_manifest_sha256=sha256(args.run / 'manifest.json'), videos={},
        historical_gru_checkpoint_sha256=sha256(checkpoint), feature_names=FEATURE_NAMES,
        source_snapshots=snapshot_sources(args.output, [__file__, 'cfp/candidate_graph.py',
            'cfp/recoverability.py', 'cfp/recoverability_diagnosis.py', 'cfp/recovery_supervision.py']),
        label_policy='Known physical edge labels only. Unknown edge labels=-1, never wrong. '
            'NULL class eligible only when every current-stage candidate has a known label, '
            'or a known positive exists; unknown candidates excluded from CE denominator, not inference pool.',
        exposure='Warm-start historical teacher accepted states. Original GRU fitted five train '
            'videos, SN-015 calibrated. Frozen FishReID saw all six. New risk/model folds must '
            'hold out entire videos. This artifact is not fully out-of-fold tracking evidence.')
    write_json(args.output / 'manifest.json', report)
    for seq in run['videos']:
        bank = physical_bank(seq, 'train')
        fp = feature_root / (seq + '.npz')
        if sha256(fp) != json.loads(fp.with_suffix('.provenance.json').read_text())['features_sha256']:
            raise ValueError('Frozen feature drift')
        with np.load(fp, allow_pickle=False) as data:
            features = data['rgb']
        trace = args.run / (seq + '.decisions.jsonl.gz')
        if sha256(trace) != run['videos'][seq]['decisions_sha256']:
            raise ValueError('Trace drift')
        refs, rows = EarlyReference(), []
        with gzip.open(trace, 'rt') as stream, torch.inference_mode():
            for line in stream:
                row = json.loads(line)
                targets = [refs.update(track, bank['labels']) for track in row['tracks']]
                if row['stage'] != 'high' or (row['frame'] - 1) % args.stride or not row['tracks']:
                    continue
                n, m = len(row['tracks']), len(row['proposals'])
                histories = [history_view(track) for track in row['tracks']]
                past = np.zeros((n, 8, 512), np.float32)
                times = np.zeros((n, 8), np.int64)
                valid = np.zeros((n, 8), bool)
                indices = np.full((n, 8), -1, np.int64)
                for i, history in enumerate(histories):
                    for j, (time, index) in enumerate(history):
                        past[i, j], times[i, j], valid[i, j], indices[i, j] = features[index], time, True, index
                prototype, available = model(torch.from_numpy(past), torch.from_numpy(times),
                    torch.from_numpy(valid), torch.full((n,), row['frame'], dtype=torch.int64))
                boxes = np.asarray(row['boxes'], np.float32).reshape(m, 4)
                costs = np.asarray(row['costs'], np.float32).reshape(n, m)
                # All current-stage columns are real inputs; the original geometry
                # blending region is recorded separately, not silently a hard mask.
                graph = CandidateGraph(torch.from_numpy(past), torch.from_numpy(times), torch.from_numpy(valid),
                    torch.from_numpy(features[np.asarray(row['proposals'], np.int64)]),
                    torch.tensor([t['box'] for t in row['tracks']], dtype=torch.float32),
                    torch.from_numpy(boxes), torch.tensor(row['scores'], dtype=torch.float32),
                    torch.from_numpy(costs), torch.ones(n, m, dtype=torch.bool),
                    torch.tensor([row['frame'] - t['start_frame'] for t in row['tracks']]),
                    torch.tensor([row['frame'] - t['end_frame'] for t in row['tracks']]), row['frame'], prototype=prototype)
                extracted = observable_candidate_features(graph).numpy()
                labels = bank['labels'][row['proposals']]
                matches = dict(row['matches'])
                for i, target in enumerate(targets):
                    if target is None:
                        continue
                    risk_labels = np.where(labels < 0, -1, (labels != target).astype(np.int8)).astype(np.int8)
                    positives = np.flatnonzero(labels == target)
                    null_known = bool(len(positives) or np.all(labels >= 0))
                    rows.append(dict(frame=row['frame'], track_id=row['tracks'][i]['track_id'], target=target,
                        history_indices=indices[i], history_frames=times[i], candidates=np.asarray(row['proposals'], np.int64),
                        geometry=np.asarray(row['geometry'], np.float32).reshape(n, m)[i],
                        features=extracted[i], risk_labels=risk_labels, null_known=null_known,
                        positive=int(positives[0]) if len(positives) else m,
                        assigned=int(matches.get(i, -1)), baseline_cost=costs[i],
                        correct_history=int(np.sum(bank['labels'][indices[i][valid[i]]] == target)),
                        unknown_history=int(np.sum(bank['labels'][indices[i][valid[i]]] < 0))))
        offsets = np.r_[0, np.cumsum([len(r['candidates']) for r in rows])]
        arrays = {key: np.array([r[key] for r in rows]) for key in ('frame', 'track_id', 'target',
            'history_indices', 'history_frames', 'null_known', 'positive', 'assigned', 'correct_history', 'unknown_history')}
        arrays['candidate_offsets'] = offsets
        for key in ('candidates', 'geometry', 'features', 'risk_labels', 'baseline_cost'):
            arrays[key] = np.concatenate([r[key] for r in rows], axis=0)
        output = args.output / (seq + '.npz')
        np.savez_compressed(output, **arrays)
        report['videos'][seq] = dict(queries=len(rows), candidate_edges=int(offsets[-1]),
            known_edges=int(np.sum(arrays['risk_labels'] >= 0)),
            known_wrong_edges=int(np.sum(arrays['risk_labels'] == 1)),
            null_eligible=int(arrays['null_known'].sum()),
            target_absent_queries=sum(r['positive'] == len(r['candidates']) for r in rows),
            feature_sha256=sha256(fp), trace_sha256=sha256(trace), sha256=sha256(output),
            detection_sha256=bank['detection_sha256'])
        write_json(args.output / 'manifest.json', report)
        print(seq, report['videos'][seq], flush=True)
    report.update(status='COMPLETE', storage_free_gib=shutil.disk_usage(root).free / 2**30)
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
