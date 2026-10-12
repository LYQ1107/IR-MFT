#!/usr/bin/env python3
"""Fair recovery/read/write controls on the same current-box-output lifecycle."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from cfp.identity_controls import MultiPrototypeMemory
from cfp.lifecycle_crossover import load_bot_reference, make_cfp_lifecycle, DecisionRecorder
from cfp.predictive import LatentIdentityPredictor
from cfp.recoverable_identity import RecoverableIdentityState
from cfp.provenance import sha256, snapshot_sources, write_json
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker import matching


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['ris', 'trained_state', 'untrained_ris', 'last', 'ema', 'fifo', 'all_history_ema', 'anchor_only', 'multi_proto', 'ordinary_gru', 'predictive_gru', 'ordinary_transformer', 'reject_write', 'soft_write'], required=True)
    p.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    p.add_argument('--weight', type=float, choices=[.1, .3, .5], required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--record-decisions', action='store_true')
    p.add_argument('--read-view', choices=['anchor_recent', 'early3_recent5'], default='anchor_recent')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Recovery replay immutable')
    root = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    if shutil.disk_usage(root).free < 40 * 2**30:
        raise OSError('40GiB reserve')
    protocol = json.loads(Path('configs/cfp/protocol_v2.json').read_text())
    feature_root = root / 'phase1' / f'causal_dev_detection_features_postneck_seed{args.seed}'
    detections_root = root / 'phase0/causal_v3_dev_detections'
    fm, dm = [json.loads((x / 'manifest.json').read_text()) for x in (feature_root, detections_root)]
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    native = load_bot_reference('/data3/liuyeqiang/CFP-FishMOT_references/BoT-SORT')
    model, mode = None, args.kind
    path = args.checkpoint
    if args.kind in ('ris', 'trained_state'):
        if path is None:
            raise ValueError('Explicit completed model required')
        metadata = json.loads(path.with_name('manifest.json').read_text())
        if metadata['status'] != 'COMPLETE' or sha256(path) != metadata['checkpoint_sha256'] or metadata['guard']['seed'] != args.seed or metadata['rgb_encoder']['checkpoint_sha256'] != fm['rgb_encoder']['checkpoint_sha256']:
            raise ValueError('Model/encoder provenance differs')
        if metadata['config'].get('read_view', 'anchor_recent') != args.read_view:
            raise ValueError('Trained recovery state and deployment context differ')
        family = metadata['config'].get('model_family', 'ris')
        if family in ('matched_gru', 'matched_transformer'):
            model = LatentIdentityPredictor(512, 'gru' if family == 'matched_gru' else 'transformer', hidden=metadata['config']['hidden'])
        else:
            model = RecoverableIdentityState(metadata['config']['hidden'], semantic=metadata['config'].get('semantic', False))
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint['epoch'] != 8:
            raise ValueError('Fixed last epoch8 required')
        model.load_state_dict(checkpoint['model'], strict=True)
        mode = 'predictor'
    elif args.kind in ('untrained_ris', 'multi_proto'):
        model = RecoverableIdentityState() if args.kind == 'untrained_ris' else MultiPrototypeMemory()
        mode = 'predictor'
    elif args.kind in ('ordinary_gru', 'predictive_gru', 'ordinary_transformer', 'reject_write'):
        architecture = 'transformer' if args.kind == 'ordinary_transformer' else 'gru'
        objective = 'predictive' if args.kind == 'predictive_gru' else 'association'
        path = root / 'phase3' / f'predictor_v3_{architecture}_{objective}_anchor_seed{args.seed}/epoch8.pt'
        metadata = json.loads(path.with_name('manifest.json').read_text())
        if metadata['status'] != 'COMPLETE' or sha256(path) != metadata['checkpoint_sha256'] or metadata['seed'] != args.seed:
            raise ValueError('Old model changed')
        source = json.loads((Path(metadata['features_root']) / 'manifest.json').read_text())
        if source['rgb_encoder']['checkpoint_sha256'] != fm['rgb_encoder']['checkpoint_sha256']:
            raise ValueError('Old model coordinate space differs')
        model = LatentIdentityPredictor(512, architecture)
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'], strict=True)
        mode = 'predictor'
    elif args.kind == 'soft_write':
        mode = 'all_history_ema'
    if model is not None:
        model.eval()
    args.output.mkdir(parents=True)
    record = dict(status='RUNNING', kind=args.kind, seed=args.seed, appearance_weight=args.weight, partition='dev', read_view=args.read_view,
                  checkpoint_path=str(path) if path else None, checkpoint_sha256=sha256(path) if path else None,
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_snapshots=snapshot_sources(args.output, [__file__, 'cfp/lifecycle_crossover.py', 'cfp/recoverable_identity.py', 'cfp/identity_controls.py', 'cfp/predictive_association.py', 'cfp/association.py']),
                  feature_manifest_sha256=sha256(feature_root / 'manifest.json'), detection_manifest_sha256=sha256(detections_root / 'manifest.json'),
                  output_policy='current_matched_measurement', videos={}, scope='Protocol A recovery control; same frozen proposals, originalCFP motion/lifecycle, current boxes output. No GT/annotation reads, no future history. Cached association timing excludes whole pipeline.')
    write_json(args.output / 'manifest.json', record)
    for seq in protocol['dev']:
        dp, fp = detections_root / (seq + '.npz'), feature_root / (seq + '.npz')
        fmeta = json.loads(fp.with_suffix('.provenance.json').read_text())
        if sha256(dp) != dm['caches'][seq]['sha256'] or sha256(fp) != fmeta['features_sha256'] or fmeta['detection_sha256'] != sha256(dp):
            raise ValueError('Frozen proposals/features changed')
        with np.load(dp, allow_pickle=False) as archive:
            frames, offsets, boxes, scores = [archive[key] for key in ('frame_ids', 'offsets', 'boxes', 'scores')]
        with np.load(fp, allow_pickle=False) as archive:
            features = archive['rgb']
        BaseTrack._count = 0
        policy = dict(minimum_confidence=.85, minimum_row_margin=.05, minimum_column_margin=.05) if args.kind == 'reject_write' else None
        tracker = make_cfp_lifecycle(native, 'gru', model, args.weight, write_policy=policy, soft_write=args.kind == 'soft_write', memory_mode=mode, read_view=args.read_view)
        tracker.memory_mode = mode
        recorder = DecisionRecorder(args.output / (seq + '.decisions.jsonl.gz'), tracker) if args.record_decisions else None
        times, count = [], 0
        with (args.output / (seq + '.txt')).open('w') as stream:
            try:
                for i, frame in enumerate(frames):
                    lo, hi = offsets[i:i + 2]
                    proposals = np.concatenate([boxes[lo:hi].copy(), scores[lo:hi, None]], axis=1)
                    start = time.perf_counter()
                    if recorder:
                        with recorder.capture([matching, native.matching]):
                            output = tracker.update(proposals, features[lo:hi].copy(), np.arange(lo, hi))
                    else:
                        output = tracker.update(proposals, features[lo:hi].copy(), np.arange(lo, hi))
                    if len({int(t.track_id) for t in output}) != len(output):
                        raise ValueError('Duplicate online identity')
                    for track in output:
                        if np.prod(track.tlwh[2:]) <= 100:
                            continue
                        index = track.current_proposal_index
                        if track.frame_id != frame or not lo <= index < hi:
                            raise ValueError('Measurement must be actual current match')
                        box = boxes[index].copy()
                        box[2:] -= box[:2]
                        stream.write(f'{frame},{int(track.track_id)},{box[0]:.3f},{box[1]:.3f},{box[2]:.3f},{box[3]:.3f},{float(track.score):.6f},-1,-1,-1\n')
                        count += 1
                    times.append(time.perf_counter() - start)
                    if i % 128 == 0 and shutil.disk_usage(root).free < 40 * 2**30:
                        raise OSError('40GiB reserve')
            finally:
                if recorder:
                    recorder.close()
        if args.kind == 'ordinary_gru' and args.weight == .3 and args.read_view == 'anchor_recent':
            reference = root / 'phase5' / f'cross_gru_cfp_s{args.seed}_c1_v1' / (seq + '.measurement.txt')
            if (args.output / (seq + '.txt')).read_bytes() != reference.read_bytes():
                raise ValueError('Disabled recovery does not reproduce strong ordinary measurement baseline')
        record['videos'][seq] = dict(frames=len(times), rows=count, prediction_sha256=sha256(args.output / (seq + '.txt')),
                                    association_seconds=sum(times), latency_p50_p95_ms=(np.quantile(times, [.5, .95]) * 1000).tolist(),
                                    write_counts=tracker.write_counts, decisions_sha256=sha256(args.output / (seq + '.decisions.jsonl.gz')) if recorder else None)
        write_json(args.output / 'manifest.json', record)
        print(seq, 'complete', flush=True)
    record['status'] = 'COMPLETE'
    write_json(args.output / 'manifest.json', record)


if __name__ == '__main__':
    main()
