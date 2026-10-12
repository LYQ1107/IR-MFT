#!/usr/bin/env python3
"""Run one frozen lifecycle/association cell; inference never loads annotations."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch

from cfp.lifecycle_crossover import load_bot_reference, make_bot_lifecycle, make_cfp_lifecycle, DecisionRecorder
from cfp.predictive import LatentIdentityPredictor
from cfp.provenance import sha256, snapshot_sources, write_json
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker import matching as cfp_matching


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--association', choices=['bot', 'gru'], required=True)
    p.add_argument('--lifecycle', choices=['bot', 'cfp'], required=True)
    p.add_argument('--candidate', type=int, choices=[0, 1, 2], required=True)
    p.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    p.add_argument('--partition', choices=['dev', 'train'], default='dev')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-frames', type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Crossover replay immutable')
    root = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    protocol_path, phase_path = Path('configs/cfp/protocol_v2.json'), Path('configs/cfp/phase_v_protocol_v1.json')
    protocol, phase = [json.loads(x.read_text()) for x in (protocol_path, phase_path)]
    if phase['state'] != 'FROZEN' or sha256(protocol_path) != phase['protocol_sha256']:
        raise ValueError('Protocol drift')
    free_at_start = shutil.disk_usage(root).free
    if free_at_start < 40 * 2**30:
        raise OSError('40GiB reserve')
    features_root = root / ('phase1' if args.partition == 'dev' else 'phase2') / f'causal_{args.partition}_detection_features_postneck_seed{args.seed}'
    detections_root = root / 'phase0' / f'causal_v3_{args.partition}_detections'
    fm, dm = [json.loads((x / 'manifest.json').read_text()) for x in (features_root, detections_root)]
    if any(x['status'] != 'COMPLETE' or x['protocol_sha256'] != sha256(protocol_path) for x in (fm, dm)):
        raise ValueError('Incomplete inputs')
    if fm['detection_manifest_sha256'] != sha256(detections_root / 'manifest.json') or fm['rgb_encoder']['seed'] != args.seed:
        raise ValueError('Feature/proposal/seed mismatch')
    native = load_bot_reference('/data3/liuyeqiang/CFP-FishMOT_references/BoT-SORT')
    external = json.loads(Path('configs/cfp/external_association_v1.json').read_text())['bot_sort']
    external.pop('parameter_source')
    threshold = external.pop('appearance_thresholds')[args.candidate]
    weight = phase['lifecycle_crossover']['gru_appearance_weights'][args.candidate]
    predictor = None
    checkpoint_path = root / 'phase3' / f'predictor_v3_gru_association_anchor_seed{args.seed}/epoch8.pt'
    if args.association == 'gru':
        metadata = json.loads(checkpoint_path.with_name('manifest.json').read_text())
        train_feature = json.loads((Path(metadata['features_root']) / 'manifest.json').read_text())
        if metadata['status'] != 'COMPLETE' or sha256(checkpoint_path) != metadata['checkpoint_sha256'] or metadata['seed'] != args.seed:
            raise ValueError('Model changed')
        if train_feature['rgb_encoder']['checkpoint_sha256'] != fm['rgb_encoder']['checkpoint_sha256'] or metadata['history'] != 'anchor_recent':
            raise ValueError('Model input coordinates/context differ')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if checkpoint['epoch'] != 8:
            raise ValueError('Fixed epoch8 required')
        predictor = LatentIdentityPredictor(512, 'gru')
        predictor.load_state_dict(checkpoint['model'], strict=True)
        predictor.eval()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True)
    record = dict(status='RUNNING', association=args.association, lifecycle=args.lifecycle, seed=args.seed, candidate=args.candidate,
                  partition=args.partition, probe=bool(args.max_frames), appearance_threshold=threshold, appearance_weight=weight,
                  phase_protocol_sha256=sha256(phase_path), protocol_sha256=sha256(protocol_path),
                  feature_manifest_sha256=sha256(features_root / 'manifest.json'), detection_manifest_sha256=sha256(detections_root / 'manifest.json'),
                  checkpoint_sha256=sha256(checkpoint_path) if predictor else None, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_snapshots=snapshot_sources(args.output, [__file__, 'cfp/lifecycle_crossover.py', 'cfp/bot_lifecycle.py', 'cfp/association.py', 'cfp/predictive_association.py', 'cfp/history.py', 'cfp/predictive.py', phase_path]),
                  videos={}, scope='Protocol A cached frozen proposals/features; observed current/past matrices, no online GT. Throughput excludes detector/encoding/decode.')
    write_json(args.output / 'manifest.json', record)
    for seq in protocol[args.partition]:
        dp, fp = detections_root / (seq + '.npz'), features_root / (seq + '.npz')
        fmeta = json.loads(fp.with_suffix('.provenance.json').read_text())
        if sha256(dp) != dm['caches'][seq]['sha256'] or sha256(fp) != fmeta['features_sha256'] or fmeta['detection_sha256'] != sha256(dp):
            raise ValueError('Current bank changed')
        with np.load(dp, allow_pickle=False) as bank:
            frames, offsets, boxes, scores = [bank[key] for key in ('frame_ids', 'offsets', 'boxes', 'scores')]
        with np.load(fp, allow_pickle=False) as bank:
            features = bank['rgb']
        if len(features) != len(boxes) or not np.isfinite(features).all() or np.any(np.linalg.norm(features, axis=1) < 1e-8):
            raise ValueError('Aligned valid features required')
        BaseTrack._count = 0
        tracker = make_bot_lifecycle(native, external, args.association, predictor, weight, threshold) if args.lifecycle == 'bot' else make_cfp_lifecycle(native, args.association, predictor, weight, threshold)
        recorder = DecisionRecorder(args.output / (seq + '.decisions.jsonl.gz'), tracker)
        times, counts = [], dict(output=0, confirmed=0, unconfirmed=0, low_match=0)
        # Auxiliary files change only output filtering/coordinates, never tracking.
        with (args.output / (seq + '.txt')).open('w') as stream, (args.output / (seq + '.confirmed.txt')).open('w') as confirmed, (args.output / (seq + '.all.txt')).open('w') as all_tracks, (args.output / (seq + '.measurement.txt')).open('w') as measurement:
            try:
                for i, frame in enumerate(frames):
                    if args.max_frames and frame > args.max_frames:
                        break
                    lo, hi = offsets[i:i + 2]
                    detections = np.concatenate([boxes[lo:hi].copy(), scores[lo:hi, None]], axis=1)
                    start = time.perf_counter()
                    with recorder.capture([native.matching, cfp_matching]):
                        output = tracker.update(detections, features[lo:hi].copy(), np.arange(lo, hi))
                    times.append(time.perf_counter() - start)
                    visible = {int(t.track_id) for t in output}
                    for track in tracker.tracked_stracks:
                        tlwh = track.tlwh
                        if np.prod(tlwh[2:]) <= 100:
                            continue
                        row = f'{frame},{int(track.track_id)},{tlwh[0]:.3f},{tlwh[1]:.3f},{tlwh[2]:.3f},{tlwh[3]:.3f},{float(track.score):.6f},-1,-1,-1\n'
                        all_tracks.write(row)
                        if track.is_activated:
                            confirmed.write(row)
                        if int(track.track_id) in visible:
                            stream.write(row)
                            counts['output'] += 1
                            counts['confirmed' if track.is_activated else 'unconfirmed'] += 1
                            index = track.current_proposal_index
                            if track.frame_id != frame or not lo <= index < hi:
                                raise ValueError('Measurement output must use actual current matched proposal')
                            measured = boxes[index].copy()
                            measured[2:] -= measured[:2]
                            measurement.write(f'{frame},{int(track.track_id)},{measured[0]:.3f},{measured[1]:.3f},{measured[2]:.3f},{measured[3]:.3f},{float(track.score):.6f},-1,-1,-1\n')
                    if len(visible) != len(output):
                        raise ValueError('Duplicate output identity')
                    if i % 128 == 0:
                        free = shutil.disk_usage(root).free
                        if free < 40 * 2**30 or free_at_start - free > 2 * 2**30:
                            raise OSError('Storage reserve/run budget reached')
            finally:
                recorder.close()
        parity = None
        if args.partition == 'dev' and not args.max_frames and args.association == args.lifecycle.replace('cfp', 'gru'):
            original = root / ('phase1' if args.association == 'bot' else 'phase3') / (f'causal_dev_postneck_bot_sort_seed{args.seed}_c{args.candidate}' if args.association == 'bot' else f'plim_v3_dev_gru_association_anchor_seed{args.seed}_w{weight}') / (seq + '.txt')
            parity = (args.output / (seq + '.txt')).read_bytes() == original.read_bytes()
            if not parity:
                raise ValueError('Diagonal adapter failed complete trajectory byte parity: ' + seq)
        record['videos'][seq] = dict(frames=len(times), output_rows=counts, original_diagonal_byte_equal=parity,
                                     prediction_sha256=sha256(args.output / (seq + '.txt')), decisions_sha256=sha256(args.output / (seq + '.decisions.jsonl.gz')),
                                     association_seconds=sum(times), latency_p50_p95_ms=(np.quantile(times, [.5, .95]) * 1000).tolist())
        write_json(args.output / 'manifest.json', record)
        print(seq, 'complete; diagonal parity', parity, flush=True)
    record['status'] = 'PROBE_COMPLETE' if args.max_frames else 'COMPLETE'
    write_json(args.output / 'manifest.json', record)


if __name__ == '__main__':
    main()
