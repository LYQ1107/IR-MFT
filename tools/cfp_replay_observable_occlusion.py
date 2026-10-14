#!/usr/bin/env python3
"""Independent association-only observableoverlap controls on unchanged native replay."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from cfp.lifecycle_crossover import load_bot_reference, make_cfp_lifecycle
from cfp.fish_motion import install_native_motion_filter
from cfp.observable_occlusion import install_observable_occlusion
from cfp.predictive import LatentIdentityPredictor
from cfp.provenance import sha256, snapshot_sources, write_json
from yolox.tracker.basetrack import BaseTrack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CPU-only isolated motion process required')
    cfg = json.loads(args.config.read_text())
    if args.output.exists() or cfg['state'] != 'FROZEN' or cfg['motion_filter'] not in ('kalman','unscented_linear') or cfg['appearance_weight'] not in (0.,.3):
        raise ValueError('Frozen route and immutable new full-video output required')
    assets = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    if shutil.disk_usage(assets).free < 40 * 2**30:
        raise OSError('40GiB reserve')
    protocol_path = Path('configs/cfp/protocol_v2.json')
    protocol = json.loads(protocol_path.read_text())
    if set(cfg['sequences']) - set(protocol['train']):
        raise ValueError('First controlled replay restricted to six training videos')
    features_root = assets / 'phase2/causal_train_detection_features_postneck_seed0'
    detection_root = assets / 'phase0/causal_v3_train_detections'
    fm, dm = [json.loads((p / 'manifest.json').read_text()) for p in (features_root, detection_root)]
    checkpoint = assets / 'phase3/predictor_v3_gru_association_anchor_seed0/epoch8.pt'
    pm = json.loads(checkpoint.with_name('manifest.json').read_text())
    predictor_features = json.loads((Path(pm['features_root']) / 'manifest.json').read_text())
    if (sha256(checkpoint) != pm['checkpoint_sha256'] or fm['status'] != 'COMPLETE' or dm['status'] != 'COMPLETE'
            or fm['protocol_sha256'] != sha256(protocol_path) or dm['protocol_sha256'] != sha256(protocol_path)
            or fm['rgb_encoder']['checkpoint_sha256'] != predictor_features['rgb_encoder']['checkpoint_sha256']):
        raise ValueError('Baseline predictor/input coordinate provenance changed')
    predictor = LatentIdentityPredictor().eval()
    predictor.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    torch.set_num_threads(2)
    native = load_bot_reference('/data3/liuyeqiang/CFP-FishMOT_references/BoT-SORT')
    args.output.mkdir(parents=True)
    report = dict(status='RUNNING', config=cfg, config_sha256=sha256(args.config), videos={}, partition='train',
        baseline_checkpoint_sha256=sha256(checkpoint), motion_filter=cfg['motion_filter'],appearance_weight=cfg['appearance_weight'],occlusion_mode=cfg['occlusion_mode'],occlusion_seed=cfg['occlusion_seed'],
        detection_manifest_sha256=sha256(detection_root / 'manifest.json'), feature_manifest_sha256=sha256(features_root / 'manifest.json'),
        protocol_sha256=sha256(protocol_path), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        scope=cfg['scope'],
        source_snapshots=snapshot_sources(args.output / 'sources',[__file__,args.config,'cfp/fish_motion.py','cfp/observable_occlusion.py','cfp/fish_geometry.py','cfp/lifecycle_crossover.py','cfp/predictive.py','cfp/history.py','cfp/association.py','cfp/predictive_association.py','yolox/tracker/kalman_filter.py','yolox/tracker/byte_tracker.py','yolox/tracker/matching.py']))
    write_json(args.output / 'manifest.json', report)
    for video in cfg['sequences']:
        dp, fp = detection_root / (video + '.npz'), features_root / (video + '.npz')
        provenance = json.loads(fp.with_suffix('.provenance.json').read_text())
        if sha256(dp) != dm['caches'][video]['sha256'] or sha256(fp) != provenance['features_sha256'] or provenance['detection_sha256'] != sha256(dp):
            raise ValueError('Frozen replay inputs changed')
        with np.load(dp, allow_pickle=False) as a:
            frames, offsets, boxes, scores = [a[k] for k in ('frame_ids', 'offsets', 'boxes', 'scores')]
        with np.load(fp, allow_pickle=False) as a:
            features = a['rgb'].copy()
        if not np.array_equal(frames, np.arange(1, len(frames) + 1)) or len(features) != len(boxes):
            raise ValueError('Complete aligned video required')
        BaseTrack._count = 0
        tracker = make_cfp_lifecycle(native, 'gru', predictor, cfg['appearance_weight'])
        tracker = install_native_motion_filter(tracker,cfg['motion_filter'])
        tracker = install_observable_occlusion(tracker,cfg['occlusion_mode'],cfg['occlusion_seed'])
        times, count = [], 0
        output_path = args.output / (video + '.txt')
        with output_path.open('w') as stream:
            for i, frame in enumerate(frames):
                lo, hi = offsets[i:i + 2]
                start = time.perf_counter()
                output = tracker.update(np.c_[boxes[lo:hi].copy(), scores[lo:hi]], features[lo:hi].copy(), np.arange(lo, hi))
                for track in output:
                    if np.prod(track.tlwh[2:]) <= 100:
                        continue
                    index = track.current_proposal_index
                    if track.frame_id != frame or not lo <= index < hi:
                        raise ValueError('Output must use actual current measured proposal')
                    measured = boxes[index].copy()
                    measured[2:] -= measured[:2]
                    stream.write(f'{frame},{int(track.track_id)},{measured[0]:.3f},{measured[1]:.3f},{measured[2]:.3f},{measured[3]:.3f},{float(track.score):.6f},-1,-1,-1\n')
                    count += 1
                times.append(time.perf_counter() - start)
                if i % 128 == 0 and shutil.disk_usage(assets).free < 40 * 2**30:
                    raise OSError('40GiB reserve')
        reference = assets / 'phase5/cross_train_gru_cfp_s0_c1_v1' / (video + '.measurement.txt')
        parity = output_path.read_bytes() == reference.read_bytes()
        if cfg['motion_filter']=='kalman' and cfg['appearance_weight']==.3 and cfg['occlusion_mode']=='native' and not parity:
            raise ValueError('Disabled full-video baseline byte parity failed')
        report['videos'][video] = dict(frames=len(times), rows=count, prediction_sha256=sha256(output_path),
            baseline_byte_equal=parity, association_seconds=sum(times),
            latency_p50_p95_ms=(np.quantile(times, [.5, .95]) * 1000).tolist(), motion_calls=getattr(tracker.kalman_filter,'calls',None),occlusion_counts=tracker.occlusion_counts)
        write_json(args.output / 'manifest.json', report)
        print(video, report['videos'][video], flush=True)
    report['status'] = 'COMPLETE'
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
