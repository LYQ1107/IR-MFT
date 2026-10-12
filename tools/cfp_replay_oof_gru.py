#!/usr/bin/env python3
"""Generate genuine accepted states with video-excluded ordinary bootstrap weights."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from cfp.ambiguity_association import AmbiguityAssociation
from cfp.lifecycle_crossover import load_bot_reference, make_cfp_lifecycle, DecisionRecorder
from cfp.online_recovery import install_risk_association
from cfp.predictive import LatentIdentityPredictor
from cfp.provenance import sha256, snapshot_sources, write_json
from cfp.recoverability import RecoverabilityEstimator
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker import matching


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if config['state'] != 'FROZEN' or args.output.exists() or config['partition'] not in ('train', 'dev'):
        raise ValueError('Frozen train/dev config and immutable new output required')
    if config['K'] == 2 and (config.get('risk_policy', 'disabled') != 'disabled' or config.get('protected_write')):
        raise ValueError('Initial branching independently controls ordinary GRU; risk combination requires separate state adapter')
    root = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    if shutil.disk_usage(root).free < 40 * 2**30:
        raise OSError('40 GiB reserve')
    args.output.mkdir(parents=True)
    protocol = json.loads(Path('configs/cfp/protocol_v2.json').read_text())
    partition, seed = config['partition'], config['seed']
    features_root = root / ('phase2' if partition == 'train' else 'phase1') / f'causal_{partition}_detection_features_postneck_seed{seed}'
    detection_root = root / 'phase0' / f'causal_v3_{partition}_detections'
    dm = json.loads((detection_root / 'manifest.json').read_text())
    fm = json.loads((features_root / 'manifest.json').read_text())
    checkpoint = Path(config['predictor_checkpoint'])
    meta = json.loads(checkpoint.with_name('manifest.json').read_text())
    if meta['status'] != 'COMPLETE' or sha256(checkpoint) != meta['inference_sha256']:
        raise ValueError('OOF ordinary export incomplete or changed')
    if meta['rgb_encoder']['checkpoint_sha256'] != fm['rgb_encoder']['checkpoint_sha256']:
        raise ValueError('OOF ordinary encoder coordinate space differs')
    if meta['config']['validation_video'] != config['validation_video'] or meta['config']['calibration_video'] != config['calibration_video']:
        raise ValueError('OOF producer excluded videos differ')
    if meta['guard']['sources']['cfp/predictive.py'] != sha256('cfp/predictive.py'):
        raise ValueError('Ordinary inference implementation changed')
    model = LatentIdentityPredictor(hidden=meta['config']['hidden']).eval()
    export = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not export['teacher_removed'] or 'teacher' in export or 'optimizer' in export:
        raise ValueError('Only ordinary inference state allowed')
    model.load_state_dict(export['model'], strict=True)
    torch.set_num_threads(2)
    native = load_bot_reference('/data3/liuyeqiang/CFP-FishMOT_references/BoT-SORT')
    report = dict(status='RUNNING', association='gru', partition=partition, seed=seed, probe=False, predictor_checkpoint=str(checkpoint), producer_excluded_videos=[config['calibration_video'], config['validation_video']], config=config, config_sha256=sha256(args.config), videos={},
        checkpoint_sha256=sha256(checkpoint), detection_manifest_sha256=sha256(detection_root / 'manifest.json'),
        feature_manifest_sha256=sha256(features_root / 'manifest.json'),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_snapshots=snapshot_sources(args.output, [__file__, args.config, 'cfp/ambiguity_association.py',
            'cfp/hypothesis_bank.py', 'cfp/online_recovery.py', 'cfp/recoverability.py',
            'cfp/candidate_graph.py', 'cfp/lifecycle_crossover.py']),
        scope='Strict current/past fixed-cache replay, not end-to-end FPS. New-module validation '
              'videos exclude fitting/calibration; historical frozen GRU/encoder exposure remains. '
              'K2 heuristic branch score is not a probability; no historical output revision.')
    write_json(args.output / 'manifest.json', report)
    for seq in config.get('sequences', protocol[partition]):
        if seq not in protocol[partition]:
            raise ValueError('Sequence outside allowed partition')
        dp, fp = detection_root / (seq + '.npz'), features_root / (seq + '.npz')
        provenance = json.loads(fp.with_suffix('.provenance.json').read_text())
        if sha256(dp) != dm['caches'][seq]['sha256'] or sha256(fp) != provenance['features_sha256']:
            raise ValueError('Frozen input changed')
        with np.load(dp, allow_pickle=False) as archive:
            frames, offsets, boxes, scores = [archive[key] for key in ('frame_ids', 'offsets', 'boxes', 'scores')]
        with np.load(fp, allow_pickle=False) as archive:
            features = archive['rgb']
        BaseTrack._count = 0
        write_policy = dict(minimum_confidence=.85, minimum_row_margin=.05, minimum_column_margin=.05) if config.get('protected_write') else None
        tracker = make_cfp_lifecycle(native, 'gru', model, .3, write_policy=write_policy)
        estimator, calibration, risk_hash = None, None, None
        policy = config.get('risk_policy', 'disabled')
        if policy.startswith('risk_'):
            path = Path(config['risk_models'][seq]) / 'inference.pt'
            rm = json.loads(path.with_name('manifest.json').read_text())
            if rm['status'] != 'COMPLETE' or sha256(path) != rm['inference_sha256'] or rm['config']['validation_video'] != seq:
                raise ValueError('Risk model must hold out this whole replay video')
            for source in ['cfp/recoverability.py', 'cfp/selective_metrics.py']:
                if sha256(source) != rm['guard']['sources'][source]:
                    raise ValueError('Risk model source changed')
            saved = torch.load(path, map_location='cpu', weights_only=False)
            estimator = RecoverabilityEstimator(rm['config']['hidden'], rm['config']['candidate_context'], rm['config']['history_evidence']).eval()
            estimator.load_state_dict(saved['model'], strict=True)
            calibration, risk_hash = saved['calibration'], sha256(path)
        if config['K'] == 1:
            tracker = install_risk_association(tracker, estimator, calibration, policy, config.get('threshold', .2))
        bank = AmbiguityAssociation(tracker, config['K'], config.get('ambiguity_delta', .05), config.get('discount', .9))
        times, rows = [], 0
        recorder = DecisionRecorder(args.output / (seq + '.decisions.jsonl.gz'), tracker)
        with (args.output / (seq + '.txt')).open('w') as stream:
            for i, frame in enumerate(frames):
                lo, hi = offsets[i:i + 2]
                current = np.c_[boxes[lo:hi].copy(), scores[lo:hi]]
                start = time.perf_counter()
                with recorder.capture([matching, native.matching]):
                    output = bank.update(current, features[lo:hi].copy(), np.arange(lo, hi))
                for track in output:
                    if np.prod(track.tlwh[2:]) <= 100:
                        continue
                    index = track.current_proposal_index
                    if track.frame_id != frame or not lo <= index < hi:
                        raise ValueError('Only actual current matching boxes may be output')
                    measured = boxes[index].copy(); measured[2:] -= measured[:2]
                    stream.write(f'{frame},{int(track.track_id)},{measured[0]:.3f},{measured[1]:.3f},{measured[2]:.3f},{measured[3]:.3f},{float(track.score):.6f},-1,-1,-1\n')
                    rows += 1
                times.append(time.perf_counter() - start)
                if i % 128 == 0 and shutil.disk_usage(root).free < 40 * 2**30:
                    raise OSError('40 GiB reserve')
        recorder.close()
        reference = root / 'phase5' / (f'cross_train_gru_cfp_s{seed}_c1_v1' if partition == 'train' else f'cross_gru_cfp_s{seed}_c1_v1') / (seq + '.measurement.txt')
        parity = None
        if config.get('expect_legacy_parity', False):
            parity = (args.output / (seq + '.txt')).read_bytes() == reference.read_bytes()
            if not parity:
                raise ValueError('Disabled new modules failed full baseline byte parity')
        report['videos'][seq] = dict(frames=len(times), rows=rows, prediction_sha256=sha256(args.output / (seq + '.txt')),
            association_seconds=sum(times), latency_p50_p95_ms=(np.quantile(times, [.5, .95]) * 1000).tolist(),
            branch_statistics=bank.statistics, baseline_byte_equal=parity, risk_checkpoint_sha256=risk_hash, decisions_sha256=sha256(args.output / (seq + '.decisions.jsonl.gz')),
            risk_decisions=getattr(bank.tracker, 'phase_vi_risk_counts', None), write_counts=bank.tracker.write_counts)
        write_json(args.output / 'manifest.json', report)
        print(seq, report['videos'][seq], flush=True)
    report['status'] = 'COMPLETE'
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
