#!/usr/bin/env python3
"""Registered causal A/B/C combinations on the same excluded-video producer."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from cfp.ambiguity_association_stable import StableAmbiguityAssociation
from cfp.hypothesis_diagnostics import HypothesisBirthAudit, birth_key
from cfp.online_recovery_write_quota import install_quota_risk_memory, ObservableWriteRisk
from cfp.lifecycle_crossover import load_bot_reference, make_cfp_lifecycle
from cfp.online_candidate_relation import install_candidate_relation, RawMeanCandidateControl
from cfp.candidate_relation_jepa import CandidateRelationStudent
from cfp.candidate_relation_residual import ResidualRelationStudent
from cfp.predictive import LatentIdentityPredictor
from cfp.provenance import sha256, snapshot_sources, write_json
from cfp.recoverability_ablation import risk_model_from_config
from yolox.tracker.basetrack import BaseTrack


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if config['state'] != 'FROZEN' or args.output.exists() or config['partition'] != 'train' or config['K'] not in (1, 2):
        raise ValueError('Frozen train-only K1/K2 config and immutable output required')
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
        raise ValueError('Ordinary excluded-video producer weight drift')
    heldout = config['sequences'][0]
    if len(config['sequences']) != 1 or meta['config']['validation_video'] != heldout:
        raise ValueError('Exactly one outer video per fold evaluation')
    excluded = {heldout, meta['config']['calibration_video']}
    if set(meta['config']['fit_videos']) & excluded:
        raise ValueError('Producer video leakage')
    if fm['rgb_encoder']['checkpoint_sha256'] != meta['rgb_encoder']['checkpoint_sha256']:
        raise ValueError('Frozen encoder coordinate-space drift')
    model = LatentIdentityPredictor(hidden=meta['config']['hidden']).eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    torch.set_num_threads(2)
    native = load_bot_reference('/data3/liuyeqiang/CFP-FishMOT_references/BoT-SORT')
    report = dict(status='RUNNING', config=config, config_sha256=sha256(args.config), videos={},
        checkpoint_sha256=sha256(checkpoint), detection_manifest_sha256=sha256(detection_root / 'manifest.json'),
        feature_manifest_sha256=sha256(features_root / 'manifest.json'),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_snapshots=snapshot_sources(args.output, [__file__, args.config, 'cfp/ambiguity_association.py',
            'cfp/hypothesis_bank.py', 'cfp/hypothesis_diagnostics.py', 'cfp/ambiguity_association_stable.py', 'cfp/online_recovery_write_quota.py', 'cfp/online_recovery_write_isolated.py', 'cfp/online_recovery.py', 'cfp/recoverability.py',
            'cfp/candidate_graph.py', 'cfp/lifecycle_crossover.py', 'cfp/recoverability_ablation.py', 'cfp/online_candidate_relation.py', 'cfp/candidate_relation_jepa.py', 'cfp/candidate_relation_residual.py']),
        scope='Strict current/past fixed-cache replay, not end-to-end FPS. New-module validation '
              'videos exclude fitting/calibration; state producer excludes this outer/calibration video; frozen encoder/detector exposure remains. '
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
        tracker = make_cfp_lifecycle(native, 'gru', model, .3)
        student_hash, risk_hash = None, None
        if config['candidate_prediction']:
            path = Path(config['student_models'][seq]) / 'inference.pt'
            sm = json.loads(path.with_name('manifest.json').read_text())
            if sm['status'] != 'COMPLETE' or sha256(path) != sm['inference_sha256']:
                raise ValueError('Incomplete or changed residual export')
            if {sm['config']['validation_video'], sm['config']['calibration_video']} != excluded or set(sm['config']['fit_videos']) & excluded:
                raise ValueError('Residual student/producer isolation differs')
            qm = json.loads((Path(sm['config']['queries']) / 'manifest.json').read_text())
            if qm['oof_producer_checkpoint_sha256'] != sha256(checkpoint):
                raise ValueError('Residual actual-state producer differs')
            for original, digest in sm['guard']['sources'].items():
                if sha256(original) != digest:
                    raise ValueError('Residual source drift: ' + original)
            saved = torch.load(path, map_location='cpu', weights_only=False)
            if not saved['teacher_removed'] or 'teacher' in saved or saved['rgb_encoder']['checkpoint_sha256'] != fm['rgb_encoder']['checkpoint_sha256']:
                raise ValueError('Only matching-coordinate student may run online')
            student = ResidualRelationStudent(latent=sm['config']['latent'], scale=sm['config']['residual_scale']).eval()
            student.load_state_dict(saved['model'], strict=True)
            tracker = install_candidate_relation(tracker, student, use_null=False, residual=True)
            student_hash = sha256(path)
        if config['risk_write_quota'] and config.get('risk_kind', 'learned') == 'learned':
            path = Path(config['risk_models'][seq]) / 'inference.pt'
            rm = json.loads(path.with_name('manifest.json').read_text())
            if rm['status'] != 'COMPLETE' or sha256(path) != rm['inference_sha256']:
                raise ValueError('Incomplete or changed risk export')
            if {rm['config']['validation_video'], rm['config']['calibration_video']} != excluded or set(rm['config']['fit_videos']) & excluded:
                raise ValueError('Risk model/producer isolation differs')
            qm = json.loads((Path(rm['config']['queries']) / 'manifest.json').read_text())
            if qm['oof_producer_checkpoint_sha256'] != sha256(checkpoint):
                raise ValueError('Risk actual-state producer differs')
            for original, digest in rm['guard']['sources'].items():
                if sha256(original) != digest:
                    raise ValueError('Risk source drift: ' + original)
            saved = torch.load(path, map_location='cpu', weights_only=False)
            estimator = risk_model_from_config(rm['config']).eval()
            estimator.load_state_dict(saved['model'], strict=True)
            tracker = install_quota_risk_memory(tracker, estimator, saved['calibration'], fraction=config['reject_fraction'], seed=seed)
            risk_hash = sha256(path)
        if config['risk_write_quota'] and config.get('risk_kind') == 'margin':
            tracker = install_quota_risk_memory(tracker, ObservableWriteRisk('margin').eval(), dict(temperature=1., bias=0.), fraction=config['reject_fraction'], seed=seed)
        if config.get('risk_kind', 'learned') not in ('learned', 'margin'):
            raise ValueError('Registered risk information control required')
        bank = StableAmbiguityAssociation(tracker, config['K'], config['ambiguity_delta'], config['discount'], config['switch_margin'], config['confirmation'])
        times, rows = [], 0
        birth_audit = HypothesisBirthAudit()
        with (args.output / (seq + '.txt')).open('x') as stream, (args.output / (seq + '.hypotheses.jsonl')).open('x') as events:
            for i, frame in enumerate(frames):
                lo, hi = offsets[i:i + 2]
                current = np.c_[boxes[lo:hi].copy(), scores[lo:hi]]
                start = time.perf_counter()
                output = bank.update(current, features[lo:hi].copy(), np.arange(lo, hi))
                birth_audit.observe(int(frame), bank.bank.hypotheses, output)
                # No GT/labels; save emitted explanations for retrospective audit.
                events.write(json.dumps(dict(frame=int(frame), winner=bank.last_winner,
                    hypotheses=[dict(branch=h.branch_id, score=h.score,
                        current=[dict(identity=int(t.track_id), proposal=int(t.current_proposal_index), birth=birth_key(t))
                                 for t in getattr(h.state, '_phase_vi_current_output', output if h.state is bank.tracker else [])
                                 if t.frame_id == frame and lo <= t.current_proposal_index < hi])
                        for h in bank.bank.hypotheses])) + '\n')
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
        parity = None
        if not config['candidate_prediction'] and not config['risk_write_quota'] and config['K'] == 1:
            reference = root / 'phase6' / f'oof_states_holdout_{seq}_s0_v1' / (seq + '.txt')
            parity = (args.output / (seq + '.txt')).read_bytes() == reference.read_bytes()
            if not parity:
                raise ValueError('All-disabled complete output parity failed')
        report['videos'][seq] = dict(frames=len(times), rows=rows, prediction_sha256=sha256(args.output / (seq + '.txt')),
            association_seconds=sum(times), latency_p50_p95_ms=(np.quantile(times, [.5, .95]) * 1000).tolist(),
            branch_statistics=bank.statistics, baseline_byte_equal=parity, student_checkpoint_sha256=student_hash, risk_checkpoint_sha256=risk_hash,
            risk_decisions=getattr(bank.tracker, 'phase_vi_memory_counts', None), write_counts=bank.tracker.write_counts, relation_counts=getattr(bank.tracker, "relation_counts", None), quota_counts=getattr(bank.tracker, "quota_counts", None), birth_audit=birth_audit.report(), hypothesis_trace_sha256=sha256(args.output / (seq + ".hypotheses.jsonl")))
        write_json(args.output / 'manifest.json', report)
        print(seq, report['videos'][seq], flush=True)
    report['status'] = 'COMPLETE'
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
