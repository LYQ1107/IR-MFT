#!/usr/bin/env python3
"""Observe exact inner native trajectories, then diagnose labels offline only."""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch
from cfp.centered_joint_identity import CenteredJointIdentity
from cfp.joint_identity_decoder import decode_joint_assignment
from cfp.lifecycle_crossover import load_bot_reference, make_cfp_lifecycle, DecisionRecorder
from cfp.online_centered_joint_identity import install_centered_joint_identity
from cfp.predictive import LatentIdentityPredictor, bounded_identity_prototype
from cfp.provenance import sha256, snapshot_sources, write_json
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker import matching


class ObservedJoint(CenteredJointIdentity):
    """Read-only post-forward observer; original output and RNG untouched."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observations = []
        self.sparse_inputs = []

    def forward(self, **inputs):
        output = super().forward(**inputs)
        frame = int(inputs['frame'][0])
        with torch.inference_mode():
            fifo, available = bounded_identity_prototype(inputs['history'], inputs['times'], inputs['valid'], inputs['frame'], mode='fifo')
            last, _ = bounded_identity_prototype(inputs['history'], inputs['times'], inputs['valid'], inputs['frame'], mode='last')
            current = inputs['detections']
            scores = output['logits'].detach().cpu().numpy()
            self.observations.append(dict(frame=frame,
                logits=[[float(x) if np.isfinite(x) else None for x in row] for row in scores],
                selected=decode_joint_assignment(output['logits'], inputs['detection_valid']).cpu().tolist(),
                fifo_cost=((1-fifo@current.T)/2).clamp(0, 1).cpu().tolist(),
                last_cost=((1-last@current.T)/2).clamp(0, 1).cpu().tolist(),
                gru_cost=inputs['appearance'].cpu().tolist(), geometry=inputs['geometry'].cpu().tolist(),
                history_available=available.cpu().tolist(), codes=inputs['codes'].cpu().tolist()))
            if (frame-1) % 5 == 0:
                self.sparse_inputs.append(dict(frame=frame, inputs={k: v.detach().cpu().clone() for k, v in inputs.items()},
                                              logits=output['logits'].detach().cpu().clone()))
        return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--video', required=True)
    a = ap.parse_args()
    config_path = Path('configs/cfp/phase_vii_centered_joint_probe_v2.json')
    config = json.loads(config_path.read_text())
    if a.video not in config['calibration_videos'] or a.video in config['fit_videos']+config['outer_videos']:
        raise ValueError('Reject outer/fitting video before any weights or banks are loaded')
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd().resolve()) or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('New outsideGit CPU-only inner observation required')
    if shutil.disk_usage(a.root).free < 42*2**30:
        raise OSError('Space reserve')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    assets = a.root.parent
    detections = assets/'phase0/causal_v3_train_detections'
    features = assets/'phase2/causal_train_detection_features_postneck_seed0'
    dm, fm = [json.loads((p/'manifest.json').read_text()) for p in [detections, features]]
    dp, fp = detections/(a.video+'.npz'), features/(a.video+'.npz')
    provenance = json.loads(fp.with_suffix('.provenance.json').read_text())
    if dm['status'] != 'COMPLETE' or fm['status'] != 'COMPLETE' or sha256(dp) != dm['caches'][a.video]['sha256'] or sha256(fp) != provenance['features_sha256'] or provenance['detection_sha256'] != sha256(dp):
        raise ValueError('Exact frozen native current input contract required')
    with np.load(dp, allow_pickle=False) as bank:
        frames, offsets, boxes, scores = [bank[k].copy() for k in ['frame_ids', 'offsets', 'boxes', 'scores']]
    with np.load(fp, allow_pickle=False) as bank:
        rgb = bank['rgb'].copy()
    if not np.array_equal(frames, np.arange(1, len(frames)+1)) or len(rgb) != len(boxes):
        raise ValueError('Complete current-frame bank alignment required')
    checkpoint = assets/'phase3/predictor_v3_gru_association_anchor_seed0/epoch8.pt'
    pm = json.loads(checkpoint.with_name('manifest.json').read_text())
    if sha256(checkpoint) != pm['checkpoint_sha256']:
        raise ValueError('Exact inherited predictor required')
    predictor = LatentIdentityPredictor().eval()
    predictor.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    native = load_bot_reference('/data3/liuyeqiang/CFP-FishMOT_references/BoT-SORT')
    sources = [__file__, 'cfp/joint_endogenous_diagnosis.py', str(config_path), 'cfp/centered_joint_identity.py',
               'cfp/joint_identity_decoder.py', 'cfp/lifecycle_crossover.py', 'cfp/online_centered_joint_identity.py',
               'cfp/predictive.py', 'cfp/predictive_association.py', 'cfp/history.py', 'cfp/association.py',
               'cfp/recoverability_diagnosis.py', 'cfp/recovery_supervision.py', 'cfp/joint_identity_supervision.py']
    a.output.mkdir()
    report = dict(status='RUNNING_GT_FREE_NATIVE_OBSERVATION', pid=os.getpid(), video=a.video, frames=len(frames), runs={},
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  config_sha256=sha256(config_path), detection_sha256=sha256(dp), feature_sha256=sha256(fp),
                  predictor_sha256=sha256(checkpoint), source_snapshots=snapshot_sources(a.output/'sources', sources),
                  scope='Inner only, nine full684frame exact existing native outputs. Read-only model/decision observation precedes ALL physical-label loading. Later physical labels/EarlyReference are offline diagnostics, never decision inputs. Conditional inherited512encoder/GRU exposure; no new model/training/MOT superiority or rawFPS claim.')
    write_json(a.output/'manifest.json', report)
    for architecture in config['architectures']:
        for seed in config['seed_plan']:
            name = f'{architecture}_s{seed}'
            fit = a.root/f'centered_joint_probe_{architecture}_s{seed}_v2'
            original = a.root/f'centered_joint_native_{architecture}_full_s{seed}_v2'
            m = json.loads((fit/'manifest.json').read_text())
            om = json.loads((original/'manifest.json').read_text())
            if (m['status'] != 'COMPLETE' or om['status'] != 'COMPLETE' or m['guard']['architecture'] != architecture
                    or m['guard']['seed'] != seed or m['guard']['config_sha256'] != sha256(config_path)
                    or sha256(fit/'epoch8.pt') != m['checkpoint_sha256']
                    or om['joint_checkpoint_sha256'] != m['checkpoint_sha256']
                    or any(sha256(p) != h for p, h in m['guard']['source_sha256'].items())):
                raise ValueError('Exact fixed-final matching architecture/seed/native completion required')
            model = ObservedJoint(config['dimension'], config['hidden'], config['heads'], config['capacity'], architecture).eval()
            model.load_state_dict(torch.load(fit/'epoch8.pt', map_location='cpu', weights_only=False)['model'])
            BaseTrack._count = 0
            tracker = install_centered_joint_identity(make_cfp_lifecycle(native, 'gru', predictor, .3), model, 'full', .03)
            folder = a.output/name; folder.mkdir()
            recorder = DecisionRecorder(folder/'decisions.jsonl.gz', tracker)
            prediction = folder/(a.video+'.txt')
            try:
                with prediction.open('w') as stream:
                    for i, frame in enumerate(frames):
                        lo, hi = offsets[i:i+2]
                        with recorder.capture([matching]):
                            output = tracker.update(np.c_[boxes[lo:hi].copy(), scores[lo:hi]], rgb[lo:hi].copy(), np.arange(lo, hi))
                        for track in output:
                            if np.prod(track.tlwh[2:]) <= 100:
                                continue
                            index = track.current_proposal_index
                            if track.frame_id != frame or not lo <= index < hi:
                                raise ValueError('Observed native output lost current proposal mapping')
                            measured = boxes[index].copy(); measured[2:] -= measured[:2]
                            stream.write(f'{frame},{int(track.track_id)},{measured[0]:.3f},{measured[1]:.3f},{measured[2]:.3f},{measured[3]:.3f},{float(track.score):.6f},-1,-1,-1\n')
            finally:
                recorder.close()
            if sha256(prediction) != om['videos'][a.video]['prediction_sha256'] or prediction.read_bytes() != (original/(a.video+'.txt')).read_bytes():
                raise ValueError('Read-only observation altered actual full native outputs')
            with gzip.open(folder/'model_observations.jsonl.gz', 'wt') as stream:
                for row in model.observations:
                    stream.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
            torch.save(model.sparse_inputs, folder/'sparse_observable_inputs.pt')
            report['runs'][name] = dict(architecture=architecture, seed=seed, frames=len(frames),
                prediction_byte_equal=True, prediction_sha256=sha256(prediction),
                original_native_manifest_sha256=sha256(original/'manifest.json'), model_checkpoint_sha256=m['checkpoint_sha256'],
                decisions_sha256=sha256(folder/'decisions.jsonl.gz'), model_observations_sha256=sha256(folder/'model_observations.jsonl.gz'),
                sparse_inputs_sha256=sha256(folder/'sparse_observable_inputs.pt'), sparse_frames=len(model.sparse_inputs),
                online_physical_labels_loaded=False)
            write_json(a.output/'manifest.json', report)
            print(json.dumps(dict(name=name, prediction_parity=True, observed_frames=len(model.observations))), flush=True)
    report['status'] = 'ALL_NINE_GT_FREE_NATIVE_OBSERVATIONS_COMPLETE'
    write_json(a.output/'manifest.json', report)
    # The label bank and offline analysis module are loaded only after all nine outputs/parities are fixed.
    from cfp.recovery_supervision import physical_bank
    from cfp.joint_endogenous_diagnosis import diagnose_native_rows
    physical = physical_bank(a.video, 'train')
    if physical['detection_sha256'] != report['detection_sha256']:
        raise ValueError('Offline physical source differs from observed proposals')
    report['offline_annotation_sha256'] = physical['annotation_sha256']
    for name, entry in report['runs'].items():
        folder = a.output/name
        entry['offline_diagnosis'] = diagnose_native_rows(folder, physical, a.root/f"centered_joint_probe_{entry['architecture']}_s{entry['seed']}_v2", config)
        write_json(a.output/'manifest.json', report)
    report['status'] = 'COMPLETE_ALL_NINE_INNER_ENDOGENOUS_IDENTITY_DIAGNOSES'
    write_json(a.output/'manifest.json', report)


if __name__ == '__main__':
    main()
