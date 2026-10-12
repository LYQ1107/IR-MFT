#!/usr/bin/env python3
"""Stream current RGB frames; keep only MOT rows and small timing/provenance files.

This entry point never opens annotations or uses cached identity features.
Optional frozen proposals are Protocol A; otherwise the detector runs per frame.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from cfp.online import CurrentFrameFrontend
from cfp.predictive import LatentIdentityPredictor
from cfp.predictive_association import PredictiveAssociationTracker
from cfp.joint_association import JointAssociationTracker
from cfp.reliable_writes import ReliableWritesMixin
from cfp.utility import UtilityMLP
from cfp.provenance import sha256, write_json, snapshot_sources
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.byte_tracker import BYTETracker


class ReliableOnlinePredictiveTracker(ReliableWritesMixin, PredictiveAssociationTracker):
    pass


class ReliableOnlineJointTracker(ReliableWritesMixin, JointAssociationTracker):
    pass


def storage_guard(path, reserve_gib, starting_free, budget_gib):
    free = shutil.disk_usage(path).free
    if free < reserve_gib * 2**30 or starting_free - free > budget_gib * 2**30:
        raise OSError('Storage reserve/output budget reached; preserve partial evidence and stop')
    return free


def load_predictor(path, protocol, execution, rgb_checkpoint):
    metadata = json.loads((path.parent / 'manifest.json').read_text())
    if (metadata['status'] != 'COMPLETE' or metadata['protocol_sha256'] != sha256(protocol)
            or metadata['training_config_sha256'] != execution['training_config_sha256']
            or metadata['checkpoint_sha256'] != sha256(path)):
        raise ValueError('Online predictor differs from the declared completed trial')
    if execution.get('history_view', 'recent') == 'anchor_recent' and metadata['history'] != 'anchor_recent':
        raise ValueError('Predictor history context differs')
    feature_manifest = Path(metadata['features_root']) / 'manifest.json'
    feature_meta = json.loads(feature_manifest.read_text())
    if sha256(feature_manifest) != metadata['feature_manifest_sha256'] or feature_meta['rgb_encoder']['checkpoint_sha256'] != sha256(rgb_checkpoint):
        raise ValueError('Predictor and online encoder use different latent coordinates')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['epoch'] != 8:
        raise ValueError('Online predictor requires the fixed epoch8 checkpoint')
    model = LatentIdentityPredictor(512, metadata['architecture'], time_encoding=metadata.get('time_encoding', 'linear_clipped'))
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.eval(), metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fixed-detections', type=Path)
    p.add_argument('--compare-ir-root', type=Path, help='Diagnostic image comparison only; never an inference input')
    p.add_argument('--precomputed-causal-ir', action='store_true', help='Protocol A only: current paired causal image, generator cost excluded')
    p.add_argument('--max-frames', type=int, default=0, help='Per-video prefix; nonzero runs cannot be final evaluation')
    p.add_argument('--reserve-gib', type=float, default=30.)
    p.add_argument('--output-budget-gib', type=float, default=2.)
    p.add_argument('--gpu-memory-fraction', type=float, default=.75)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    protocol = json.loads(args.protocol.read_text())
    if config['state'] != 'FROZEN' or config['protocol_sha256'] != sha256(args.protocol) or protocol['state'] != 'FROZEN':
        raise ValueError('A frozen, matching development configuration is required')
    if set(config['sequences']) != set(protocol['dev']) or args.max_frames < 0:
        raise ValueError('Online integration trial is canonical dev only')
    if args.output.exists():
        raise FileExistsError('Online evidence is immutable')
    if not 0 < args.gpu_memory_fraction <= 1 or torch.cuda.device_count() != 1:
        raise ValueError('Use one explicitly visible GPU and a valid allocation cap')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    starting_free = shutil.disk_usage(args.output.parent).free
    storage_guard(args.output.parent, args.reserve_gib, starting_free, args.output_budget_gib)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    execution = json.loads(Path(config['execution_config']).read_text())
    if execution['state'] != 'FROZEN' or sha256(config['execution_config']) != config['execution_config_sha256']:
        raise ValueError('Frozen association execution changed')
    predictor = predictor_metadata = None
    if config['memory'] == 'predictor':
        predictor, predictor_metadata = load_predictor(Path(config['predictor_checkpoint']), args.protocol, execution, config['rgb_checkpoint'])
    if config['memory'] not in ('bytetrack', 'predictor', 'last', 'ema', 'fifo', 'all_history_ema', 'constant_zero', 'constant_half', 'anchor_only'):
        raise ValueError('Unknown online association mode')
    utility_kwargs = {}
    if config.get('utility_kind'):
        uc_path, ue_path = Path(config['utility_config']), Path(config['utility_execution_config'])
        uc, ue = json.loads(uc_path.read_text()), json.loads(ue_path.read_text())
        if uc['state'] != 'FROZEN' or ue['state'] != 'FROZEN' or ue['parent_target_config'] != uc_path.name:
            raise ValueError('Frozen online utility target/execution differs')
        gate = calibration = None
        if config['utility_kind'] in ('cmue', 'ordinary', 'quality'):
            ck = torch.load(config['utility_checkpoint'], map_location='cpu', weights_only=False)
            rgb_meta = torch.load(config['rgb_checkpoint'], map_location='cpu', weights_only=False)
            if ck['epoch'] != uc['epochs'] or ck['kind'] != config['utility_kind'] or ck['seed'] != rgb_meta['seed'] or ck['metadata']['config_sha256'] != sha256(uc_path) or ck['metadata']['protocol_sha256'] != sha256(args.protocol):
                raise ValueError('Online gate checkpoint differs from declared seed/protocol')
            del rgb_meta
            gate = UtilityMLP(uc['mlp']['hidden'], quality_only=config['utility_kind'] == 'quality').eval()
            gate.load_state_dict(ck['model'], strict=True)
            calibration = ck['calibration']
        utility_kwargs = dict(utility_model=gate, utility_kind=config['utility_kind'], calibration=calibration,
                              utility_threshold=ue['threshold'], utility_context_version=uc.get('candidate_context_version', 1),
                              teacher_association_weight=uc.get('teacher_association_weight', .3))
        if not config.get('ir_checkpoint'):
            raise ValueError('Joint online IR utility needs its matching current-frame encoder')
    policy = None
    if config.get('write_policy_config'):
        policy = json.loads(Path(config['write_policy_config']).read_text())
        if policy['state'] != 'FROZEN' or policy['parent_execution_sha256'] != sha256(config['execution_config']) or policy['appearance_weight'] != config['appearance_weight']:
            raise ValueError('Online write toggle changed')
    if args.precomputed_causal_ir:
        if not args.fixed_detections:
            raise ValueError('Precomputed causal IR is restricted to Protocol A diagnostics')
        gen = json.loads((args.data_root / 'manifest.json').read_text())
        if gen['status'] != 'COMPLETE' or gen.get('probe_frames', 0) or gen['config']['mode'] != 'temporal_ema' or gen['config']['alpha'] != .01:
            raise ValueError('Only complete pinned causal inputs may bypass generation')
    frontend = CurrentFrameFrontend(config['detector_checkpoint'], config['scripts_root'],
                                    rgb_checkpoint=config.get('rgb_checkpoint'), ir_checkpoint=config.get('ir_checkpoint'),
                                    detector_batch=config['detector_batch'], protocol_sha256=sha256(args.protocol),
                                    input_regime='causal_precomputed' if args.precomputed_causal_ir else 'causal',
                                    preprocessing=config.get('preprocessing', 'original'))
    args.output.mkdir()
    source_files = [__file__, 'cfp/online.py', 'cfp/pseudo_ir.py', 'cfp/reid.py', 'cfp/identity_data.py',
                    'cfp/predictive.py', 'cfp/predictive_association.py', 'cfp/association.py', 'cfp/history.py',
                    args.config, args.protocol, config['execution_config']]
    if config.get('utility_kind'):
        source_files += ['cfp/joint_association.py', 'cfp/utility.py', 'cfp/utility_association.py', 'cfp/candidate_context.py', config['utility_config'], config['utility_execution_config']]
    if policy:
        source_files += ['cfp/reliable_writes.py', config['write_policy_config']]
    report = dict(status='RUNNING', partition='dev', config=config, config_sha256=sha256(args.config),
                  protocol_sha256=sha256(args.protocol), probe=bool(args.max_frames),
                  protocol='A/current-frame features' if args.fixed_detections else 'B/current-frame complete pipeline',
                  rgb_encoder=frontend.rgb_metadata, ir_encoder=frontend.ir_metadata,
                  predictor_metadata=predictor_metadata,
                  torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                  physical_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu_memory_fraction=args.gpu_memory_fraction,
                  device=torch.cuda.get_device_name(), git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_snapshots=snapshot_sources(args.output, source_files), videos={},
                  parameters=dict(detector=sum(v.numel() for v in frontend.detector.parameters()),
                                  rgb_encoder=sum(v.numel() for v in frontend.rgb_encoder.parameters()) if frontend.rgb_encoder else 0,
                                  ir_encoder=sum(v.numel() for v in frontend.ir_encoder.parameters()) if frontend.ir_encoder else 0,
                                  predictor=sum(v.numel() for v in predictor.parameters()) if predictor else 0),
                  storage=dict(starting_free_bytes=starting_free, reserve_gib=args.reserve_gib, budget_gib=args.output_budget_gib,
                               policy='No generated images, feature banks, model copies or videos; stream MOT/timing only'),
                  scope='Current/past RGB only, no annotation reads. Historical V3 detector exposure to dev disclosed. Padded batches contain this frame only. Shared-load timing is not a dedicated deployment benchmark.')
    report['precomputed_causal_ir'] = args.precomputed_causal_ir
    if args.precomputed_causal_ir:
        report['generation_manifest_sha256'] = sha256(args.data_root / 'manifest.json')
        report['scope'] += ' Protocol A precomputed causal pixels: generator and detector costs excluded; FPS is NOT end-to-end.'
    write_json(args.output / 'manifest.json', report)
    dm = None
    if args.fixed_detections:
        dm = json.loads((args.fixed_detections / 'manifest.json').read_text())
        if dm['status'] != 'COMPLETE' or dm['probe'] or dm['protocol_sha256'] != sha256(args.protocol) or dm['checkpoint_sha256'] != sha256(config['detector_checkpoint']):
            raise ValueError('Protocol A cache differs')
        if args.precomputed_causal_ir and dm['config']['generation_manifest_sha256'] != sha256(args.data_root / 'manifest.json'):
            raise ValueError('Protocol A current causal images differ from frozen detector inputs')
        report['detection_manifest_sha256'] = sha256(args.fixed_detections / 'manifest.json')
    for sequence in config['sequences']:
        frontend.reset()
        BaseTrack._count = 0
        g = execution['geometry']
        tracker_args = SimpleNamespace(track_thresh=g['track_thresh'], track_buffer=g['track_buffer'], match_thresh=g['match_thresh'], mot20=False)
        tracker_type = ReliableOnlineJointTracker if utility_kwargs else ReliableOnlinePredictiveTracker
        tracker = BYTETracker(tracker_args, frame_rate=g['frame_rate']) if config['memory'] == 'bytetrack' else tracker_type(
            tracker_args, predictor=predictor, memory=config['memory'], history_view=execution.get('history_view', 'recent'),
            appearance_weight=config['appearance_weight'], frame_rate=g['frame_rate'], ema=execution['ema'], proximity_iou=execution['proximity_iou'], write_policy=policy, **utility_kwargs)
        paths = sorted((args.data_root / 'train' / sequence / 'visible').glob('*.jpg'))
        if not paths or any(path.stem != f'{i:06d}' for i, path in enumerate(paths, 1)):
            raise ValueError('Missing/nonconsecutive current-frame sequence')
        if args.max_frames:
            paths = paths[:args.max_frames]
        boxes = scores = offsets = None
        if dm:
            dp = args.fixed_detections / (sequence + '.npz')
            if sha256(dp) != dm['caches'][sequence]['sha256']:
                raise ValueError('Fixed proposals changed')
            with np.load(dp) as a:
                boxes, scores, offsets, frame_ids = [a[k] for k in ('boxes', 'scores', 'offsets', 'frame_ids')]
            if len(frame_ids) < len(paths) or not np.array_equal(frame_ids, np.arange(1, len(frame_ids) + 1)):
                raise ValueError('Frozen proposals have different frame order')
        totals, stage_rows, count, ir_mismatches, ir_comparisons = [], {}, 0, [], 0
        pixel_hash = hashlib.sha256()
        prediction = args.output / (sequence + '.txt')
        with prediction.open('x') as output, (args.output / (sequence + '_timings.jsonl')).open('x') as timing:
            for frame, path in enumerate(paths, 1):
                if (frame - 1) % 128 == 0:
                    storage_guard(args.output, args.reserve_gib, starting_free, args.output_budget_gib)
                start = time.perf_counter()
                rgb = cv2.imread(str(path))
                if rgb is None:
                    raise ValueError('Current RGB decode failed')
                decode_seconds = time.perf_counter() - start
                provided_ir = None
                if args.precomputed_causal_ir:
                    t = time.perf_counter()
                    provided_ir = cv2.imread(str(args.data_root / 'train' / sequence / 'infrared' / path.name))
                    if provided_ir is None:
                        raise ValueError('Missing current causal response image')
                    decode_seconds += time.perf_counter() - t
                fixed = None
                if dm:
                    lo, hi = offsets[frame - 1:frame + 1]
                    fixed = (boxes[lo:hi].copy(), scores[lo:hi].copy())
                result = frontend.process(rgb, provided_ir=provided_ir, fixed_proposals=fixed, ir_intervention=config.get('ir_intervention', 'none'))
                stages = dict(decode=decode_seconds, **result['stage_seconds'])
                t = time.perf_counter()
                detections = np.concatenate([result['boxes'], result['scores'][:, None]], axis=1)
                if config['memory'] == 'bytetrack':
                    targets = tracker.update(detections, (1, 1), (1, 1))
                else:
                    targets = tracker.update(detections, (1, 1), (1, 1), result['rgb_features'], result['ir_features'])
                stages['association'] = time.perf_counter() - t
                t = time.perf_counter()
                for target in targets:
                    x, y, w, h = target.tlwh
                    if w * h > g['min_box_area']:
                        output.write(f'{frame},{target.track_id},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{float(target.score):.6f},-1,-1,-1\n')
                        count += 1
                stages['output'] = time.perf_counter() - t
                elapsed = time.perf_counter() - start
                totals.append(elapsed)
                for name, value in stages.items():
                    stage_rows.setdefault(name, []).append(value)
                timing.write(json.dumps(dict(frame=frame, total_seconds=elapsed, stage_seconds=stages), allow_nan=False) + '\n')
                # Diagnostics happen AFTER decisions and outside pipeline latency.
                pixel_hash.update(result['ir_image'].tobytes())
                if args.compare_ir_root:
                    ref = cv2.imread(str(args.compare_ir_root / 'train' / sequence / 'infrared' / path.name))
                    ir_comparisons += 1
                    if ref is None or not np.array_equal(ref, result['ir_image']):
                        ir_mismatches.append(frame)
                if frame % 128 == 0:
                    print(sequence, frame, '/', len(paths), 'pipeline_ms', round(elapsed * 1000, 2), flush=True)
        report['videos'][sequence] = dict(frames=len(paths), output_rows=count, prediction_sha256=sha256(prediction),
                    timings_sha256=sha256(args.output / (sequence + '_timings.jsonl')), generated_pixels_sha256=pixel_hash.hexdigest(),
                    pipeline_seconds=sum(totals), fps=len(totals) / sum(totals),
                    latency_p50_p95_ms=(np.quantile(totals, [.5, .95]) * 1000).tolist(),
                    stage_latency_p50_p95_ms={k: (np.quantile(v, [.5, .95]) * 1000).tolist() for k, v in stage_rows.items()},
                    ir_comparisons=ir_comparisons, ir_mismatch_frames=ir_mismatches,
                    prediction_counts=getattr(tracker, 'prediction_counts', None))
        report['videos'][sequence]['write_counts'] = getattr(tracker, 'write_counts', None)
        if utility_kwargs:
            report['videos'][sequence].update(joint_counts=tracker.joint_counts, utility_counts=tracker.utility_scorer.utility_counts)
        write_json(args.output / 'manifest.json', report)
        print(sequence, report['videos'][sequence], flush=True)
    report.update(status='COMPLETE', peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                  ending_free_bytes=storage_guard(args.output, args.reserve_gib, starting_free, args.output_budget_gib))
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
