#!/usr/bin/env python3
"""Run unchanged pinned external tracker association with shared fish features."""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np

from cfp.provenance import sha256, write_json


class PrecomputedFishInterface:
    """Explicit feature-provider adaptation, not a native FastReID model."""
    def __init__(self, *_):
        self.boxes = self.features = None

    def set_frame(self, detections, features, high_threshold, low_threshold):
        selected = (detections[:, 4] > high_threshold) & (detections[:, 4] > low_threshold)
        self.boxes = detections[selected, :4].copy()
        self.features = features[selected].copy()

    def inference(self, image, boxes):
        boxes = np.asarray(boxes).reshape(-1, 4)
        if self.boxes is None or not np.array_equal(boxes, self.boxes):
            raise ValueError('BoT-SORT feature request lost proposal correspondence')
        return self.features.copy()


def scalar_score(value):
    array = np.asarray(value)
    if array.size != 1 or not np.isfinite(array).all():
        raise ValueError('Expected a finite singleton upstream prediction score')
    return float(array.reshape(-1)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=['hybrid_sort', 'bot_sort'], required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--features-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--partition', choices=['train', 'dev', 'historical_test'], required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--candidate', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--final-freeze', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('External replay outputs are immutable')
    if args.partition == 'historical_test' and (not args.final_freeze or json.loads(args.final_freeze.read_text())['state'] != 'FROZEN'):
        raise ValueError('Historical test requires final frozen configuration')
    config = json.loads(args.config.read_text())
    if config['state'] != 'FROZEN':
        raise ValueError('Undeclared external baseline')
    protocol = json.loads(args.protocol.read_text())
    name = 'HybridSORT' if args.family == 'hybrid_sort' else 'BoT-SORT'
    reference = next(ref for ref in json.loads(args.references.read_text())['references'] if ref['name'] == name)
    commit = subprocess.check_output(['git', '-C', str(args.reference_root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != reference['commit']:
        raise ValueError('External reference commit changed')
    for file, metadata in reference['sources'].items():
        if sha256(args.reference_root / file) != metadata['sha256']:
            raise ValueError(f'Pinned external source changed: {file}')
    subtree = 'trackers/hybrid_sort_tracker' if args.family == 'hybrid_sort' else 'tracker'
    source_hashes = {str(file.relative_to(args.reference_root)): sha256(file) for file in sorted((args.reference_root / subtree).glob('*.py'))}
    # Pinned upstream numpy aliases are restored only in this isolated process.
    for alias, value in [('float', float), ('int', int), ('bool', bool)]:
        if alias not in np.__dict__:
            setattr(np, alias, value)
    sys.path.insert(0, str(args.reference_root))
    parameters = dict(config[args.family])
    if args.family == 'hybrid_sort':
        module = importlib.import_module('trackers.hybrid_sort_tracker.hybrid_sort_reid')
        # Upstream puts singleton score arrays beside Python scalars in a row;
        # numpy 1.24 rejects this. Convert only scores, preserving their values.
        original_predict = module.KalmanBoxTracker.predict
        def compatible_predict(track):
            box, kalman_score, simple_score = original_predict(track)
            return box, scalar_score(kalman_score), scalar_score(simple_score)
        module.KalmanBoxTracker.predict = compatible_predict
        parameters['EG_weight_high_score'] = parameters.pop('appearance_high_weights')[args.candidate]
        parameters['EG_weight_low_score'] = parameters.pop('appearance_low_ratio') * parameters['EG_weight_high_score']
        constructor = parameters.pop('constructor')
        make_tracker = lambda: module.Hybrid_Sort_ReID(SimpleNamespace(**parameters), parameters['track_thresh'], **constructor)
    else:
        # Satisfy only the native provider import with a verified feature adapter;
        # all motion, assignment, lifecycle and memory methods remain upstream.
        shim = ModuleType('fast_reid.fast_reid_interfece')
        shim.FastReIDInterface = PrecomputedFishInterface
        sys.modules['fast_reid.fast_reid_interfece'] = shim
        module = importlib.import_module('tracker.bot_sort')
        parameters['appearance_thresh'] = parameters.pop('appearance_thresholds')[args.candidate]
        constructor = {}
        make_tracker = lambda: module.BoTSORT(SimpleNamespace(**parameters), frame_rate=config['frame_rate'])
    if not Path(module.__file__).resolve().is_relative_to(args.reference_root.resolve()):
        raise ValueError('Imported wrong tracker package')
    detection_manifest = json.loads((args.detections / 'manifest.json').read_text())
    feature_manifest = json.loads((args.features_root / 'manifest.json').read_text())
    if detection_manifest['status'] != 'COMPLETE' or detection_manifest['probe'] or feature_manifest['status'] != 'COMPLETE':
        raise ValueError('Incomplete input banks')
    if detection_manifest['protocol_sha256'] != sha256(args.protocol) or feature_manifest['protocol_sha256'] != sha256(args.protocol):
        raise ValueError('Input split changed')
    if detection_manifest['config']['input_regime'] != 'causal' or feature_manifest['input_regime'] != 'causal':
        raise ValueError('New baselines require causal inputs')
    args.output.mkdir(parents=True)
    manifest = dict(status='RUNNING', family=args.family, candidate=args.candidate, parameters=parameters,
                    constructor=constructor, config_sha256=sha256(args.config), protocol_sha256=sha256(args.protocol),
                    reference_commit=commit, reference_source_sha256=source_hashes, reference_license_sha256=sha256(args.reference_root / 'LICENSE'),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    source_sha256=sha256(__file__), feature_manifest_sha256=sha256(args.features_root / 'manifest.json'),
                    rgb_encoder=feature_manifest['rgb_encoder'],
                    compatibility_adapter='numpy singleton prediction scores to Python float' if args.family == 'hybrid_sort' else 'precomputed fish feature provider; numpy aliases',
                    scope='Protocol A external association with shared fish RGB features; not native person-ReID weights, no GT-assisted tracking')
    write_json(args.output / 'manifest.json', manifest)
    reports = {}
    for sequence in protocol[args.partition]:
        detections_path, features_path = args.detections / (sequence + '.npz'), args.features_root / (sequence + '.npz')
        provenance = json.loads(features_path.with_suffix('.provenance.json').read_text())
        if sha256(detections_path) != detection_manifest['caches'][sequence]['sha256'] or provenance['detection_sha256'] != sha256(detections_path):
            raise ValueError('Proposal bank changed')
        if provenance['status'] != 'COMPLETE' or sha256(features_path) != provenance['features_sha256']:
            raise ValueError('Features changed')
        cache, features = np.load(detections_path), np.load(features_path)['rgb']
        if len(features) != len(cache['boxes']) or not np.isfinite(features).all() or np.any(np.linalg.norm(features, axis=1) < 1e-8):
            raise ValueError('External tracker requires valid aligned features; no proposal silently removed')
        tracker = make_tracker()
        rows, times = [], []
        for i, frame in enumerate(cache['frame_ids']):
            lo, hi = cache['offsets'][i:i + 2]
            detections = np.concatenate([cache['boxes'][lo:hi].copy(), cache['scores'][lo:hi, None]], axis=1)
            current_features = features[lo:hi].astype(np.float32, copy=True)
            start = time.perf_counter()
            if args.family == 'hybrid_sort':
                output = tracker.update(detections, (1, 1), (1, 1), id_feature=current_features)
                entries = [(row[4], row[:2], row[2:4] - row[:2], 1.0) for row in output]
            else:
                tracker.encoder.set_frame(detections, current_features, parameters['track_high_thresh'], parameters['track_low_thresh'])
                output = tracker.update(detections, None)  # GMC none never reads image pixels.
                entries = [(track.track_id, track.tlwh[:2], track.tlwh[2:], float(track.score)) for track in output]
            times.append(time.perf_counter() - start)
            for identity, xy, wh, score in entries:
                if np.prod(wh) > config['min_box_area']:
                    rows.append(f'{frame},{int(identity)},{xy[0]:.3f},{xy[1]:.3f},{wh[0]:.3f},{wh[1]:.3f},{score:.6f},-1,-1,-1\n')
        path = args.output / (sequence + '.txt')
        path.write_text(''.join(rows))
        reports[sequence] = dict(frames=len(cache['frame_ids']), detection_sha256=sha256(detections_path),
                                 features_sha256=sha256(features_path), prediction_sha256=sha256(path),
                                 association_seconds=sum(times), association_latency_p50_p95_ms=(np.quantile(times, [.5, .95]) * 1000).tolist())
        print(args.family, sequence, 'complete', flush=True)
    manifest.update(status='COMPLETE', videos=reports,
                    latency_scope='Cached features association under shared CPU load; excludes encoding, generation, detector, decode; not a dedicated end-to-end benchmark')
    write_json(args.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
