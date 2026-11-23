#!/usr/bin/env python3
"""Ordinary scalar RGB calibration, frozen candidates and unchanged ByteTrack."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.special import expit
from tools.eval_mft26ir import run_trackeval, write_sequence_results
from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.tracker.byte_tracker import BYTETracker


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); config_path = Path(args.config); config = json.loads(config_path.read_text())
    plan_path = ROOT/'configs/perceptual_mft_20261122.json'; plan = json.loads(plan_path.read_text())
    spec_path = ROOT/'configs/perceptual_RGB_score_calibration_20261123.json'
    protocol = json.loads((ROOT/plan['protocol']).read_text()); work = Path(plan['work_root'])
    model_path = work/'RGB_score_calibration/models.json'; fitted = json.loads(model_path.read_text())
    assert fitted['status'] == 'COMPLETE_BOTH_RGB_TRAIN_CALIBRATIONS_BEFORE_DEV' and fitted['config_sha256'] == sha256(spec_path)
    assert sha256(model_path) == config['fitted_maps_sha256']
    method = config['method']; model = fitted['models'][method]
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); assert not (output/'metrics.json').exists()
    sources = {}; coverage = {}; started = time.perf_counter()
    for sequence in protocol['dev']:
        source = Path(plan['frozen_detection_sources']['rgb_dev'])/'detections'/(sequence+'.npz')
        with np.load(source) as arrays:
            data = {key: arrays[key].copy() for key in ['frame_ids', 'offsets', 'boxes', 'scores']}
        np.testing.assert_array_equal(data['frame_ids'], np.arange(1, len(data['frame_ids'])+1))
        if method != 'identity':
            score = np.clip(data['scores'].astype(float), 1e-6, 1-1e-6)
            data['scores'] = expit(model['slope']*np.log(score/(1-score))+model['intercept']).astype(np.float32)
        tracker = BYTETracker(SimpleNamespace(track_thresh=.6, track_buffer=30, match_thresh=.9, mot20=False), frame_rate=25)
        rows = []
        for frame in data['frame_ids']:
            lo, hi = data['offsets'][frame-1:frame+1]
            detections = np.c_[data['boxes'][lo:hi], data['scores'][lo:hi]].astype(np.float32)
            rows.extend((int(frame), track.tlwh.copy(), track.track_id, track.score)
                for track in tracker.update(detections, (1, 1), (1, 1)) if track.tlwh[2]*track.tlwh[3] > 100)
        target = output/'track_results'/(sequence+'.txt'); write_sequence_results(str(target), rows)
        coverage[sequence] = dict(frames=len(data['frame_ids']), detections=len(data['boxes']), output_rows=len(rows), output_sha256=sha256(target))
        sources[sequence] = dict(path=str(source), sha256=sha256(source))
    runtime = time.perf_counter()-started
    manifest = dict(status='INFERENCE_COMPLETE_BEFORE_GT_EVALUATION', config=config, config_sha256=sha256(config_path),
        plan_sha256=sha256(plan_path), split='dev', sequences=protocol['dev'], frames=sum(x['frames'] for x in coverage.values()),
        sources=sources, coverage=coverage, statistics={}, calibration=dict(method=method, **model),
        fitted_map_source_sha256=sha256(model_path), replay_source_sha256=sha256(__file__),
        byte_tracker_source_sha256=sha256(ROOT/'yolox/tracker/byte_tracker.py'),
        GT_inference_input=False, future_frame_input=False, previous_outputs_rewritten=False,
        replay_seconds=runtime, replay_FPS=sum(x['frames'] for x in coverage.values())/runtime,
        replay_FPS_scope='Cached score mapping and tracking, not pixel encoder FPS.',
        limitation='Only canonical original NMS candidates at source confidence>=0.1; no additional below-threshold anchor recall.')
    assert manifest['frames'] == 4834; dump(output/'inference_manifest.json', manifest)
    metrics = run_trackeval(str(output/'track_results'), plan['evaluation']['GT_root'], str(output),
                           protocol['dev'], plan['evaluation']['TrackEval_root'], data_split='train')
    dump(output/'metrics.json', dict(status='COMPLETE', trackeval=metrics, manifest=manifest))
    print('RGB_SCORE_CALIBRATION_COMPLETE', json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
