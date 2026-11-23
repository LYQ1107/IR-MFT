#!/usr/bin/env python3
"""Full current-frame score-role replay with unchanged semantic observations."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.special import expit
from tools.eval_mft26ir import run_trackeval, write_sequence_results
from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.tracker.semantic_physical_scores import geometry_reference_scores, SemanticPhysicalScoreTracker


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); config_path = Path(args.config); config = json.loads(config_path.read_text())
    plan_path = ROOT/'configs/perceptual_mft_20261122.json'; plan = json.loads(plan_path.read_text()); work = Path(plan['work_root'])
    protocol = json.loads((ROOT/plan['protocol']).read_text()); output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True); assert not (output/'metrics.json').exists()
    sources = {}; coverage = {}; statistics = {}; started = time.perf_counter()
    for sequence in protocol['dev']:
        canonical = Path(plan['frozen_detection_sources']['rgb_dev'])/'detections'/(sequence+'.npz')
        with np.load(canonical) as arrays:
            reference = {k: arrays[k].copy() for k in ['frame_ids', 'offsets', 'boxes', 'scores']}
        if config['source_kind'] == 'TC':
            directory = work/'temporal_geometry_predictions/dev'/sequence
            meta = json.loads((directory/'complete.json').read_text()); info = meta['outputs'][config['source_id']]
            source = Path(info['path']); assert sha256(source) == info['sha256']
            assert meta['status'] == 'COMPLETE_CORRELATION_OBSERVATIONS' and not meta['GT_read'] and not meta['future_frame_input']
            with np.load(source) as arrays:
                data = {k: arrays[k].copy() for k in ['frame_ids', 'offsets', 'boxes', 'scores']}
        else:
            assert config['source_kind'] == 'RGB_map'
            source = canonical; data = {k: v.copy() for k, v in reference.items()}
            fitted_path = work/'RGB_score_calibration/models.json'
            assert sha256(fitted_path) == config['fitted_maps_sha256']
            model = json.loads(fitted_path.read_text())['models'][config['score_map']]
            if config['score_map'] != 'identity':
                score = np.clip(data['scores'].astype(float), 1e-6, 1-1e-6)
                data['scores'] = expit(model['slope']*np.log(score/(1-score))+model['intercept']).astype(np.float32)
        np.testing.assert_array_equal(data['frame_ids'], reference['frame_ids'])
        np.testing.assert_array_equal(data['frame_ids'], np.arange(1, len(data['frame_ids'])+1))
        data['association_scores'] = data['scores'].copy(); protected = 0
        for frame in data['frame_ids']:
            a, b = map(int, data['offsets'][frame-1:frame+1]); c, d = map(int, reference['offsets'][frame-1:frame+1])
            values, supported = geometry_reference_scores(data['boxes'][a:b], data['scores'][a:b],
                reference['boxes'][c:d], reference['scores'][c:d])
            data['association_scores'][a:b] = values; protected += int(supported.sum())
        tracker = SemanticPhysicalScoreTracker(config); rows = []
        for frame in data['frame_ids']:
            rows.extend(tracker.update(int(frame), data))
        target = output/'track_results'/(sequence+'.txt'); write_sequence_results(str(target), rows)
        coverage[sequence] = dict(frames=len(data['frame_ids']), detections=len(data['boxes']),
                                  output_rows=len(rows), output_sha256=sha256(target))
        sources[sequence] = dict(path=str(source), sha256=sha256(source), current_RGB_reference=str(canonical), reference_sha256=sha256(canonical))
        statistics[sequence] = dict(tracker.statistics, geometry_reference_supported_observations=protected)
    runtime = time.perf_counter()-started
    manifest = dict(status='INFERENCE_COMPLETE_BEFORE_GT_EVALUATION', config=config, config_sha256=sha256(config_path),
        plan_sha256=sha256(plan_path), split='dev', sequences=protocol['dev'], frames=sum(v['frames'] for v in coverage.values()),
        sources=sources, coverage=coverage, statistics=statistics, GT_inference_input=False, future_frame_input=False,
        previous_outputs_rewritten=False, replay_seconds=runtime, replay_FPS=sum(v['frames'] for v in coverage.values())/runtime,
        replay_source_sha256=sha256(__file__), score_tracker_source_sha256=sha256(ROOT/'yolox/tracker/semantic_physical_scores.py'),
        evidence_tracker_source_sha256=sha256(ROOT/'yolox/tracker/evidence_byte_tracker.py'),
        byte_tracker_source_sha256=sha256(ROOT/'yolox/tracker/byte_tracker.py'),
        scope='Detection confidence/thresholds/geometry remain the source values; only primary/pending confidence weights use current geometry-matched strongRGB references. No exact dense-anchor reference is claimed.',
        replay_FPS_scope='Cached CPU score-role replay, not encoder end-to-end FPS.')
    assert manifest['frames'] == 4834; dump(output/'inference_manifest.json', manifest)
    metrics = run_trackeval(str(output/'track_results'), plan['evaluation']['GT_root'], str(output),
                           protocol['dev'], plan['evaluation']['TrackEval_root'], data_split='train')
    dump(output/'metrics.json', dict(status='COMPLETE', trackeval=metrics, manifest=manifest))
    print('SEMANTIC_PHYSICAL_SCORE_REPLAY_COMPLETE', json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
