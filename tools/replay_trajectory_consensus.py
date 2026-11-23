#!/usr/bin/env python3
"""Dedicated full-Dev replay; leaves already-running legacy replays unchanged."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.eval_mft26ir import run_trackeval, write_sequence_results
from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.tracker.trajectory_consensus import TrajectoryConsensusTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config_path = Path(args.config); config = json.loads(config_path.read_text())
    plan_path = ROOT/'configs/perceptual_correlation_backend_fixed_20261123.json'
    plan = json.loads(plan_path.read_text())
    assert sha256(plan_path) == config['correlation_backend_plan_sha256']
    protocol = json.loads((ROOT/plan['protocol']).read_text()); work = Path(plan['work_root'])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    assert not (output/'metrics.json').exists()
    source_files = ['yolox/tracker/trajectory_consensus.py', 'yolox/tracker/calibrated_foreground_tracker.py',
        'yolox/tracker/foreground_byte_tracker.py', 'yolox/tracker/candidate_foreground_tracker.py',
        'yolox/tracker/evidence_byte_tracker.py', 'yolox/tracker/byte_tracker.py',
        'yolox/tracker/association_calibrator.py', 'yolox/tracker/association_features.py']
    source_hashes = {p: sha256(ROOT/p) for p in source_files}
    started = time.perf_counter(); sources = {}; coverage = {}; statistics = {}
    for sequence in protocol['dev']:
        arm = config['detection_arm']; path = work/'features/dev'/sequence/(arm+'.npz')
        meta = json.loads((path.parent/'complete.json').read_text())
        assert meta['status'] == 'COMPLETE' and sha256(path) == meta['outputs'][arm]['sha256']
        with np.load(path) as arrays:
            data = {k: arrays[k] for k in arrays.files}
        point = work/'transport/dev'/sequence/(arm+'.npz')
        point_meta = json.loads((point.parent/'complete.json').read_text())
        assert point_meta['status'] == 'COMPLETE' and sha256(point) == point_meta['outputs'][arm]['sha256']
        with np.load(point) as arrays:
            data['foreground_transport'] = {k: arrays[k] for k in arrays.files}
        np.testing.assert_array_equal(data['foreground_transport']['detection_offsets'], data['offsets'])
        np.testing.assert_array_equal(data['frame_ids'], np.arange(1, len(data['frame_ids'])+1))
        assert sha256(config['association_checkpoint']) == config['association_checkpoint_sha256']
        tracker = TrajectoryConsensusTracker(config); rows = []
        for frame in data['frame_ids']:
            rows.extend(tracker.update(int(frame), data))
        result = output/'track_results'/(sequence+'.txt'); write_sequence_results(str(result), rows)
        coverage[sequence] = dict(frames=len(data['frame_ids']), detections=len(data['boxes']),
                                  output_rows=len(rows), output_sha256=sha256(result))
        sources[sequence] = dict(path=str(path), sha256=sha256(path), point_path=str(point), point_sha256=sha256(point))
        statistics[sequence] = tracker.statistics
    manifest = dict(status='INFERENCE_COMPLETE_BEFORE_GT_EVALUATION', config=config, config_sha256=sha256(config_path),
        plan_sha256=sha256(plan_path), split='dev', sequences=protocol['dev'], frames=sum(x['frames'] for x in coverage.values()),
        sources=sources, coverage=coverage, statistics=statistics, calibration=tracker.calibration_metadata,
        association_checkpoint_sha256=sha256(config['association_checkpoint']),
        source_versions=source_hashes, trajectory_tracker_source_sha256=source_hashes['yolox/tracker/trajectory_consensus.py'],
        replay_source_sha256=sha256(__file__), GT_inference_input=False, future_frame_input=False, previous_outputs_rewritten=False,
        replay_seconds=time.perf_counter()-started, replay_FPS_scope='Cached replay; pixel encoders and point production excluded, not end-to-end FPS.')
    assert manifest['frames'] == 4834
    manifest['replay_FPS'] = manifest['frames']/manifest['replay_seconds']
    dump(output/'inference_manifest.json', manifest)
    metrics = run_trackeval(str(output/'track_results'), plan['evaluation']['GT_root'], str(output),
                           protocol['dev'], plan['evaluation']['TrackEval_root'], data_split='train')
    dump(output/'metrics.json', dict(status='COMPLETE', trackeval=metrics, manifest=manifest))
    print('TRAJECTORY_CONSENSUS_REPLAY_COMPLETE', json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
