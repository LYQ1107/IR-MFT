#!/usr/bin/env python3
"""Exact same-anchor strong RGB confidence for physical matching only."""
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
from yolox.tracker.semantic_physical_scores import SemanticPhysicalScoreTracker


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); config_path = Path(args.config); config = json.loads(config_path.read_text())
    plan_path = ROOT/'configs/perceptual_mft_20261122.json'; plan = json.loads(plan_path.read_text())
    protocol = json.loads((ROOT/plan['protocol']).read_text()); work = Path(plan['work_root'])
    policy = ROOT/'configs/perceptual_single_seed_policy_20261123.json'
    assert config['seed'] == 0 and config['single_seed_user_policy_sha256'] == sha256(policy)
    assert config['reliable_threshold'] == .6
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    assert not (output/'metrics.json').exists()
    sources = {}; coverage = {}; statistics = {}; started = time.perf_counter()
    for sequence in protocol['dev']:
        base = work/'single_seed_geometry_predictions/dev'/sequence
        meta = json.loads((base/'complete.json').read_text())
        assert meta['status'] == 'COMPLETE_CORRELATION_OBSERVATIONS' and not meta['GT_read'] and not meta['future_frame_input']
        assert meta['specification_sha256'] == config['producer_specification_sha256']
        info = meta['outputs'][config['source_id']]; path = Path(info['path']); assert sha256(path) == info['sha256']
        with np.load(path) as packet:
            data = {key: packet[key].copy() for key in ['frame_ids', 'offsets', 'boxes', 'scores', 'rgb_reference_scores', 'anchor_indices']}
        np.testing.assert_array_equal(data['frame_ids'], np.arange(1, len(data['frame_ids'])+1))
        assert len(data['scores']) == len(data['rgb_reference_scores']) == len(data['anchor_indices'])
        assert ((data['anchor_indices'] >= 0) & (data['anchor_indices'] < 23625)).all()
        assert ((data['rgb_reference_scores'] >= 0) & (data['rgb_reference_scores'] <= 1)).all()
        protected = data['rgb_reference_scores'] >= .6
        data['association_scores'] = np.where(protected, data['rgb_reference_scores'], data['scores'])
        tracker = SemanticPhysicalScoreTracker(config); rows = []
        for frame in data['frame_ids']:
            rows.extend(tracker.update(int(frame), data))
        target = output/'track_results'/(sequence+'.txt'); write_sequence_results(str(target), rows)
        coverage[sequence] = dict(frames=len(data['frame_ids']), detections=len(data['boxes']),
                                  output_rows=len(rows), output_sha256=sha256(target))
        sources[sequence] = dict(path=str(path), sha256=sha256(path), producer_meta_sha256=sha256(base/'complete.json'))
        statistics[sequence] = dict(tracker.statistics, exact_strong_RGB_reference_rows=int(protected.sum()),
            changed_association_scores=int((data['association_scores'] != data['scores']).sum()))
    seconds = time.perf_counter()-started
    names = ['tools/replay_exact_reference_scores.py', 'yolox/tracker/semantic_physical_scores.py',
             'yolox/tracker/evidence_byte_tracker.py', 'yolox/tracker/byte_tracker.py', 'yolox/tracker/matching.py']
    manifest = dict(status='INFERENCE_COMPLETE_BEFORE_GT_EVALUATION', config=config, config_sha256=sha256(config_path),
        plan_sha256=sha256(plan_path), split='dev', sequences=protocol['dev'], frames=sum(r['frames'] for r in coverage.values()),
        sources=sources, coverage=coverage, statistics=statistics, source_versions={name: sha256(ROOT/name) for name in names},
        GT_inference_input=False, future_frame_input=False, previous_outputs_rewritten=False,
        replay_seconds=seconds, replay_FPS=4834/seconds,
        scope='Exact same-original-dense-anchor RGBconfidence>=.6 for primary/pending physical cost; weaker RGB keeps fusedconfidence. All source detection fields, births andsecondaryIoU stay unchanged. No fitted parameters or newseed.',
        replay_FPS_scope='Cached CPU replay only, not end-to-end encoder FPS.')
    assert manifest['frames'] == 4834; dump(output/'inference_manifest.json', manifest)
    metrics = run_trackeval(str(output/'track_results'), plan['evaluation']['GT_root'], str(output),
                           protocol['dev'], plan['evaluation']['TrackEval_root'], data_split='train')
    dump(output/'metrics.json', dict(status='COMPLETE', trackeval=metrics, manifest=manifest))
    print('EXACT_REFERENCE_REPLAY_COMPLETE', json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
