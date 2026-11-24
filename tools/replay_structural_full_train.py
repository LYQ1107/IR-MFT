#!/usr/bin/env python3
"""Original ByteTrack/TrackEval; only the registered full-Train source family differs."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.eval_mft26ir import run_trackeval, write_sequence_results
from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.tracker.byte_tracker import BYTETracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    args = parser.parse_args()
    case_path = Path(args.config)
    case = json.loads(case_path.read_text())
    plan_path = ROOT / 'configs/perceptual_mft_20261122.json'
    plan = json.loads(plan_path.read_text())
    work = Path(plan['work_root'])
    protocol = json.loads((ROOT / plan['protocol']).read_text())
    sequences = protocol['dev' if args.split == 'dev' else 'historical_test']
    study_path = ROOT / 'configs/perceptual_structural_fulltrain_study_20261123.json'
    study = json.loads(study_path.read_text())
    assert sha256(study_path) == case['full_train_study_sha256']
    assert case['fusion_plan_path'] == study['configuration']
    assert sha256(ROOT / study['configuration']) == case['fusion_plan_sha256'] == study['configuration_sha256']
    policy_path = ROOT / 'configs/perceptual_single_seed_policy_20261123.json'
    assert case['seed'] == 0 and sha256(policy_path) == case['single_seed_user_policy_sha256']
    if args.split == 'test':
        freeze = json.loads((work / 'structural_full_train_Test_selection_frozen.json').read_text())
        assert freeze['study_sha256'] == sha256(study_path) and freeze['complete50Dev_before_Test']
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'metrics.json').exists()
    sources, coverage, versions, model_sources = {}, {}, {}, {}
    started = time.perf_counter()
    for sequence in sequences:
        if case.get('baseline') and not case.get('fresh_RGB'):
            path = Path(plan['frozen_detection_sources']['rgb_' + args.split]) / 'detections' / (sequence + '.npz')
        else:
            folder = work / case['producer_directory'] / args.split / sequence
            producer = json.loads((folder / 'complete.json').read_text())
            info = producer['outputs'][case['proposal_key']]
            path = Path(info['path'])
            assert producer['status'] == 'COMPLETE_STRUCTURAL_RELATIVE_OBSERVATIONS'
            assert not producer['GT_read'] and not producer['future_frame_input']
            assert producer['config_sha256'] == case['fusion_plan_sha256']
            assert producer['specification_sha256'] == case['specification_sha256']
            assert producer['full_train_study_sha256'] == case['full_train_study_sha256']
            assert sha256(path) == info['sha256']
            versions.update(producer['source_hashes'])
            model_sources[sequence] = producer['models'][case['model_id']]
            versions[sequence + '_adapter_checkpoint'] = model_sources[sequence].get('checkpoint_sha256', producer['source_RGB_sha256'])
        with np.load(path) as z:
            data = {key: z[key] for key in ['frame_ids', 'offsets', 'boxes', 'scores']}
        np.testing.assert_array_equal(data['frame_ids'], np.arange(1, len(data['frame_ids']) + 1))
        tracker = BYTETracker(SimpleNamespace(track_thresh=.6, track_buffer=30, match_thresh=.9, mot20=False), frame_rate=25)
        rows = []
        for frame in data['frame_ids']:
            lo, hi = data['offsets'][frame - 1:frame + 1]
            detections = np.c_[data['boxes'][lo:hi], data['scores'][lo:hi]].astype(np.float32)
            rows.extend((int(frame), track.tlwh.copy(), track.track_id, track.score)
                for track in tracker.update(detections, (1, 1), (1, 1)) if track.tlwh[2] * track.tlwh[3] > 100)
        target = output / 'track_results' / (sequence + '.txt')
        write_sequence_results(str(target), rows)
        sources[sequence] = dict(path=str(path), sha256=sha256(path))
        coverage[sequence] = dict(frames=len(data['frame_ids']), detections=len(data['boxes']),
                                 output_rows=len(rows), output_sha256=sha256(target))
    manifest = dict(status='INFERENCE_COMPLETE_BEFORE_GT_EVALUATION', config=case, config_sha256=sha256(case_path),
        plan_sha256=sha256(plan_path), split=args.split, sequences=sequences,
        frames=sum(v['frames'] for v in coverage.values()), sources=sources, coverage=coverage,
        source_versions=versions, model_sources=model_sources, statistics={},
        GT_inference_input=False, future_frame_input=False, previous_outputs_rewritten=False,
        replay_seconds=time.perf_counter() - started, replay_source_sha256=sha256(__file__),
        byte_tracker_source_sha256=sha256(ROOT / 'yolox/tracker/byte_tracker.py'),
        replay_FPS_scope='Cached proposals only; excludes encoder/fusion/preparation/IRgeneration, not end-to-end FPS.')
    manifest['replay_FPS'] = manifest['frames'] / manifest['replay_seconds']
    dump(output / 'inference_manifest.json', manifest)
    metrics = run_trackeval(str(output / 'track_results'), plan['evaluation']['GT_root'], str(output), sequences,
                           plan['evaluation']['TrackEval_root'], data_split='train' if args.split == 'dev' else 'test')
    dump(output / 'metrics.json', dict(status='COMPLETE', trackeval=metrics, manifest=manifest))
    print('FULL_TRAIN_REPLAY_COMPLETE', json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
