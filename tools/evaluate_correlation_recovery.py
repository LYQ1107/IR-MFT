#!/usr/bin/env python3
"""Official CPU metrics only after complete frozen CGR pixel production."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.correlation_recovery_workflow import WORKFLOW, context, read, registered
from tools.eval_mft26ir import run_trackeval
from tools.run_pseudo_ir_comparison import dump, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['dev', 'test'], required=True)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plan, spec, protocol, parent, calibration, work = context()
    workflow = registered()
    source = next(row for row in workflow['case_files'] if row['id'] == args.case)
    config_path = ROOT / source['path']
    assert sha256(config_path) == source['sha256']
    config = read(config_path)
    assert config['id'] == args.case
    sequences = protocol['dev' if args.split == 'dev' else 'historical_test']
    directory = work / ('correlation_recovery_'+args.split)
    # Check every video and every case before any official Test score is read.
    producers = {}
    for sequence in sequences:
        path = directory / 'sequence_manifests' / (sequence+'.json')
        meta = read(path)
        assert meta['status'] == 'COMPLETE_CGR_CURRENT_IMAGE_STREAMS_WITHOUT_GT'
        assert meta['workflow_sha256'] == sha256(WORKFLOW) and not meta['GT_read']
        assert not meta['future_frame_information_used'] and not meta['previous_outputs_rewritten']
        assert set(meta['streams']) == {row['id'] for row in spec['cases']}
        for result in meta['streams'].values():
            assert result['frames'] == meta['frames'] and sha256(result['track_path']) == result['track_sha256']
        producers[sequence] = dict(meta=meta, manifest_sha256=sha256(path), manifest_path=str(path))
    total = sum(row['meta']['frames'] for row in producers.values())
    assert total == (4834 if args.split == 'dev' else 15985)
    if args.split == 'test':
        from tools.run_correlation_recovery_gpu import require_test_freeze
        require_test_freeze(work)
    output = directory / args.case
    assert not (output / 'metrics.json').exists()
    coverage, sources, models = {}, {}, {}
    for seq, entry in producers.items():
        meta = entry['meta']
        stream = meta['streams'][args.case]
        coverage[seq] = dict(frames=stream['frames'], output_rows=stream['track_rows'],
            detections=stream['detections'], output_sha256=stream['track_sha256'])
        sources[seq] = dict(pixel_manifest=entry['manifest_path'], pixel_manifest_sha256=entry['manifest_sha256'],
            actual_images_delivery_sha256=meta['original_image_bytes_delivery_sha256'],
            canonical_proposal_sha256=meta['canonical_proposal_sha256'],
            detection_packet_sha256=stream['detection_packet_sha256'], physical_gpu=meta['physical_gpu'],
            GPU_uuid=meta['GPU_uuid'])
        models[seq] = meta['models'][args.case]
    manifest = dict(status='INFERENCE_COMPLETE_BEFORE_GT_EVALUATION', config=config,
        config_sha256=sha256(config_path), config_path=str(config_path), workflow_sha256=sha256(WORKFLOW),
        split=args.split, sequences=sequences, frames=total, sources=sources, coverage=coverage,
        model_sources=models, plan_sha256=sha256(ROOT / 'configs/perceptual_mft_20261122.json'),
        GT_inference_input=False, future_frame_input=False, previous_outputs_rewritten=False,
        statistics={}, runtime_scope='Metrics evaluation of completed streams; no method FPS from evaluation wall time.')
    dump(output / 'inference_manifest.json', manifest)
    metrics = run_trackeval(str(output / 'track_results'), plan['evaluation']['GT_root'], str(output),
        sequences, plan['evaluation']['TrackEval_root'], data_split='train' if args.split == 'dev' else 'test')
    assert len(metrics) == 39
    dump(output / 'metrics.json', dict(status='COMPLETE', trackeval=metrics, manifest=manifest))
    print(json.dumps(dict(case=args.case, split=args.split, frames=total, status='COMPLETE_OFFICIAL_CGR_METRICS')), flush=True)


if __name__ == '__main__':
    main()
