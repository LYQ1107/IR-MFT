#!/usr/bin/env python3
"""CPU complete-Dev canonical RGB replay parity for the streaming writer."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from tools.eval_mft26ir import run_trackeval
from tools.evaluate_native_pyramid import packet
from tools.native_pyramid_experiment import CANONICAL_PROOF, PIPELINE_REVISION, STUDY, context
from tools.native_pyramid_stream_tracks import StreamTrackWriter
from tools.run_pseudo_ir_comparison import dump, sha256


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plan, study, protocol, calibration, work, checkpoint = context()
    folder = work / f'native_pyramid_canonical_Dev_engineering_v{PIPELINE_REVISION}'
    output = CANONICAL_PROOF
    assert not folder.exists() and not output.exists()
    sources = ['tools/probe_native_pyramid_canonical_dev.py', 'tools/evaluate_native_pyramid.py',
        'tools/native_pyramid_stream_tracks.py', 'tools/eval_mft26ir.py',
        'yolox/tracker/byte_tracker.py', 'yolox/tracker/basetrack.py',
        'yolox/tracker/matching.py', 'yolox/tracker/kalman_filter.py']
    hashes = {name: sha256(ROOT / name) for name in sources}
    baseline = Path(plan['frozen_detection_sources']['rgb_dev'])
    started, coverage = time.monotonic(), {}
    for sequence in protocol['dev']:
        path = baseline / 'detections' / (sequence+'.npz')
        source = packet(path)
        writer = StreamTrackWriter(folder, sequence, ['source_RGB'])
        try:
            for frame in source['frame_ids']:
                lo, hi = source['offsets'][frame-1:frame+1]
                values = np.c_[source['boxes'][lo:hi], source['scores'][lo:hi]].astype(np.float32)
                writer.append(int(frame), dict(source_RGB=values))
            result = writer.close(len(source['frame_ids']))['source_RGB']
            coverage[sequence] = dict(result, canonical_source_sha256=sha256(path))
        except Exception:
            writer.abort()
            raise
    assert sum(row['frames'] for row in coverage.values()) == 4834
    metrics = run_trackeval(str(folder / 'source_RGB/track_results'), plan['evaluation']['GT_root'], str(folder),
        protocol['dev'], plan['evaluation']['TrackEval_root'], data_split='train')
    original = json.loads((baseline / 'metrics.json').read_text())
    assert metrics == original['trackeval'] and len(metrics) == 39
    csv_path = folder / 'trackeval_output/pedestrian_detailed.csv'
    original_csv = baseline / 'trackeval_output/pedestrian_detailed.csv'
    assert sha256(csv_path) == sha256(original_csv)
    assert all(sha256(ROOT / name) == digest for name, digest in hashes.items())
    dump(output, dict(status='COMPLETE_NATIVE_STREAM_CANONICAL_ALL4834DEV_ALL39_CSV_EXACT',
        metrics=metrics, coverage=coverage, study_sha256=sha256(STUDY), source_hashes=hashes, pipeline_revision=PIPELINE_REVISION,
        official_CSV_sha256=sha256(csv_path), original_official_CSV_sha256=sha256(original_csv),
        source_metrics_sha256=sha256(baseline / 'metrics.json'), seconds=time.monotonic()-started,
        isolated_sequence_local_IDs_from1=True, all39_summary_exact=True, complete_official_CSV_exact=True,
        Dev_frames=4834, Test_rows=0, new_model_fitted=False, GPU_work=False,
        no_new_detection_or_image_or_feature_cache=True, saved_weights=False,
        scope='Complete canonical-Dev replay/official evaluation proof, not live GPU encoder or new fusion accuracy.',
        goal_met=False, supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_NATIVE_STREAM_CANONICAL_ALL4834DEV_ALL39_CSV_EXACT',
                          seconds=time.monotonic()-started)), flush=True)


if __name__ == '__main__': main()
