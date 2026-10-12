#!/usr/bin/env python3
"""Validate full-video online/intervention artifacts and retain honest cost scope."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from cfp.provenance import sha256, snapshot_sources, write_json


def checked_run(root, name, protocol_path, protocol):
    directory = root / name
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    evaluation_path = root / (name + '_evaluation') / 'metrics.json'
    evaluation = json.loads(evaluation_path.read_text())
    if manifest['status'] != 'COMPLETE' or manifest['probe'] or manifest['protocol_sha256'] != sha256(protocol_path) or manifest['partition'] != 'dev':
        raise ValueError('Final collection requires complete canonical dev runs')
    for source, expected in manifest['source_snapshots'].items():
        if sha256(directory / 'source_snapshots' / source) != expected:
            raise ValueError('Exact run source snapshot changed')
    totals = []
    for sequence, frames in [('BT-005', 1834), ('SN-013', 3000)]:
        info = manifest['videos'][sequence]
        if info['frames'] != frames or sha256(directory / (sequence + '.txt')) != info['prediction_sha256'] or evaluation['artifacts'][sequence]['prediction_sha256'] != info['prediction_sha256']:
            raise ValueError('Incomplete or changed current-frame trajectories')
        if evaluation['artifacts'][sequence]['gt_sha256'] != protocol['derived_gt_sha256']['train/' + sequence]:
            raise ValueError('Online metrics use a different GT protocol')
        timings = directory / (sequence + '_timings.jsonl')
        if sha256(timings) != info['timings_sha256']:
            raise ValueError('Measured timing rows changed')
        records = [json.loads(row) for row in timings.read_text().splitlines()]
        if [r['frame'] for r in records] != list(range(1, frames + 1)):
            raise ValueError('Missing current-frame timing events')
        totals.extend(r['total_seconds'] for r in records)
    metrics = {s: {k: v for k, v in values.items() if isinstance(v, (int, float))}
               for s, values in evaluation['metrics'].items()}
    return dict(manifest_sha256=sha256(manifest_path), evaluation_sha256=sha256(evaluation_path),
                protocol=manifest['protocol'], precomputed_causal_ir=manifest.get('precomputed_causal_ir', False),
                metrics=metrics, videos=manifest['videos'], config=manifest['config'],
                parameters=manifest['parameters'], peak_gpu_memory_bytes=manifest['peak_gpu_memory_bytes'],
                aggregate_pipeline_seconds=float(sum(totals)), aggregate_fps=len(totals) / sum(totals),
                aggregate_latency_p50_p95_ms=(np.quantile(totals, [.5, .95]) * 1000).tolist(),
                scope=manifest['scope'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets-root', type=Path, required=True)
    p.add_argument('--causal-data-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runtime-variant', choices=['cu118_b1', 'split_cu117_b8'], default='cu118_b1')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Measured integration report is immutable')
    protocol = json.loads(args.protocol.read_text())
    names = [f'online_anchor_gru_seed{s}_'+('b1_full_v1' if args.runtime_variant=='cu118_b1' else 'split_full_v1') for s in range(3)]
    names += [f'online_ir_intervention_{kind}_{view}_seed0_v1' for kind, views in
              [('fixed_matched', ['none', 'zero', 'repeat_rgb', 'spatial_shuffle']), ('cmue', ['none', 'zero'])] for view in views]
    runs = {name: checked_run(args.assets_root, name, args.protocol, protocol) for name in names}
    generation = json.loads((args.causal_data_root / 'manifest.json').read_text())
    if generation['status'] != 'COMPLETE' or generation.get('probe_frames', 0):
        raise ValueError('Current-frame equivalence needs complete original causal generation')
    # Check all generated frame bytes/decoded pixels once, in a bounded stream.
    canonical = {}
    cv2.setNumThreads(1)
    for sequence, length in [('BT-005', 1834), ('SN-013', 3000)]:
        expected = {}
        for path in sorted((args.causal_data_root / 'provenance').glob('train_' + sequence + '_*.json')):
            chunk = json.loads(path.read_text())
            for row in chunk['frames_manifest']:
                expected[row['frame_id']] = row['ir_sha256']
        if set(expected) != set(range(1, length + 1)):
            raise ValueError('Original causal image provenance is incomplete')
        digest = hashlib.sha256()
        for frame in range(1, length + 1):
            path = args.causal_data_root / 'train' / sequence / 'infrared' / f'{frame:06d}.jpg'
            if sha256(path) != expected[frame]:
                raise ValueError('Original causal response frame bytes changed')
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Cannot decode canonical causal frame')
            digest.update(image.tobytes())
        canonical[sequence] = dict(frames=length, generated_pixels_sha256=digest.hexdigest())
        for name, run in runs.items():
            if run['videos'][sequence]['generated_pixels_sha256'] != digest.hexdigest():
                raise ValueError('Current-frame generation differs from audited causal images: ' + name)
        print('Exact complete causal pixel equivalence', sequence, length, flush=True)
    metrics = runs[names[0]]['metrics']['COMBINED_SEQ']
    fields = [k for k in metrics if isinstance(metrics[k], (int, float))]
    aggregate = dict(mean={k: float(np.mean([runs[name]['metrics']['COMBINED_SEQ'][k] for name in names[:3]])) for k in fields},
                     sample_std={k: float(np.std([runs[name]['metrics']['COMBINED_SEQ'][k] for name in names[:3]], ddof=1)) for k in fields},
                     runs=names[:3], runtime_variant=args.runtime_variant, scope='Frozen GRU anchor candidate, complete current-frame pipeline. Explicit detector runtime/batch comes from each run config; encoderpadding128. Keep original cu118/batch1 diagnosis separate from corrected cu117/batch8 pipeline.')
    report = dict(status='COMPLETE', protocol_sha256=sha256(args.protocol),
                  source_snapshots=snapshot_sources(args.output.with_suffix(''), [__file__, args.protocol]),
                  canonical_pixel_equivalence=canonical, three_seed_current_frame=aggregate, runs=runs,
                  interpretation='Report every declared pixel intervention and all current-frame seeds. Protocol A precomputed responses/proposals exclude detector/generator cost; never label its FPS end-to-end. Host is shared. Original cu118/batch1 seeds share physicalGPU0; corrected split-runtime seeds each use GPU0/1/2, detector and encoder share their respective GPU. No historical-test or independent-detector-holdout claim.')
    write_json(args.output, report)
    print('Current-frame mean', {k: round(aggregate['mean'][k], 4) for k in ['HOTA', 'AssA', 'IDF1', 'IDSW']})
    for name in names:
        entry = runs[name]
        print(name, {k: round(entry['metrics']['COMBINED_SEQ'][k], 4) for k in ['HOTA', 'AssA', 'IDF1', 'IDSW']},
              'FPS', round(entry['aggregate_fps'], 3), 'p50/p95', entry['aggregate_latency_p50_p95_ms'])


if __name__ == '__main__':
    main()
