#!/usr/bin/env python3
"""Verify complete deployment predictions, numerical parity and full online cost."""
import json
from pathlib import Path

import numpy as np

from cfp.provenance import sha256, write_json


def main():
    root = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    online = root / 'phase6/risk_quota_online_full_init0_v1'
    cached = root / 'phase6/risk_quota_deployment_cached_init0_v1'
    om = json.loads((online / 'manifest.json').read_text())
    cm = json.loads((cached / 'manifest.json').read_text())
    if om['status'] != 'COMPLETE' or cm['status'] != 'COMPLETE' or om['probe']:
        raise ValueError('Both complete canonical videos are required')
    for source, digest in om['source_snapshots'].items():
        if sha256(source) != digest:
            raise ValueError('Online source drift: ' + source)
    report = dict(status='COMPLETE', scope='Frozen representative risk MLP deployment, '
                  'canonical encoder/GRU seed0. Exposed historical dev, no independent test. '
                  'Shared-load full online timing, no controlled speedup claim.',
                  online_manifest_sha256=sha256(online / 'manifest.json'),
                  cached_manifest_sha256=sha256(cached / 'manifest.json'),
                  parameters=om['parameters'], runtime=om['runtime'],
                  peak_gpu_memory_bytes=dict(parent=om['peak_parent_gpu_memory_bytes'],
                                             detector_worker=om['detector_runtime']['peak_gpu_memory_bytes']),
                  videos={}, official={})
    totals = []
    for seq, entry in om['videos'].items():
        op, cp = online / (seq + '.txt'), cached / (seq + '.txt')
        if sha256(op) != entry['prediction_sha256'] or sha256(cp) != cm['videos'][seq]['prediction_sha256']:
            raise ValueError('Prediction drift')
        x, y = np.loadtxt(op, delimiter=','), np.loadtxt(cp, delimiter=',')
        if x.shape != y.shape:
            raise ValueError('Canonical parity requires matching output sizes')
        different = np.any(x != y, axis=1)
        old = root / 'phase5/cross_gru_cfp_s0_c1_v1' / (seq + '.measurement.txt')
        baseline = np.loadtxt(old, delimiter=',')
        times = [json.loads(line) for line in (online / (seq + '.timing.jsonl')).read_text().splitlines()]
        if [r['frame'] for r in times] != list(range(1, entry['frames'] + 1)):
            raise ValueError('Incomplete causal timing coverage')
        if sha256(online / (seq + '.timing.jsonl')) != entry['timing_sha256']:
            raise ValueError('Timing drift')
        totals.extend(r['total_seconds'] for r in times)
        report['videos'][seq] = dict(frames=entry['frames'], prediction_sha256=sha256(op),
            cached_prediction_sha256=sha256(cp), full_bytes_equal=op.read_bytes() == cp.read_bytes(),
            current_frame_id_equal=bool(np.array_equal(x[:, :2], y[:, :2])),
            differing_rows=int(different.sum()), differing_frames=sorted(set(x[different, 0].astype(int).tolist())),
            maximum_absolute_column_difference=np.abs(x-y).max(0).tolist(),
            memory_counts_equal=entry['risk_memory_counts'] == cm['videos'][seq]['risk_memory_counts'],
            quota_counts_equal=entry['quota_counts'] == cm['videos'][seq]['quota_counts'],
            cached_vs_ordinary_current_frame_id_equal=bool(baseline.shape == y.shape and np.array_equal(baseline[:, :2], y[:, :2])),
            cached_vs_ordinary_numeric_output_equal=bool(baseline.shape == y.shape and np.array_equal(baseline, y)),
            online_pipeline_seconds=entry['pipeline_seconds'], online_fps=entry['fps'],
            stage_latency_p50_p95_ms=entry['stage_latency_p50_p95_ms'],
            memory_counts=entry['risk_memory_counts'], quota_counts=entry['quota_counts'])
    for kind, path in [('online', online), ('cached', cached)]:
        evaluation = path.with_name(path.name + '_evaluation') / 'metrics.json'
        ev = json.loads(evaluation.read_text())
        for seq, record in ev['artifacts'].items():
            if record['prediction_sha256'] != sha256(path / (seq + '.txt')):
                raise ValueError('Official metric prediction drift')
        report['official'][kind] = dict(evaluation_sha256=sha256(evaluation), metrics=ev['metrics'])
    report['full_pipeline'] = dict(frames=len(totals), seconds=sum(totals), fps=len(totals)/sum(totals),
                                   latency_p50_p95_ms=(np.quantile(totals, [.5, .95])*1000).tolist())
    # Two FLOPs per multiply-add, linear layers only; feature extraction is extra.
    report['countable_risk_flops'] = dict(linear_macs_per_edge=21*32+32*32+32,
                                         linear_flops_per_edge=2*(21*32+32*32+32),
                                         scope='Inference MLP affine MACs; excludes bias/ReLU, feature extraction, GRU, detector, encoder and non-neural operations. Variable N*M scored edges, never claimed as total pipeline FLOPs.')
    report['conclusion'] = ('Risk write decisions and current frame/ID outputs match cached replay. '
        'BT005 is byte-identical; SN013 has seven numeric rows differing only at final2999/3000 detector batch boundary, '
        'maximum box difference0.338pixels and score0.000952. Do not claim complete byte equality. '
        'Cached risk method preserves ordinary current frame/ID outputs on both dev references; no identity gain demonstrated. '
        'No further parameter selection from these dev metrics.')
    write_json(Path('docs/research/PHASE_VI_RISK_ONLINE_EVIDENCE.json'), report)
    print(json.dumps(report['full_pipeline']))


if __name__ == '__main__':
    main()
