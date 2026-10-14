#!/usr/bin/env python3
"""Complete firstfold B native results; all three seeds and official metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    ap = argparse.ArgumentParser()
    for name in ('root', 'scored', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise ValueError('New immutable complete report required')
    scored = json.loads(a.scored.read_text())
    by_name = {r['name']: r for r in scored['rows']}
    modes = ['global', 'global_gru', 'local_attention', 'local_temporal', 'local_contrastive',
             'no_ema', 'no_prediction', 'no_id', 'full', 'context_full']
    result = dict(status='COMPLETE_FIRST_FOLD_B_NATIVE_THREE_SEED_CONTROLS', video='SN-015', frames=6570,
                  seeds=[0, 1, 2], runs={}, summaries={}, scored_path=str(a.scored), scored_sha256=sha256(a.scored),
                  sources=snapshot_sources(a.output.with_suffix('.sources'), [__file__]),
                  scope='Ten distinct firstfold native configurations ×three seeds ×6570frames. Current RGB frozenDINO/V3 proposals; strictly past accepted proposal histories. GT used by offline official evaluator only. Not sixfold native confirmation, rawFPS, independent external generalization, successful JEPA novelty or fullgoal completion. No lucky-seed selection. V1 and V2 noPrediction/temporal semantics differ; V2 aliases not counted twice.')
    for mode in modes:
        values = []
        for seed in result['seeds']:
            context = mode == 'context_full'
            native = f'native_identity_context_full_SN-015_s{seed}_v2' if context else f'native_identity_local_{mode}_SN-015_s{seed}_v1'
            fit = f'local_dense_context_full_SN-015_k4_s{seed}_v2' if context else f'local_dense_{mode}_SN-015_k4_s{seed}_v1'
            row = by_name[native]
            m = json.loads((a.root/native/'manifest.json').read_text())
            t = json.loads((a.root/fit/'manifest.json').read_text())
            g = t['guard']
            if (m['status'] != 'COMPLETE_NATIVE_CURRENT_IDENTITY_REPLAY' or t['status'] != 'COMPLETE'
                    or g['outer'] != 'SN-015' or g['inner'] != 'BT-001' or g['seed'] != seed
                    or 'SN-015' in g['fit'] or 'BT-001' in g['fit'] or m['context_identity'] != context
                    or sha256(a.root/fit/'final.pt') != m['model_checkpoint_sha256']
                    or row['replay_manifest_sha256'] != sha256(a.root/native/'manifest.json')
                    or sha256(a.root/native/'SN-015.txt') != m['prediction_sha256']
                    or row['metrics']['Frames'] != 6570 or m['frames'] != 6570 or m['completed_frames'] != 6570
                    or m['online_GT_teacher_future_used'] is not False
                    or m['model_manifest_sha256'] != sha256(a.root/fit/'manifest.json')
                    or m['feature_manifest_sha256'] != sha256(a.root/'native_dino_frozen_SN-015_k4_v1/manifest.json')
                    or any(sha256(p) != h for p, h in m['sources'].items())):
                raise ValueError('Exact completed native/source/excluded-fit/checkpoint provenance required: '+native)
            metrics_path = Path(row['official_metrics_path'])
            official = json.loads(metrics_path.read_text())
            if sha256(metrics_path) != row['metrics_sha256'] or official['artifacts']['SN-015']['prediction_sha256'] != m['prediction_sha256']:
                raise ValueError('Official evaluator artifact drift')
            result['runs'][native] = dict(row, training=str(a.root/fit), training_manifest_sha256=sha256(a.root/fit/'manifest.json'), guard=g)
            values.append(row['metrics'])
        keys = [k for k, v in values[0].items() if isinstance(v, (int, float))]
        result['summaries'][mode] = {k: dict(values=[v[k] for v in values], mean=float(np.mean([v[k] for v in values])),
                                           sample_sd=float(np.std([v[k] for v in values], ddof=1))) for k in keys}
    reference = by_name['native_identity_frozen_SN-015_fifo_s0_v1']
    if sha256(reference['official_metrics_path']) != reference['metrics_sha256']:
        raise ValueError('Frozen same-input reference drift')
    result['frozen_FIFO_reference'] = reference
    write_json(a.output, result)
    print(json.dumps(dict(status=result['status'], native_runs=len(result['runs']))))


if __name__ == '__main__':
    main()
