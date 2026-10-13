"""Collect the registered first-fold visual controls across all three seeds."""
import argparse
import json
from pathlib import Path

import numpy as np

from cfp.provenance import sha256, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    actual_path = args.assets / 'identity_crop_retrieval_pilot_s0_v1.json'
    actual = json.loads(actual_path.read_text())
    rows, proofs = {}, {}
    result = dict(status='COMPLETE_FIRST_FOLD_VISUAL_THREE_SEED_CONTROLS', research_status='IN_PROGRESS',
        seeds=[0, 1, 2], fit=['BT-003', 'MSK-002', 'PF-001', 'SN-001'], inner='BT-001', outer='SN-015',
        fixed_selection='Registered fixed last epoch8 for every model/seed; no outer hyperparameter/checkpoint selection',
        metric_scope='Sparse past-only current crop retrieval at gaps1/25/100, retaining unknown detector gallery rows. Known-positive-supported conditional R1/R5/mAP; no native MOT or new-coordinate GRU claim.',
        frozen_baseline={v: {view: actual['videos'][v][view]['features']['imagenet_global']['retrieval'] for view in ('gt', 'detector')} for v in ('BT-001', 'SN-015')},
        baseline_proof={'path': str(actual_path), 'sha256': sha256(actual_path)}, runs=rows, summary={}, proofs=proofs)
    for mode in ('global', 'learned'):
        values = []
        for seed in (0, 1, 2):
            name = f'visual_identity_{mode}_SN-015_s{seed}_v1'
            path = args.assets / name / 'manifest.json'
            d = json.loads(path.read_text())
            if (d['status'] != 'COMPLETE' or d['guard']['seed'] != seed or d['guard']['mode'] != mode
                    or d['config']['fit'] != result['fit'] or d['config']['inner'] != result['inner']
                    or d['config']['outer'] != result['outer'] or len(d['epochs']) != 8):
                raise ValueError('Only exact registered completed split/model/seed/epochs')
            rows[name] = {k: d[k] for k in ('status', 'guard', 'epochs', 'evaluations', 'parameters', 'training_seconds', 'checkpoint_sha256', 'peak_gpu_allocated_bytes')}
            proofs[name] = {'path': str(path), 'sha256': sha256(path)}
            values.append([d['evaluations'][v][view]['25']['rank1_percent'] for v in ('BT-001', 'SN-015') for view in ('gt', 'detector')])
        result['summary'][mode] = dict(column_order=['inner_gt', 'inner_detector', 'outer_gt', 'outer_detector'],
            rank1_gap25_mean_percent=np.mean(values, axis=0).tolist(), rank1_gap25_sample_std_percent=np.std(values, axis=0, ddof=1).tolist(),
            scope='Sample SD across3registeredtrainingseeds, not perquery independence or confidence/significance certification')
    result['conclusion'] = 'Neither new trained head beats frozen rawImageNet outer detectorR1. Seed0 local-versus-global difference is not stable across3seeds; no retained visual gain or universal local superiority.'
    write_json(args.output, result)
    print(json.dumps({'status': result['status'], 'summary': result['summary']}))


if __name__ == '__main__':
    main()
