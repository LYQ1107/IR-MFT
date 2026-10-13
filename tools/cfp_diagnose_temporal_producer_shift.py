#!/usr/bin/env python3
"""Re-evaluate the frozen ordinary GRU on its original producer and pools.

Comparison with warm GRU-produced actual states is distributional, not a paired
causal experiment: different pools, cutoffs and histories are kept explicit.
"""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from cfp.identity_evidence import query_retrieval
from cfp.predictive import LatentIdentityPredictor, bounded_identity_prototype
from cfp.provenance import sha256, snapshot_sources, write_json


def summary(records):
    return dict(queries=len(records), rank1_percent=100 * np.mean([r['rank1'] for r in records]),
                mAP_percent=100 * np.mean([r['average_precision'] for r in records]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('New immutable diagnostic output required')
    if shutil.disk_usage(args.output.parent).free < 40 * 2**30:
        raise OSError('40GiB reserve')
    torch.set_num_threads(2)
    assets = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    root = assets / 'phase3/prediction_train_queries_anchor_seed0_lifecycle_fix'
    checkpoint = assets / 'phase3/predictor_v3_gru_association_anchor_seed0/epoch8.pt'
    metadata = json.loads(checkpoint.with_name('manifest.json').read_text())
    if sha256(checkpoint) != metadata['checkpoint_sha256'] or sha256(root / 'manifest.json') != metadata['query_manifest_sha256']:
        raise ValueError('Inherited GRU or its actual training query manifest changed')
    spec = importlib.util.spec_from_file_location('original_prediction_loader', Path(__file__).with_name('cfp_train_predictor.py'))
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    rows, features, manifest = loader.load_queries(root, Path('configs/cfp/plim_trial_v1.json'),
        Path('configs/cfp/plim_anchor_query_v3.json'), 'train')
    model = LatentIdentityPredictor().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    report = dict(status='RUNNING', videos={}, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        checkpoint_sha256=sha256(checkpoint), query_manifest_sha256=sha256(root / 'manifest.json'),
        exposure='Frozen RGB backbone saw all6 videos. GRU fitting5; SN-015 calibration. Fit-video results are in-sample, never encoder-level heldout generalization.',
        producer='Ordinary RGB AppearanceTracker weight0.1, prior actual accepted histories and GT-clean supervision histories; same-stage known unique physical candidate IDs with positive present, requested cutoffs1/5/15/30. No retrospective per-row geometry restriction.',
        comparison='Different original producer/pools versus warm GRU-produced histories. This is not matched counterfactual evidence of a producer effect.',
        source_snapshots=snapshot_sources(args.output.parent / (args.output.stem + '_sources'),
            [__file__, 'tools/cfp_train_predictor.py', 'tools/cfp_build_prediction_queries.py', 'cfp/predictive.py', 'cfp/identity_evidence.py']))
    write_json(args.output, report)
    with torch.inference_mode():
        for video in manifest['videos']:
            selected = [row for row in rows if row['sequence'] == video]
            results = {}
            for history in ('clean', 'actual', 'anchor_recent'):
                metrics, groups = defaultdict(list), defaultdict(lambda: defaultdict(list))
                for start in range(0, len(selected), 128):
                    batch = selected[start:start + 128]
                    past, times, valid, frame, _, candidates, positive, candidate_valid = loader.make_batch(batch, features, history, 'cpu')
                    prototypes = {name: bounded_identity_prototype(past, times, valid, frame, mode)[0]
                                  for name, mode in (('last', 'last'), ('fifo', 'fifo'), ('ema', 'ema'))}
                    prototypes['gru'] = model(past, times, valid, frame)[0]
                    scores = {name: torch.einsum('bd,bmd->bm', value, F.normalize(candidates, dim=-1)).numpy()
                              for name, value in prototypes.items()}
                    for i, row in enumerate(batch):
                        labels = np.ones(len(row['candidates']), np.int8)
                        labels[row['positive']] = 0
                        for name, values in scores.items():
                            result = query_retrieval(values[i, :len(labels)], labels)
                            metrics[name].append(result)
                            groups['requested_gap/' + str(row['gap'])][name].append(result)
                            if history != 'clean':
                                support = 'all_verified_correct' if row['purity'] == 1 else 'some_verified_correct' if row['purity'] > 0 else 'no_verified_correct'
                                # purity describes original actual history, not anchor_recent.
                                groups['original_actual_history_purity/' + support][name].append(result)
                results[history] = dict(retrieval={name: summary(records) for name, records in metrics.items()},
                    strata={group: {name: summary(records) for name, records in values.items()} for group, values in groups.items()})
            report['videos'][video] = dict(role='calibration' if video == 'SN-015' else 'fit',
                query_count=len(selected), histories=results)
            write_json(args.output, report)
            print(video, {history: {name: round(value['rank1_percent'], 4) for name, value in result['retrieval'].items()}
                          for history, result in results.items()}, flush=True)
    report['status'] = 'COMPLETE'
    write_json(args.output, report)


if __name__ == '__main__':
    main()
