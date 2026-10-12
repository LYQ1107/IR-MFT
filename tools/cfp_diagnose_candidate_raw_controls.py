#!/usr/bin/env python3
"""Same actual queries/current pools: raw RGB versus new learned coordinates."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from cfp.candidate_training_data import load_candidate_training, candidate_batch
from cfp.predictive import LatentIdentityPredictor
from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queries', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Immutable diagnostic output required')
    rows, banks, meta = load_candidate_training(args.queries, 0)
    checkpoint = Path('/data3/liuyeqiang/IR-MFT_assets/cfp/phase3/predictor_v3_gru_association_anchor_seed0/epoch8.pt')
    cm = json.loads(checkpoint.with_name('manifest.json').read_text())
    if sha256(checkpoint) != cm['checkpoint_sha256']:
        raise ValueError('Frozen ordinary GRU changed')
    model = LatentIdentityPredictor().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    torch.set_num_threads(2)
    report = dict(status='COMPLETE', query_manifest_sha256=sha256(args.queries / 'manifest.json'),
        checkpoint_sha256=sha256(checkpoint), source_snapshots=snapshot_sources(
            args.output.parent / (args.output.stem + '_sources'), [__file__, 'cfp/candidate_training_data.py', 'cfp/predictive.py']),
        videos={}, scope='Pure candidate ranking on identical actual states/pools; no new training. '
            'Old GRU fitting/calibration exposure disclosed. No NULL learning or Hungarian/MOT gain claimed. '
            'Unknown chosen candidates counted separately, not as wrong physical identities. Empty history falls back to original row cost.')
    with torch.inference_mode():
        for seq in meta['videos']:
            selected = [r for r in rows if r['sequence'] == seq]
            results = {k: dict(known_positive=0, correct=0, unknown_predicted=0, unavailable_history=0,
                N01=0, N10=0) for k in ('raw_last', 'raw_mean', 'frozen_ordinary_gru', 'original_row_cost')}
            for lo in range(0, len(selected), 128):
                chunk = selected[lo:lo + 128]
                past, times, valid, frame, candidates, legal, evidence, *_ = candidate_batch(chunk, banks, 'cpu')
                prototype, available = model(past, times, valid, frame)
                normalized = F.normalize(past, dim=-1)
                mean = F.normalize((normalized * valid[..., None]).sum(1), dim=-1)
                last = normalized[torch.arange(len(chunk)), (valid.sum(1) - 1).clamp_min(0)]
                scores = {name: torch.einsum('bd,bmd->bm', vector, F.normalize(candidates, dim=-1)).numpy()
                    for name, vector in [('raw_last', last), ('raw_mean', mean), ('frozen_ordinary_gru', prototype)]}
                for i, row in enumerate(chunk):
                    m = len(row['candidates'])
                    if row['positive'] >= m:
                        continue
                    for name, counts in results.items():
                        cost = row['evidence'][:, 19]
                        predicted = int(np.argmin(cost)) if name == 'original_row_cost' or not available[i] else int(np.argmax(scores[name][i, :m]))
                        counts['known_positive'] += 1
                        counts['unavailable_history'] += int(not available[i])
                        known = row['edge_labels'][predicted] >= 0
                        good = known and row['edge_labels'][predicted] == 0
                        counts['correct'] += int(good); counts['unknown_predicted'] += int(not known)
                        chosen = row['assigned']
                        if known and chosen >= 0 and row['edge_labels'][chosen] >= 0:
                            old_good = row['edge_labels'][chosen] == 0
                            counts['N01'] += int(not old_good and good)
                            counts['N10'] += int(old_good and not good)
            for name, counts in results.items():
                counts.update(rank1_percent=100 * counts['correct'] / max(counts['known_positive'], 1),
                    net_correction=counts['N01'] - counts['N10'])
            report['videos'][seq] = results
            print(seq, results, flush=True)
    write_json(args.output, report)


if __name__ == '__main__':
    main()
