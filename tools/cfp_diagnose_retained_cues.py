#!/usr/bin/env python3
"""Train-only utility of GT-verified cues already retained in actual memory."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from cfp.candidate_training_data import load_candidate_training
from cfp.provenance import write_json, sha256, snapshot_sources


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queries', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Immutable diagnostic output required')
    rows, banks, meta = load_candidate_training(args.queries, 0)
    record = dict(status='COMPLETE', source_snapshots=snapshot_sources(args.output.parent / (args.output.stem + '_sources'),
        [__file__, 'cfp/candidate_training_data.py']),
        scope='Offline train-only GT selection of actually retained past tokens; no extra past/future observations. '
              'Not an online method or unconditional bound. Unknown past tokens are not wrong. '
              'Conditional known-positive candidate retrieval, not MOT improvement.',
        query_manifest_sha256=sha256(args.queries / 'manifest.json'), videos={})
    for seq in meta['videos']:
        counts, gaps = Counter(), []
        for row in [r for r in rows if r['sequence'] == seq]:
            if row['positive'] >= len(row['candidates']):
                continue
            counts['known_positive'] += 1
            bank = banks[seq]
            indices = row['indices'][row['indices'] >= 0]
            chosen = indices[bank['labels'][indices] == row['identity']]
            if not len(chosen):
                counts['no_verified_cue'] += 1
                continue
            counts['retained_verified_cue'] += 1
            raw = bank['features'][chosen].mean(0)
            raw /= max(np.linalg.norm(raw), 1e-8)
            similarity = bank['features'][row['candidates']] @ raw
            predicted = int(np.argmax(similarity))
            correct = row['edge_labels'][predicted] == 0
            counts['oracle_correct'] += int(correct)
            counts['oracle_unknown_selected'] += int(row['edge_labels'][predicted] < 0)
            base = row['assigned']
            known = base >= 0 and row['edge_labels'][base] >= 0 and row['edge_labels'][predicted] >= 0
            if known:
                old = row['edge_labels'][base] == 0
                counts['N01'] += int(not old and correct); counts['N10'] += int(old and not correct)
            negative = row['edge_labels'] > 0
            if negative.any():
                gaps.append(float(similarity[row['positive']] - similarity[negative].max()))
            # Same-supported-subset comparator prevents a denominator mismatch.
            ordinary = bank['features'][indices].mean(0)
            ordinary /= max(np.linalg.norm(ordinary), 1e-8)
            ordinary_chosen = int(np.argmax(bank['features'][row['candidates']] @ ordinary))
            counts['ordinary_mean_correct_same_subset'] += int(row['edge_labels'][ordinary_chosen] == 0)
            counts['ordinary_mean_unknown_same_subset'] += int(row['edge_labels'][ordinary_chosen] < 0)
        counts['net_correction'] = counts['N01'] - counts['N10']
        record['videos'][seq] = dict(counts=dict(counts),
            retained_correct_cue_rank1_percent=100 * counts['oracle_correct'] / max(counts['retained_verified_cue'], 1),
            ordinary_mean_rank1_same_subset_percent=100 * counts['ordinary_mean_correct_same_subset'] / max(counts['retained_verified_cue'], 1),
            correct_vs_known_negative_margin_quantiles=np.quantile(gaps, [.1, .5, .9]).tolist() if gaps else None)
        print(seq, record['videos'][seq], flush=True)
    write_json(args.output, record)


if __name__ == '__main__':
    main()
