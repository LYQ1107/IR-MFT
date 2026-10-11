#!/usr/bin/env python3
"""Oracle-history identity diagnosis, distinct from detector-based tracking."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from cfp.provenance import sha256, write_json


def auc(positive, negative):
    values = np.r_[positive, negative]
    ranks = rankdata(values, method='average')
    n, m = len(positive), len(negative)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * m))


def diagnostic_partition(records, rgb, ir, partition, config):
    rng = np.random.default_rng(config['seed'])
    sequences = defaultdict(list)
    for i, row in enumerate(records):
        if row['partition'] == partition:
            sequences[row['sequence']].append(i)
    reports = {}
    for sequence, ids in sequences.items():
        positions = np.array(ids)
        r, q = rgb[positions], ir[positions]
        rows = [records[i] for i in ids]
        times = np.array([row['frame_id'] for row in rows])
        labels = np.array([row['track_id'] for row in rows])
        # Deliberately corrupted correspondence is an OFFLINE intervention.
        permutation = rng.permutation(len(ids))
        wrong = q[permutation]
        noise = rng.normal(size=q.shape).astype(np.float32)
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        modes = ['rgb', 'ir', 'equal_fusion', 'wrong_ir', 'noise_ir', 'zero_ir', 'repeat_rgb']
        successes = {mode: [] for mode in modes}
        per_identity = {mode: defaultdict(list) for mode in modes}
        eligible = 0
        # Score before adding current observations; a 25-frame gap removes
        # immediately adjacent positives. Gallery carries oracle historical IDs.
        for start in range(0, len(ids), 128):
            end = min(start + 128, len(ids))
            sr, si = r[start:end] @ r.T, q[start:end] @ q.T
            sw, sn = wrong[start:end] @ wrong.T, noise[start:end] @ noise.T
            for local, current in enumerate(range(start, end)):
                gallery = np.flatnonzero(times <= times[current] - config['minimum_frame_gap'])
                if len(gallery) == 0 or labels[current] not in labels[gallery]:
                    continue
                eligible += 1
                scores = dict(rgb=sr[local], ir=si[local], equal_fusion=(sr[local] + si[local]) / 2,
                              wrong_ir=(sr[local] + sw[local]) / 2, noise_ir=(sr[local] + sn[local]) / 2,
                              zero_ir=sr[local], repeat_rgb=sr[local])
                for mode in modes:
                    winner = gallery[np.argmax(scores[mode][gallery])]
                    hit = bool(labels[winner] == labels[current])
                    successes[mode].append(hit)
                    per_identity[mode][int(labels[current])].append(hit)
        identity_rows = {int(label): np.flatnonzero(labels == label) for label in np.unique(labels)}
        positive, negative = [], []
        eligible_identities = [identity for identity, loc in identity_rows.items()
                               if times[loc].max() - times[loc].min() >= config['minimum_frame_gap']]
        if len(identity_rows) < 2 or not eligible_identities:
            raise ValueError('Insufficient identities or temporal range for pair diagnosis')
        identities = sorted(identity_rows)
        budget = config['pairs_per_class_per_sequence']
        for _ in range(budget):
            identity = int(rng.choice(eligible_identities))
            loc = identity_rows[identity]
            for attempt in range(1000):
                a, b = rng.choice(loc, 2, replace=False)
                if abs(int(times[a]) - int(times[b])) >= config['minimum_frame_gap']:
                    break
            else:
                raise ValueError('Pair sampler exhausted rather than silently changing temporal gap')
            positive.append((a, b))
            for attempt in range(1000):
                first, second = rng.choice(identities, 2, replace=False)
                a, b = int(rng.choice(identity_rows[first])), int(rng.choice(identity_rows[second]))
                if abs(int(times[a]) - int(times[b])) >= config['minimum_frame_gap']:
                    break
            else:
                raise ValueError('Negative pair sampler exhausted')
            negative.append((a, b))
        pairs = np.array(positive + negative)
        scores = {}
        for name, feature in [('rgb', r), ('ir', q), ('wrong', wrong), ('noise', noise)]:
            scores[name] = np.einsum('ij,ij->i', feature[pairs[:, 0]], feature[pairs[:, 1]])
        combined = dict(rgb=scores['rgb'], ir=scores['ir'], equal_fusion=(scores['rgb'] + scores['ir']) / 2,
                        wrong_ir=(scores['rgb'] + scores['wrong']) / 2,
                        noise_ir=(scores['rgb'] + scores['noise']) / 2,
                        zero_ir=scores['rgb'], repeat_rgb=scores['rgb'])
        reports[sequence] = dict(samples=len(ids), identities=len(identity_rows), eligible_queries=eligible,
                                query_coverage=eligible / len(ids),
                                interventions={mode: dict(causal_oracle_gallery_recall1=float(np.mean(successes[mode])),
                                      identity_macro_recall1=float(np.mean([np.mean(v) for v in per_identity[mode].values()])),
                                      pair_auc=auc(combined[mode][:budget], combined[mode][budget:]),
                                      positive_mean_cosine=float(combined[mode][:budget].mean()),
                                      negative_mean_cosine=float(combined[mode][budget:].mean())) for mode in modes},
                                fixed_fusion_corrections=int(np.sum(np.array(successes['equal_fusion']) & ~np.array(successes['rgb']))),
                                fixed_fusion_harms=int(np.sum(~np.array(successes['equal_fusion']) & np.array(successes['rgb']))))
        print(sequence, 'queries', eligible,
              {m: round(reports[sequence]['interventions'][m]['causal_oracle_gallery_recall1'], 4) for m in modes}, flush=True)
    return reports


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features-root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--partition', choices=['train', 'dev'], default='dev')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    if config['state'] != 'FROZEN':
        raise ValueError('Predeclare the diagnostic sampler')
    manifest = json.loads((args.features_root / 'manifest.json').read_text())
    if manifest['status'] != 'COMPLETE':
        raise ValueError('Incomplete features')
    feature_path = args.features_root / 'features.npz'
    if sha256(feature_path) != manifest['features_sha256']:
        raise ValueError('Feature hash changed')
    index = json.loads((args.features_root / 'index.json').read_text())
    if sha256(args.features_root / 'index.json') != manifest['index_sha256']:
        raise ValueError('Feature-to-record index changed')
    feature = np.load(feature_path)
    result = diagnostic_partition(index['records'], feature['rgb'], feature['ir'], args.partition, config)
    write_json(args.output, dict(config=config, partition=args.partition, features_manifest_sha256=sha256(args.features_root / 'manifest.json'),
                                source_sha256=sha256(__file__), videos=result,
                                scope=manifest['scientific_scope'],
                                limitations=['GT crops and oracle gallery identity labels; not measured tracking performance',
                                             'Wrong/noise IR are offline feature interventions, not acquired modalities',
                                             'Repeated RGB and zero IR reduce algebraically to the RGB cosine score',
                                             'Candidate gallery density varies by sequence; report per-sequence and identity macro scores',
                                             'Frozen ImageNet encoder is not a fish-finetuned ReID baseline']))


if __name__ == '__main__':
    main()
