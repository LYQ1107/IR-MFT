#!/usr/bin/env python3
"""Reuse immutable physical indices with explicitly different frozen RGB coordinates."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, choices=[1, 2], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Immutable new coordinate manifest required')
    assets = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    parent = assets / 'phase6/oof_clean_bootstrap_queries_v1'
    features = assets / 'phase2' / f'causal_train_detection_features_postneck_seed{args.seed}'
    original = json.loads((parent / 'manifest.json').read_text())
    fm = json.loads((features / 'manifest.json').read_text())
    if original['status'] != 'COMPLETE' or fm['status'] != 'COMPLETE':
        raise ValueError('Complete bootstrap and seeded feature banks required')
    args.output.mkdir()
    report = copy.deepcopy(original)
    report.update(seed=args.seed, parent_manifest_sha256=sha256(parent / 'manifest.json'),
        feature_manifest_sha256=sha256(features / 'manifest.json'),
        index_reuse='Byte-identical physical GT-clean training indices, symlinked without copying crops/features; not online accepted history.',
        rgb_encoder=fm['rgb_encoder'],
        source_snapshots=snapshot_sources(args.output, [__file__, 'tools/cfp_train_oof_gru_identity_seed.py']))
    for seq, entry in report['videos'].items():
        index, feature = parent / (seq + '.npz'), features / (seq + '.npz')
        if sha256(index) != entry['sha256'] or sha256(feature) != json.loads(feature.with_suffix('.provenance.json').read_text())['features_sha256']:
            raise ValueError('Frozen physical indices or feature values drifted')
        with np.load(index, allow_pickle=False) as bank, np.load(feature, allow_pickle=False) as crops:
            size = len(crops['rgb'])
            if bank['candidates'].max() >= size or bank['history_indices'].max() >= size or crops['rgb'].shape[1] != 512:
                raise ValueError('Proposal/feature coordinate indices do not align')
        (args.output / index.name).symlink_to(index.resolve())
        entry['feature_sha256'] = sha256(feature)
    write_json(args.output / 'manifest.json', report)
    print(args.seed, 'physical index reuse validated')


if __name__ == '__main__':
    main()
