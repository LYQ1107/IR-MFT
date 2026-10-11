#!/usr/bin/env python3
"""Create immutable disjoint-video teacher crop views without copying arrays."""
import argparse
import json
from pathlib import Path
import subprocess

from cfp.identity_data import build_identity_index
from cfp.provenance import sha256, write_json


def observation_signature(row):
    return {key: value for key, value in row.items() if key not in ('partition', 'train_label')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--rgb-crops', type=Path, required=True)
    parser.add_argument('--ir-crops', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Crossfit crop views are immutable')
    parent = json.loads(args.protocol.read_text())
    groups = parent['crossfit_groups']
    if len(groups) != 2 or set(groups[0]) & set(groups[1]) or set(groups[0] + groups[1]) != set(parent['train']):
        raise ValueError('Expected the predeclared two disjoint teacher groups')
    sources = {}
    for modality, root in [('rgb', args.rgb_crops), ('ir', args.ir_crops)]:
        meta = json.loads((root / 'manifest.json').read_text())
        if meta['status'] != 'COMPLETE' or meta['protocol_sha256'] != sha256(args.protocol):
            raise ValueError('Incomplete or changed teacher crop source')
        if sha256(root / 'crops.npy') != meta['crops_sha256'] or sha256(root / 'index.json') != meta['index_sha256']:
            raise ValueError('Teacher source values changed')
        index = json.loads((root / 'index.json').read_text())
        sources[modality] = root, meta, index
    if sources['rgb'][1]['index_sha256'] != sources['ir'][1]['index_sha256']:
        raise ValueError('Unpaired teacher crops')
    if sources['ir'][1]['input_regime'] != 'complete causal constructed IR':
        raise ValueError('Crossfit IR must be causal')
    args.output.mkdir(parents=True)
    reports = []
    for fold in range(2):
        directory = args.output / f'fold{fold}'
        protocol = dict(parent, protocol_id=f'cfp_crossfit_teacher_fold{fold}',
                        train=groups[fold], dev=groups[1 - fold] + parent['dev'],
                        teacher_heldout=groups[1 - fold],
                        canonical_dev=parent['dev'], parent_protocol_sha256=sha256(args.protocol),
                        role='Out-of-fold identity teacher only; not a replacement association development protocol',
                        checkpoint_selection='Fixed last epoch; no teacher heldout or canonical dev checkpoint selection',
                        initialization='Original ImageNet only; never a fish model trained on the heldout group')
        protocol.pop('identity_counts')
        protocol.pop('frame_counts')
        protocol_path = directory / 'protocol.json'
        write_json(protocol_path, protocol)
        index = build_identity_index(args.data_root, protocol_path)
        originals = sources['rgb'][2]['records']
        if ([observation_signature(row) for row in index['records']] !=
                [observation_signature(row) for row in originals]):
            raise ValueError('Teacher label split changed crop values/order')
        for modality, (root, meta, _) in sources.items():
            target = directory / modality
            target.mkdir(parents=True)
            (target / 'crops.npy').symlink_to(root / 'crops.npy')
            write_json(target / 'index.json', index)
            view = dict(meta, protocol_sha256=sha256(protocol_path), index_sha256=sha256(target / 'index.json'),
                        parent_crop_manifest_sha256=sha256(root / 'manifest.json'),
                        source_sha256=sha256(__file__), git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                        scope='Same crop values/order; only declared video membership and classifier labels changed',
                        fold=fold, teacher_heldout=groups[1 - fold], training_videos=groups[fold],
                        initialization_requirement='Original ImageNet only')
            write_json(target / 'manifest.json', view)
        reports.append(dict(fold=fold, protocol_path=str(protocol_path), protocol_sha256=sha256(protocol_path),
                            train_identity_count=index['train_identity_count'], heldout=groups[1 - fold]))
    write_json(args.output / 'manifest.json', dict(status='COMPLETE', parent_protocol_sha256=sha256(args.protocol),
               source_sha256=sha256(__file__), folds=reports,
               scope='OOF teacher views prepared; no teacher trained and no counterfactual utility target yet'))
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
