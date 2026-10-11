#!/usr/bin/env python3
"""Reuse two measured RGB feature banks for the declared capacity control."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from cfp.provenance import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--secondary', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Control feature banks are immutable')
    config = json.loads(args.config.read_text())
    sources = [json.loads((root / 'manifest.json').read_text()) for root in (args.primary, args.secondary)]
    seeds = [source['rgb_encoder']['seed'] for source in sources]
    if config['state'] != 'FROZEN' or seeds not in config['rgb_independent_repeat']['seed_pairs']:
        raise ValueError('Undeclared independent RGB seed pairing')
    for source in sources:
        if source['status'] != 'COMPLETE' or source['input_regime'] != 'causal':
            raise ValueError('Incomplete source feature banks')
        if source['rgb_encoder']['training_metadata']['input_regime'] != 'RGB current-frame only; no IR read':
            raise ValueError('Capacity control requires independently trained current-frame RGB encoders')
    for key in ['protocol_sha256', 'partition', 'detection_manifest_sha256', 'generation_manifest_sha256', 'crop_hw', 'margin', 'normalize']:
        if sources[0][key] != sources[1][key]:
            raise ValueError(f'Control source mismatch: {key}')
    if sources[0].get('embedding_stage', 'pre_neck') != sources[1].get('embedding_stage', 'pre_neck'):
        raise ValueError('Control source representation stages differ')
    if sources[0]['videos'].keys() != sources[1]['videos'].keys():
        raise ValueError('Control video coverage differs')
    args.output.mkdir(parents=True)
    manifest = dict(sources[0], status='RUNNING', auxiliary_view='rgb_independent_repeat',
                    ir_encoder=sources[1]['rgb_encoder'],
                    source_bank_sha256=[sha256(root / 'manifest.json') for root in (args.primary, args.secondary)],
                    control_config_sha256=sha256(args.config), source_sha256=sha256(__file__),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    scope='Same fixed proposals; two independent RGB encoders, no constructed IR features',
                    latency_scope='Cached composition only; original two encoder inference costs must be measured separately',
                    videos={})
    write_json(args.output / 'manifest.json', manifest)
    for sequence in sources[0]['videos']:
        banks, provenance = [], []
        for root in (args.primary, args.secondary):
            path = root / (sequence + '.npz')
            meta = json.loads(path.with_suffix('.provenance.json').read_text())
            if meta['status'] != 'COMPLETE' or sha256(path) != meta['features_sha256']:
                raise ValueError('Source feature bank changed')
            banks.append(np.load(path))
            provenance.append(meta)
        if provenance[0]['detection_sha256'] != provenance[1]['detection_sha256']:
            raise ValueError('Proposal correspondence changed')
        if banks[0]['rgb'].shape != banks[1]['rgb'].shape:
            raise ValueError('Feature dimensions disagree')
        path = args.output / (sequence + '.npz')
        np.savez_compressed(path, rgb=banks[0]['rgb'], ir=banks[1]['rgb'],
                            valid_rgb=banks[0]['valid_rgb'], valid_ir=banks[1]['valid_rgb'])
        report = dict(manifest, status='COMPLETE', sequence=sequence,
                      detection_sha256=provenance[0]['detection_sha256'], features_sha256=sha256(path),
                      detections=len(banks[0]['rgb']), source_feature_sha256=[m['features_sha256'] for m in provenance])
        write_json(path.with_suffix('.provenance.json'), report)
        manifest['videos'][sequence] = dict(features_sha256=report['features_sha256'], detections=report['detections'])
    manifest['status'] = 'COMPLETE'
    write_json(args.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
