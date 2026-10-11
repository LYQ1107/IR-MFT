#!/usr/bin/env python3
"""Pair trained RGB/causal-IR embeddings for offline GT identity diagnosis."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from cfp.provenance import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb-run', type=Path, required=True)
    parser.add_argument('--ir-run', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = [json.loads((root / 'manifest.json').read_text()) for root in (args.rgb_run, args.ir_run)]
    for source, root in zip(sources, (args.rgb_run, args.ir_run)):
        if source['status'] != 'COMPLETE' or source['index_sha256'] != sha256(args.index):
            raise ValueError('Incomplete/unaligned identity features')
        if sha256(root / 'features.npz') != source['features_sha256']:
            raise ValueError('Identity features changed')
    if sources[0]['seed'] != sources[1]['seed'] or sources[0]['config_sha256'] != sources[1]['config_sha256']:
        raise ValueError('Main paired diagnostic requires matched encoder seed and budget')
    if sources[0]['input_regime'] != 'RGB current-frame only; no IR read' or sources[1]['input_regime'] != 'complete causal constructed IR':
        raise ValueError('Expected current RGB and causal IR encoders')
    features = [np.load(root / 'features.npz')['embedding'] for root in (args.rgb_run, args.ir_run)]
    if features[0].shape != features[1].shape or len(features[0]) != len(json.loads(args.index.read_text())['records']):
        raise ValueError('Feature correspondence changed')
    args.output.mkdir(parents=True)
    shutil.copyfile(args.index, args.output / 'index.json')
    np.savez_compressed(args.output / 'features.npz', rgb=features[0], ir=features[1])
    write_json(args.output / 'manifest.json', dict(status='COMPLETE', seed=sources[0]['seed'],
               features_sha256=sha256(args.output / 'features.npz'), index_sha256=sha256(args.index),
               source_manifests_sha256=[sha256(root / 'manifest.json') for root in (args.rgb_run, args.ir_run)],
               encoders=sources, input_regime='causal', source_sha256=sha256(__file__),
               git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               scientific_scope='Fish-trained RGB/causal-IR GT crops with oracle whole-lifespan sampling and prior labeled gallery; offline diagnosis, not tracking'))


if __name__ == '__main__':
    main()
