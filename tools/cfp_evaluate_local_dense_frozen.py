#!/usr/bin/env python3
"""Frozen control diagnoses without allocating any occupied GPU."""
import argparse
import json
import os
from pathlib import Path

import torch

from cfp.local_dense_data import LocalSceneBank
from cfp.local_dense_evaluation import evaluate_local_scenes
from cfp.local_dense_frozen_controls import FrozenSceneControl
from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    ap = argparse.ArgumentParser()
    for name in ('features', 'index', 'legacy-features', 'config', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or args.output.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError('New outsideGit frozen control and CPU-only required')
    config = json.loads(args.config.read_text())
    if config['state']!='FROZEN':
        raise ValueError('Same preregistered scene/query protocol required')
    torch.set_num_threads(2)
    bank = LocalSceneBank(args.features, config['first_count'], args.index, args.legacy_features)
    report = dict(status='RUNNING', config_sha256=sha256(args.config),
        feature_manifest_sha256=sha256(args.features/'manifest.json'), models={},
        sources=snapshot_sources(args.output/'sources', [__file__, args.config,
            'cfp/local_dense_frozen_controls.py', 'cfp/local_dense_data.py', 'cfp/local_dense_evaluation.py',
            'cfp/identity_evidence.py', 'cfp/provenance.py']),
        scope='Four zero-parameter same-current-scene currentcrop/pastknownphysical controls, same GT/verifieddetector histories and both current pools/unknowns/gaps/query ordinals. Not actual native tracker states; no external/pretrain zero-overlap claim.')
    write_json(args.output/'manifest.json', report)
    for mode in ('cls_last', 'cls_fifo', 'local_corresponding', 'local_flexible'):
        model = FrozenSceneControl(mode)
        assert not list(model.parameters())
        report['models'][mode] = {}
        for video in (config['first_inner'], config['first_outer']):
            measured = evaluate_local_scenes(model, bank, video, config, args.output/mode/'query_support')
            report['models'][mode][video] = measured
            write_json(args.output/'manifest.json', report)
            print(mode, video, 'GT history actual detector gap25', measured['protocol']['gt']['detector']['25']['retrieval'], flush=True)
    report.update(status='COMPLETE_FROZEN_MATCHED_LOCAL_SCENES', cuda_initialized=torch.cuda.is_initialized())
    write_json(args.output/'manifest.json', report)


if __name__ == '__main__':
    main()
