#!/usr/bin/env python3
"""Same-pool heldout diagnosis of real trained DINO crop evidence."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import runpy
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    ap = argparse.ArgumentParser()
    for name in ('training', 'frozen-features', 'frozen-diagnostic', 'config', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if args.output.exists() or physical not in ('0', '1', '2', '3'):
        raise ValueError('New immutable output and exactly one approved idle GPU required')
    memory, utilization = [int(x.strip()) for x in subprocess.check_output(['nvidia-smi', '-i', physical,
        '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).split(',')]
    if memory > 64 or utilization:
        raise ValueError('Evaluation GPU not idle')
    config = json.loads(args.config.read_text())
    training = json.loads((args.training/'manifest.json').read_text())
    frozen = json.loads((args.frozen_features/'manifest.json').read_text())
    reference = json.loads(args.frozen_diagnostic.read_text())
    if (config['state'] != 'FROZEN' or training['status'] != 'COMPLETE'
            or reference['status'] != 'COMPLETE_MATCHED_FROZEN_DINOV2_CROP_DIAGNOSTIC'
            or training['guard']['features_sha256'] != sha256(args.frozen_features/'manifest.json')
            or reference['feature_manifest_sha256'] != sha256(args.frozen_features/'manifest.json')
            or training['checkpoint_sha256'] != sha256(args.training/'final.pt')):
        raise ValueError('Completed same-input producer and frozen reference required')
    names = config['features'] if training['guard']['mode'] != 'global' else config['features'][:3]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Reuse the exact executed frozen diagnostic operator and snapshot its code.
    operator_path = Path(__file__).with_name('cfp_evaluate_dinov2_crop_evidence.py')
    measure = runpy.run_path(str(operator_path))['measure']
    report = dict(status='RUNNING', guard=training['guard'], config=config,
        training_manifest_sha256=sha256(args.training/'manifest.json'),
        frozen_diagnostic_sha256=sha256(args.frozen_diagnostic), config_sha256=sha256(args.config),
        sources=snapshot_sources(args.output.with_suffix('.sources'), [__file__, operator_path,
            args.config, 'cfp/identity_evidence.py']), scope=config['scope'], videos={})
    write_json(args.output, report)
    for video, metadata in training['evaluations'].items():
        directory = args.training/video
        records = json.loads((directory/'records.json').read_text())
        if sha256(directory/'records.json') != metadata['records_sha256'] or metadata['records_sha256'] != frozen['videos'][video]['records_sha256']:
            raise ValueError('Changed current-crop physical pools')
        valid_path = args.frozen_features/video/'valid.npy'
        if sha256(valid_path) != frozen['videos'][video]['artifacts']['valid']['sha256']:
            raise ValueError('Changed current-crop support')
        valid = np.load(valid_path, allow_pickle=False)
        arrays = {}
        for name, entry in metadata['artifacts'].items():
            path = directory/(name+'.npy')
            if sha256(path) != entry['sha256']:
                raise ValueError('Changed final-epoch trained evidence')
            arrays[name] = np.load(path, mmap_mode='r')
        values = {n: torch.from_numpy(np.array(arrays[n])).float() for n in names if n in arrays}
        if 'local_concat' in names:
            tokens = torch.from_numpy(np.array(arrays['local_tokens'])).float()
            weights = torch.from_numpy(np.array(arrays['local_weights'])).float()
            available = torch.from_numpy(np.array(arrays['local_valid'])).bool()
            if (weights[~available].abs() > 1.e-6).any():
                raise ValueError('Absent local regions received confidence mass')
            values['local_concat'] = F.normalize((tokens*available[..., None]).flatten(1), dim=-1)
            values['local_reliability_mean'] = F.normalize((tokens*weights[..., None]).sum(1), dim=-1)
        frozen_path = args.frozen_features/video/'cls.npy'
        if sha256(frozen_path) != frozen['videos'][video]['artifacts']['cls']['sha256']:
            raise ValueError('Changed official frozen baseline')
        values['frozen_cls'] = F.normalize(torch.from_numpy(np.array(np.load(frozen_path, mmap_mode='r'))).float(), dim=-1)
        by_view = {v: np.array([i for i, row in enumerate(records) if row['view'] == v], dtype=np.int64)
                   for v in ('gt', 'detector')}
        keys = {v: defaultdict(list) for v in by_view}
        for i, row in enumerate(records):
            if row['identity'] >= 0 and valid[i]:
                keys[row['view']][row['frame'], row['identity']].append(i)
        intersect = set(keys['gt']) & set(keys['detector'])
        paired = sorted(k for k in intersect if len(keys['gt'][k]) == len(keys['detector'][k]) == 1)
        result = dict(actual_pool={}, paired={}, paired_keys=len(paired), excluded_nonunique_pairs=len(intersect)-len(paired))
        for pool in ('actual_pool', 'paired'):
            for view in by_view:
                pos = by_view[view] if pool == 'actual_pool' else np.array([keys[view][key][0] for key in paired], dtype=np.int64)
                rows = [records[i] for i in pos]
                result[pool][view] = {n: measure(v[pos], rows, valid[pos], config) for n, v in values.items()}
                expected = reference['videos'][video][pool][view]['dino_cls']
                if result[pool][view]['frozen_cls'] != expected:
                    raise ValueError('Frozen reference exact metric/sample parity failed')
                if pool == 'actual_pool':
                    for gap, entry in result[pool][view]['embedding']['retrieval'].items():
                        source_entry = dict(metadata['retrieval'][view][gap])
                        source_entry['target_absent'] = source_entry.pop('missing_target')
                        if entry != source_entry:
                            raise ValueError('Training evaluator exact retrieval parity failed')
        if result['paired_keys'] != reference['videos'][video]['paired_keys']:
            raise ValueError('Changed paired physical identity keys')
        report['videos'][video] = result
        write_json(args.output, report)
        print(video, 'paired detector gap25', {n: result['paired']['detector'][n]['retrieval']['25']['rank1_percent'] for n in values}, flush=True)
    report['status'] = 'COMPLETE_MATCHED_TRAINED_DINOV2_CROP_DIAGNOSTIC'
    report['exact_frozen_reference_and_training_retrieval_parity'] = True
    write_json(args.output, report)


if __name__ == '__main__':
    main()
