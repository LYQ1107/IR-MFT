#!/usr/bin/env python3
"""Change crop pixels at identical known physical keys/query/gallery masks."""
import argparse
from collections import defaultdict
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

from cfp.identity_evidence import pair_discrimination, query_retrieval
from cfp.provenance import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('New paired diagnostic and explicit idle GPU required')
    spec = importlib.util.spec_from_file_location('crop_metric_helpers', Path(__file__).with_name('cfp_evaluate_crop_identity.py'))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    config = json.loads(args.config.read_text())
    meta = json.loads((args.features / 'manifest.json').read_text())
    if config['state'] != 'FROZEN' or meta['status'] != 'COMPLETE_SPARSE_TWO_CROP_FEATURES':
        raise ValueError('Complete registered sparse feature source required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(status='RUNNING', videos={}, source_sha256=sha256(__file__),
        config_sha256=sha256(args.config), feature_manifest_sha256=sha256(args.features / 'manifest.json'),
        scope='Offline uniquely-known GT/detector pair subset at identical(video,frame,fish)keys and identical per-video query/gallery masks. This isolates crop representation on covered known keys; it is not the actual all-proposal inference pool, and coverage selection is disclosed.',
        actual_pool_reference='identity_crop_retrieval_pilot_s0_v1.json keeps unknown detector proposals and distinct full view pools; compare that separately, never silently replace it with this known-only subset.')
    write_json(args.output, report)
    for video, entry in meta['videos'].items():
        directory = args.features / video
        records = json.loads((directory / 'records.json').read_text())
        if sha256(directory / 'records.json') != entry['records_sha256']:
            raise ValueError('Crop record drift')
        arrays = {}
        for name, artifact in entry['artifacts'].items():
            path = directory / (name + '.npy')
            if sha256(path) != artifact['sha256']:
                raise ValueError('Crop feature drift')
            arrays[name] = np.load(path, mmap_mode='r')
        keys_by_view = {v: defaultdict(list) for v in ('gt', 'detector')}
        for i, row in enumerate(records):
            if row['identity'] >= 0 and arrays['valid'][i]:
                keys_by_view[row['view']][(row['frame'], row['identity'])].append(i)
        intersect = set(keys_by_view['gt']) & set(keys_by_view['detector'])
        keys = sorted(key for key in intersect if len(keys_by_view['gt'][key]) == len(keys_by_view['detector'][key]) == 1)
        positions = {v: np.asarray([keys_by_view[v][key][0] for key in keys]) for v in ('gt', 'detector')}
        times = np.asarray([k[0] for k in keys])
        identities = np.asarray([k[1] for k in keys])
        queries = []
        for identity in np.unique(identities):
            observed = np.flatnonzero(identities == identity)
            select = np.unique(np.linspace(0, len(observed) - 1, min(len(observed), config['maximum_queries_per_physical_identity_per_view']), dtype=int))
            queries.extend(observed[select].tolist())
        queries.sort()
        measurements = {}
        for feature in config['features']:
            all_features = helper.descriptors(feature, arrays)
            measurements[feature] = {}
            for view in ('gt', 'detector'):
                values = all_features[positions[view]].cuda()
                metrics = defaultdict(list)
                positives, negatives = [], []
                for start in range(0, len(queries), config['batch_size']):
                    chosen = queries[start:start + config['batch_size']]
                    scores = (values[chosen] @ values.T).cpu().numpy()
                    for j, query in enumerate(chosen):
                        labels = (identities != identities[query]).astype(np.int8)
                        for gap in config['minimum_past_gaps']:
                            gallery = times <= times[query] - gap
                            metrics[str(gap)].append(query_retrieval(scores[j, gallery], labels[gallery]))
                        positive = np.flatnonzero((times < times[query]) & (identities == identities[query]))
                        if len(positive):
                            last = positive[np.argmax(times[positive])]
                            positives.append(float(scores[j, last]))
                        negatives.extend(scores[j, (times == times[query]) & (identities != identities[query])].tolist())
                measurements[feature][view] = dict(retrieval={gap: helper.summarize(rows) for gap, rows in metrics.items()},
                                                   discrimination=pair_discrimination(positives, negatives))
                del values
        report['videos'][video] = dict(paired_keys=len(keys), queries=len(queries),
            gt_known_keys=len(keys_by_view['gt']), detector_known_keys=len(keys_by_view['detector']),
            excluded_nonunique_pairs=len(intersect) - len(keys),
            measurements=measurements)
        write_json(args.output, report)
        print(video, {view: {feature: round(measurements[feature][view]['retrieval']['25']['rank1_percent'], 3) for feature in ('fish_post', 'fish_local8_concat', 'imagenet_global')}
                      for view in ('gt', 'detector')}, flush=True)
    report['status'] = 'COMPLETE_PAIRED_CROP_DIAGNOSTIC'
    write_json(args.output, report)


if __name__ == '__main__':
    main()
