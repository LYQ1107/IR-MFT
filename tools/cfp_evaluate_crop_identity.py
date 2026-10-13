#!/usr/bin/env python3
"""Same-video past-gallery retrieval and same-frame hard-negative metrics."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from cfp.identity_evidence import pair_discrimination, query_retrieval
from cfp.provenance import sha256, write_json


def summarize(records):
    present = [r for r in records if r['target_present']]
    margins = [r['hard_negative_margin'] for r in present if r['hard_negative_margin'] is not None]
    return dict(queries=len(records), supported_queries=len(present), target_absent=len(records) - len(present),
        rank1_percent=100 * np.mean([r['rank1'] for r in present]) if present else None,
        rank5_percent=100 * np.mean([r['rank5'] for r in present]) if present else None,
        mAP_percent=100 * np.mean([r['average_precision'] for r in present]) if present else None,
        unknown_selected=sum(r['unknown_selected'] for r in present),
        hard_negative_margin_mean=float(np.mean(margins)) if margins else None)


def descriptors(name, arrays):
    if name in ('fish_raw', 'fish_pre', 'fish_post', 'imagenet_global'):
        return F.normalize(torch.from_numpy(np.array(arrays[name])).float(), dim=-1)
    count = 4 if 'local4' in name else 8
    local = F.normalize(torch.from_numpy(np.array(arrays['fish_local' + str(count)])).float(), dim=-1)
    if name.endswith('_concat'):
        return F.normalize(local.flatten(1), dim=-1)
    if 'visibility' in name:
        visibility = torch.from_numpy(np.array(arrays['visibility8'])).float()
        value = (local * visibility[..., None]).sum(1) / visibility.sum(1).clamp_min(1e-6)[:, None]
    else:
        value = local.mean(1)
    return F.normalize(value, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Immutable metrics and explicit idle GPU required')
    config = json.loads(args.config.read_text())
    meta = json.loads((args.features / 'manifest.json').read_text())
    if config['state'] != 'FROZEN' or meta['status'] != 'COMPLETE_SPARSE_TWO_CROP_FEATURES':
        raise ValueError('Complete registered sparse features required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(status='RUNNING', config=config, config_sha256=sha256(args.config),
        source_sha256=sha256(__file__), feature_manifest_sha256=sha256(args.features / 'manifest.json'),
        fish_checkpoint_sha256=meta['fish_checkpoint_sha256'], imagenet_checkpoint_sha256=meta['imagenet_checkpoint_sha256'],
        scope=config['scope'], pair_scope=config['pair_metrics'], videos={})
    write_json(args.output, report)
    for video, entry in meta['videos'].items():
        directory = args.features / video
        if sha256(directory / 'records.json') != entry['records_sha256']:
            raise ValueError('Crop records changed')
        records = json.loads((directory / 'records.json').read_text())
        arrays = {}
        for name, artifact in entry['artifacts'].items():
            path = directory / (name + '.npy')
            if sha256(path) != artifact['sha256']:
                raise ValueError('Crop feature values changed')
            arrays[name] = np.load(path, mmap_mode='r')
        current = {}
        for view in ('gt', 'detector'):
            positions = np.asarray([i for i, r in enumerate(records) if r['view'] == view])
            times = np.asarray([records[i]['frame'] for i in positions], np.int64)
            identities = np.asarray([records[i]['identity'] for i in positions], np.int64)
            available = arrays['valid'][positions]
            queries = []
            for identity in np.unique(identities[identities >= 0]):
                observed = np.flatnonzero((identities == identity) & available)
                if len(observed):
                    selected = np.unique(np.linspace(0, len(observed) - 1, min(len(observed), config['maximum_queries_per_physical_identity_per_view']), dtype=int))
                    queries.extend(observed[selected].tolist())
            queries.sort()
            result = {}
            for name in config['features']:
                all_features = descriptors(name, arrays)
                values = all_features[positions].cuda()
                retrieval = defaultdict(list)
                positives, negatives = [], []
                pairs_by_gap = defaultdict(list)
                for start in range(0, len(queries), config['batch_size']):
                    chosen = queries[start:start + config['batch_size']]
                    scores = (values[chosen] @ values.T).cpu().numpy()
                    for j, query in enumerate(chosen):
                        labels = np.where(identities < 0, -1, (identities != identities[query]).astype(np.int8)).astype(np.int8)
                        for gap in config['minimum_past_gaps']:
                            gallery = times <= times[query] - gap
                            retrieval[str(gap)].append(query_retrieval(scores[j, gallery], labels[gallery]))
                        positive = np.flatnonzero((times < times[query]) & (identities == identities[query]) & available)
                        hard_negative = (times == times[query]) & (identities >= 0) & (identities != identities[query]) & available
                        if len(positive):
                            # All same-time proposal ordering ties are deterministic.
                            p = positive[np.argmax(times[positive])]
                            value = float(scores[j, p])
                            positives.append(value)
                            age = times[query] - times[p]
                            pairs_by_gap['1' if age == 1 else '2-25' if age <= 25 else '26-100' if age <= 100 else '101+'].append(value)
                        negatives.extend(scores[j, hard_negative].tolist())
                result[name] = dict(retrieval={gap: summarize(rows) for gap, rows in retrieval.items()},
                    discrimination=pair_discrimination(positives, negatives),
                    positive_cosine_by_actual_past_gap={gap: dict(count=len(rows), mean=float(np.mean(rows))) for gap, rows in pairs_by_gap.items()})
                del values
            current[view] = dict(observations=len(positions), known_queries=len(queries),
                unknown_observations=int((identities < 0).sum()), unavailable_observations=int((~available).sum()), features=result)
            print(video, view, {name: round(r['retrieval']['25']['rank1_percent'], 3) if r['retrieval']['25']['rank1_percent'] is not None else None for name, r in result.items()}, flush=True)
        report['videos'][video] = current
        write_json(args.output, report)
    report['status'] = 'COMPLETE_SPARSE_RETRIEVAL_DIAGNOSTIC'
    write_json(args.output, report)


if __name__ == '__main__':
    main()
