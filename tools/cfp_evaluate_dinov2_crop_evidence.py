#!/usr/bin/env python3
"""Matched past retrieval and paired-crop diagnosis, with unknowns retained."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from cfp.identity_evidence import pair_discrimination, query_retrieval
from cfp.provenance import sha256, snapshot_sources, write_json


def query_positions(identities, valid, cap):
    queries = []
    for identity in np.unique(identities[identities >= 0]):
        observed = np.flatnonzero((identities == identity) & valid)
        if len(observed):
            selected = np.unique(np.linspace(0, len(observed)-1, min(len(observed), cap), dtype=int))
            queries.extend(observed[selected].tolist())
    return sorted(queries)


def summarize(rows):
    supported = [r for r in rows if r['target_present']]
    return dict(queries=len(rows), supported_queries=len(supported), target_absent=len(rows)-len(supported),
        rank1_percent=100*np.mean([r['rank1'] for r in supported]) if supported else None,
        rank5_percent=100*np.mean([r['rank5'] for r in supported]) if supported else None,
        mAP_percent=100*np.mean([r['average_precision'] for r in supported]) if supported else None,
        unknown_selected=sum(r['unknown_selected'] for r in supported))


def measure(values, records, valid, config):
    times = np.array([r['frame'] for r in records])
    identities = np.array([r['identity'] for r in records])
    queries = query_positions(identities, valid, config['maximum_queries_per_physical_identity_per_view'])
    values = values.cuda()
    ranks = defaultdict(list)
    positive, negative = [], []
    by_gap = defaultdict(list)
    with torch.inference_mode():
        for start in range(0, len(queries), config['batch_size']):
            chosen = queries[start:start+config['batch_size']]
            scores = (values[chosen] @ values.T).cpu().numpy()
            for j, q in enumerate(chosen):
                labels = np.where(identities < 0, -1, (identities != identities[q]).astype(np.int8)).astype(np.int8)
                for gap in config['minimum_past_gaps']:
                    gallery = times <= times[q]-gap
                    ranks[str(gap)].append(query_retrieval(scores[j, gallery], labels[gallery]))
                past = np.flatnonzero((times < times[q]) & (identities == identities[q]) & valid)
                if len(past):
                    p = past[np.argmax(times[past])]
                    positive.append(float(scores[j, p]))
                    age = times[q]-times[p]
                    by_gap['1' if age == 1 else '2-25' if age <= 25 else '26-100' if age <= 100 else '101+'].append(float(scores[j, p]))
                hard = (times == times[q]) & (identities >= 0) & (identities != identities[q]) & valid
                negative.extend(scores[j, hard].tolist())
    return dict(retrieval={gap: summarize(rows) for gap, rows in ranks.items()},
        discrimination=pair_discrimination(positive, negative),
        positive_cosine_by_actual_past_gap={gap: dict(count=len(rows), mean=float(np.mean(rows))) for gap, rows in by_gap.items()})


def descriptors(name, arrays):
    if name in ('fish_post', 'imagenet_global', 'dino_cls', 'dino_patch_mean'):
        return F.normalize(torch.from_numpy(np.array(arrays[name])).float(), dim=-1)
    count = 4 if 'local4' in name else 8
    tokens = F.normalize(torch.from_numpy(np.array(arrays['dino_local'+str(count)])).float(), dim=-1)
    if name.endswith('_concat'):
        return F.normalize(tokens.flatten(1), dim=-1)
    visibility = torch.from_numpy(np.array(arrays['dino_visibility8'])).float()
    return F.normalize((tokens*visibility[..., None]).sum(1)/visibility.sum(1).clamp_min(1e-6)[:, None], dim=-1)


def main():
    ap = argparse.ArgumentParser()
    for name in ('features', 'legacy-features', 'config', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if args.output.exists() or physical not in ('0', '1', '2', '3'):
        raise ValueError('Immutable metrics and one approved idle GPU required')
    memory, utilization = [int(x.strip()) for x in subprocess.check_output(['nvidia-smi', '-i', physical,
        '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).split(',')]
    if memory > 64 or utilization:
        raise ValueError('Chosen evaluation GPU is not idle')
    config = json.loads(args.config.read_text())
    meta = json.loads((args.features/'manifest.json').read_text())
    old = json.loads((args.legacy_features/'manifest.json').read_text())
    if config['state'] != 'FROZEN' or meta['status'] != 'COMPLETE_FROZEN_DINOV2_TWO_CROP_EVIDENCE':
        raise ValueError('Complete preregistered feature extraction required')
    if meta['original_feature_manifest_sha256'] != sha256(args.legacy_features/'manifest.json'):
        raise ValueError('Same original crops/pools required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(status='RUNNING', config=config, config_sha256=sha256(args.config),
        feature_manifest_sha256=sha256(args.features/'manifest.json'),
        legacy_manifest_sha256=sha256(args.legacy_features/'manifest.json'), videos={},
        sources=snapshot_sources(args.output.with_suffix('.sources'), [__file__, args.config,
            'cfp/identity_evidence.py']), scope=config['scope'])
    write_json(args.output, report)
    for video, e in meta['videos'].items():
        directory, legacy = args.features/video, args.legacy_features/video
        if (sha256(directory/'records.json') != e['records_sha256']
                or e['records_sha256'] != old['videos'][video]['records_sha256']
                or sha256(directory/'valid.npy') != e['artifacts']['valid']['sha256']):
            raise ValueError('New/old crop record or validity alignment differs')
        records = json.loads((directory/'records.json').read_text())
        valid = np.load(directory/'valid.npy', allow_pickle=False)
        arrays = {}
        for name in ('cls', 'patch_mean', 'local4', 'local8', 'visibility8'):
            path = directory/(name+'.npy')
            if sha256(path) != e['artifacts'][name]['sha256']:
                raise ValueError('Pretrained features drift')
            arrays['dino_'+name] = np.load(path, mmap_mode='r')
        for name in ('fish_post', 'imagenet_global'):
            path = legacy/(name+'.npy')
            if sha256(path) != old['videos'][video]['artifacts'][name]['sha256']:
                raise ValueError('Legacy baseline drift')
            arrays[name] = np.load(path, mmap_mode='r')
        values = {name: descriptors(name, arrays) for name in config['features']}
        result = {'actual_pool': {}, 'paired': {}}
        by_view = {view: np.array([i for i, row in enumerate(records) if row['view'] == view], dtype=np.int64)
                   for view in ('gt', 'detector')}
        for view, pos in by_view.items():
            rows = [records[i] for i in pos]
            result['actual_pool'][view] = {name: measure(value[pos], rows, valid[pos], config) for name, value in values.items()}
        keys = {view: defaultdict(list) for view in by_view}
        for i, row in enumerate(records):
            if row['identity'] >= 0 and valid[i]:
                keys[row['view']][(row['frame'], row['identity'])].append(i)
        intersect = set(keys['gt']) & set(keys['detector'])
        paired = sorted(k for k in intersect if len(keys['gt'][k]) == len(keys['detector'][k]) == 1)
        result['paired_keys'] = len(paired)
        result['excluded_nonunique_pairs'] = len(intersect)-len(paired)
        for view in by_view:
            pos = np.array([keys[view][key][0] for key in paired], dtype=np.int64)
            rows = [records[i] for i in pos]
            result['paired'][view] = {name: measure(value[pos], rows, valid[pos], config) for name, value in values.items()}
        report['videos'][video] = result
        write_json(args.output, report)
        print(video, 'paired detector gap25', {name: round(result['paired']['detector'][name]['retrieval']['25']['rank1_percent'], 3)
            for name in config['features']}, flush=True)
    report['status'] = 'COMPLETE_MATCHED_FROZEN_DINOV2_CROP_DIAGNOSTIC'
    write_json(args.output, report)


if __name__ == '__main__':
    main()
