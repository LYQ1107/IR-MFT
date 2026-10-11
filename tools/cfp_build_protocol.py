#!/usr/bin/env python3
"""Freeze video-disjoint association protocol using training metadata only."""
import argparse
from collections import defaultdict
import configparser
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_partition(data, train, dev):
    names = {v['id']: v['file_name'] for v in data['videos']}
    if set(train) & set(dev) or set(train) | set(dev) != set(names.values()):
        raise ValueError('Partitions must cover all videos without overlap')
    image_video = {im['id']: names[im['video_id']] for im in data['images']}
    if len(image_video) != len(data['images']):
        raise ValueError('Duplicate image IDs')
    keys = defaultdict(set)
    for ann in data['annotations']:
        video = image_video[ann['image_id']]
        keys['train' if video in train else 'dev'].add((video, ann['track_id']))
    assert not keys['train'] & keys['dev']
    for vid in names:
        frames = [im['frame_id'] for im in data['images'] if im['video_id'] == vid]
        assert frames == list(range(1, len(frames) + 1)), names[vid]
    return {part: len(values) for part, values in keys.items()}


def audit_video(root, name, images):
    gt = np.loadtxt(root / 'train' / name / 'gt/gt.txt', delimiter=',', ndmin=2)
    gt = gt[gt[:, 6] > 0]
    by_frame = defaultdict(list)
    for row in gt:
        by_frame[int(row[0])].append(row[2:6])
    overlaps, densities = [], []
    for frame in range(1, len(images) + 1):
        boxes = np.array(by_frame[frame]).reshape(-1, 4)
        densities.append(len(boxes))
        if len(boxes) < 2:
            overlaps.append(0.0)
            continue
        xy1, xy2 = boxes[:, :2], boxes[:, :2] + boxes[:, 2:]
        wh = np.maximum(0, np.minimum(xy2[:, None], xy2) - np.maximum(xy1[:, None], xy1))
        inter = wh.prod(axis=-1)
        area = boxes[:, 2:].prod(axis=-1)
        iou = inter / np.maximum(area[:, None] + area - inter, 1e-8)
        np.fill_diagonal(iou, 0)
        overlaps.append(float(np.mean(iou.max(axis=1) > .1)))
    sampled = []
    for index in np.unique(np.linspace(0, len(images) - 1, min(24, len(images)), dtype=int)):
        im = images[index]
        rgb, ir = [cv2.imread(str(root / 'train' / im[k])) for k in ('vi_file_name', 'ir_file_name')]
        if rgb is None or ir is None or rgb.shape != ir.shape:
            raise ValueError(f'Invalid pair {name}:{im["frame_id"]}')
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        sampled.append({'frame_id': im['frame_id'], 'rgb_mean_gray': float(gray.mean()),
                        'rgb_std_gray': float(gray.std()),
                        'ir_mean_channels_bgr': ir.mean(axis=(0, 1)).tolist()})
    return {'frames': len(images), 'identities': len(np.unique(gt[:, 1])),
            'gt_rows': len(gt), 'gt_sha256': digest(root / 'train' / name / 'gt/gt.txt'),
            'density_mean': float(np.mean(densities)), 'density_max': max(densities),
            'mean_fraction_with_gt_box_iou_above_0_1': float(np.mean(overlaps)),
            'overlap_is_proxy_not_occlusion_annotation': True, 'sampled_lighting': sampled,
            'species': 'unverified', 'original_recording_and_physical_fish_overlap': 'unverified'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.data_root / 'annotations/train.json').read_text())
    dev = ['BT-005', 'SN-013']
    train = [v['file_name'] for v in data['videos'] if v['file_name'] not in dev]
    counts = check_partition(data, train, dev)
    by_video = defaultdict(list)
    for image in data['images']:
        by_video[image['video_id']].append(image)
    audit = {v['file_name']: audit_video(args.data_root, v['file_name'], by_video[v['id']])
             for v in data['videos']}
    test = json.loads((args.data_root / 'annotations/test.json').read_text())
    protocol = {
        'schema_version': 1, 'protocol_id': 'cfp_video_holdout_v1', 'state': 'FROZEN',
        'train': train, 'dev': dev, 'historical_test': [v['file_name'] for v in test['videos']],
        'identity_key': ['sequence', 'track_id'], 'identity_counts': counts,
        'frame_counts': {p: sum(audit[v]['frames'] for v in vs) for p, vs in [('train', train), ('dev', dev)]},
        'train_annotation_sha256': digest(args.data_root / 'annotations/train.json'),
        'test_annotation_sha256': digest(args.data_root / 'annotations/test.json'),
        'selection_rationale': 'Median-duration video in each repeated BT/SN prefix group; preserve singleton MSK/PF groups for training. Chosen before association outcomes.',
        'selection_limitations': ['Prefix is not verified species/recording metadata',
                                  'Dev excludes the 49-identity dense MSK scene',
                                  'Historical detector trained on all eight train videos; only new identity/association modules held out',
                                  'Historical test was tuned previously; not independent blind evaluation'],
        'gt_hashes': {v: x['gt_sha256'] for v, x in audit.items()},
        'crossfit_groups': [['BT-001', 'PF-001', 'SN-001'], ['BT-003', 'MSK-002', 'SN-015']],
        'supplemental_video_cv': [['BT-001', 'SN-015'], ['BT-003', 'SN-001'], ['MSK-002', 'PF-001']],
        'seed_plan': [0, 1, 2], 'checkpoint_selection': 'dev HOTA, then AssA, then IDF1; never historical_test',
        'success_targets_pp': {'HOTA': 1.0, 'AssA': 2.0, 'IDF1': 2.0},
        'success_constraints': 'Three seeds versus strongest reproducible Protocol A baseline; independent CMUE/PLIM ablation benefit; IDSW not increased; MOTA decrease <=0.5pp; end-to-end latency <=2x baseline. Targets aspirational, all tradeoffs reported.',
        'search_budget': 'Initial baseline settings + at most three targeted repair rounds per core hypothesis; record all trials; no test selection',
        'evaluation': {'tool': 'TrackEval', 'commit': '12c8791b303e0a0b50f753af204249e622d0281a',
                       'metrics': ['HOTA', 'CLEAR', 'Identity'], 'class': 'pedestrian (fish mapped to class 1)',
                       'hota_alpha': [round(x / 100, 2) for x in range(5, 100, 5)],
                       'clear_identity_threshold': .5, 'aggregate': 'official combine_sequences',
                       'preprocess': 'keep GT marked=1,class=1; no distractors exist; parity required with historical fork'},
        'ir_generation_provenance': 'PENDING; no sensor or generator claim without evidence',
    }
    args.protocol.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.protocol.write_text(json.dumps(protocol, indent=2) + '\n')
    args.audit.write_text(json.dumps({'protocol_sha256': digest(args.protocol), 'videos': audit}, indent=2) + '\n')
    print(json.dumps({'protocol': str(args.protocol), 'sha256': digest(args.protocol),
                      'identities': counts, 'frames': protocol['frame_counts'],
                      'audit': {v: {k: x[k] for k in ('frames', 'identities', 'density_mean', 'mean_fraction_with_gt_box_iou_above_0_1')} for v, x in audit.items()}}, indent=2))


if __name__ == '__main__':
    main()
