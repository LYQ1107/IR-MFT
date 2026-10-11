"""Paired identity samples with video namespaces and temporal coverage."""
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .provenance import sha256


def build_identity_index(data_root, protocol_path, samples_per_identity=128):
    root = Path(data_root)
    protocol = json.loads(Path(protocol_path).read_text())
    data = json.loads((root / 'annotations/train.json').read_text())
    names = {v['id']: v['file_name'] for v in data['videos']}
    images = {x['id']: x for x in data['images']}
    records = defaultdict(list)
    for ann in data['annotations']:
        image = images[ann['image_id']]
        seq = names[image['video_id']]
        if ann.get('conf', 1) <= 0 or ann['bbox'][2] <= 0 or ann['bbox'][3] <= 0:
            continue
        if seq not in protocol['train'] + protocol['dev']:
            raise ValueError('Unexpected training video')
        if Path(image['vi_file_name']).stem != Path(image['ir_file_name']).stem:
            raise ValueError('Unpaired image names')
        records[(seq, int(ann['track_id']))].append({
            'sequence': seq, 'track_id': int(ann['track_id']), 'frame_id': int(image['frame_id']),
            'image_id': int(image['id']), 'rgb_file': image['vi_file_name'], 'ir_file': image['ir_file_name'],
            'bbox_xywh': ann['bbox'], 'partition': 'train' if seq in protocol['train'] else 'dev'})
    selected = []
    excluded = []
    train_keys = sorted(k for k in records if k[0] in protocol['train'])
    label_map = {key: i for i, key in enumerate(train_keys)}
    for key, observations in sorted(records.items()):
        observations.sort(key=lambda r: r['frame_id'])
        frame_counts = defaultdict(int)
        for row in observations:
            frame_counts[row['frame_id']] += 1
        duplicate_frames = {frame for frame, count in frame_counts.items() if count > 1}
        if duplicate_frames:
            if protocol.get('identity_annotation_policy') != 'exclude_all_duplicate_keys':
                raise ValueError('Duplicate frame/identity annotation')
            excluded.extend(dict(sequence=key[0], track_id=key[1], frame_id=frame, rows=frame_counts[frame]) for frame in sorted(duplicate_frames))
            observations = [row for row in observations if row['frame_id'] not in duplicate_frames]
        indices = np.unique(np.linspace(0, len(observations) - 1,
                                        min(samples_per_identity, len(observations)), dtype=int))
        for index in indices:
            row = dict(observations[index], train_label=label_map.get(key, -1))
            selected.append(row)
    # Adjacent same-frame fish remain contiguous, retaining hard-negative context.
    selected.sort(key=lambda r: (r['sequence'], r['frame_id'], r['track_id']))
    return {'schema_version': 1, 'source_split': 'train', 'crop_source': 'GT identity diagnosis, not detector online boxes',
            'samples_per_identity': samples_per_identity, 'sampling': 'uniform order statistics over each complete identity lifespan; no random adjacent-frame-only pairs',
            'protocol_sha256': sha256(protocol_path), 'annotation_sha256': sha256(root / 'annotations/train.json'),
            'train_identity_count': len(train_keys), 'identity_key': ['sequence', 'track_id'],
            'excluded_ambiguous_labels': excluded, 'records': selected}


def crop_to_tensor(image, xywh, size=(128, 256), margin=0.05):
    """Clip a fish box, retain aspect ratio, RGB ImageNet-normalized letterbox."""
    x, y, w, h = map(float, xywh)
    if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0:
        raise ValueError('Invalid crop box')
    x1, y1 = max(0, int(np.floor(x - margin * w))), max(0, int(np.floor(y - margin * h)))
    x2, y2 = min(image.shape[1], int(np.ceil(x + (1 + margin) * w))), min(image.shape[0], int(np.ceil(y + (1 + margin) * h)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Crop lies outside image')
    crop = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    out_h, out_w = size
    ratio = min(out_h / crop.shape[0], out_w / crop.shape[1])
    crop = cv2.resize(crop, (max(1, round(crop.shape[1] * ratio)), max(1, round(crop.shape[0] * ratio))), interpolation=cv2.INTER_LINEAR)
    # ImageNet mean-valued pad becomes zero after normalization for both modalities.
    mean = np.array([.485, .456, .406], dtype=np.float32)
    canvas = np.broadcast_to(mean, (out_h, out_w, 3)).copy()
    dy, dx = (out_h - crop.shape[0]) // 2, (out_w - crop.shape[1]) // 2
    canvas[dy:dy + crop.shape[0], dx:dx + crop.shape[1]] = crop.astype(np.float32) / 255
    canvas = (canvas - mean) / np.array([.229, .224, .225], dtype=np.float32)
    return torch.from_numpy(canvas.transpose(2, 0, 1).copy())


class PairedIdentityDataset(Dataset):
    def __init__(self, data_root, records, size=(128, 256)):
        self.root = Path(data_root) / 'train'
        self.records = records
        self.size = size
        self.cached_image_id = None
        self.cached_pair = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        if self.cached_image_id != row['image_id']:
            rgb, ir = [cv2.imread(str(self.root / row[k])) for k in ('rgb_file', 'ir_file')]
            if rgb is None or ir is None or rgb.shape != ir.shape:
                raise ValueError(f'Invalid paired image {row["image_id"]}')
            self.cached_pair = rgb, ir
            self.cached_image_id = row['image_id']
        return (*(crop_to_tensor(im, row['bbox_xywh'], self.size) for im in self.cached_pair), index)


class SingleIdentityDataset(Dataset):
    """Read only the requested modality; RGB training never opens legacy IR."""
    def __init__(self, data_root, records, modality='rgb', size=(128, 256)):
        if modality not in ('rgb', 'ir'):
            raise ValueError('Unknown modality')
        self.root = Path(data_root) / 'train'
        self.records, self.modality, self.size = records, modality, size
        self.cached_image_id, self.cached_image = None, None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        if self.cached_image_id != row['image_id']:
            self.cached_image = cv2.imread(str(self.root / row[self.modality + '_file']))
            if self.cached_image is None:
                raise ValueError(f'Cannot decode {self.modality} image {row["image_id"]}')
            self.cached_image_id = row['image_id']
        return crop_to_tensor(self.cached_image, row['bbox_xywh'], self.size), index
