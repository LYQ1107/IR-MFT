"""Immutable current crops/physical labels; heldout banks never fit a model."""
import json
from pathlib import Path

import numpy as np
import torch

from .local_identity_encoder import letterbox_visibility
from .provenance import sha256


def load_crop_banks(crop_root, feature_root, index_root):
    crop_root, feature_root, index_root = map(Path, (crop_root, feature_root, index_root))
    cm = json.loads((crop_root/'manifest.json').read_text())
    fm = json.loads((feature_root/'manifest.json').read_text())
    im = json.loads((index_root/'manifest.json').read_text())
    if (cm['status'] != 'COMPLETE_ALIGNED_PILOT_CROPS'
            or fm['status'] != 'COMPLETE_FROZEN_DINOV2_TWO_CROP_EVIDENCE'
            or fm['crop_manifest_sha256'] != sha256(crop_root/'manifest.json')
            or fm['index_manifest_sha256'] != sha256(index_root/'manifest.json')
            or cm['feature_manifest_sha256'] != fm['original_feature_manifest_sha256']):
        raise ValueError('Exact aligned completed current-crop banks required')
    annotation_path = Path(im['data_root'])/'annotations/train.json'
    if sha256(annotation_path) != im['annotation_sha256']:
        raise ValueError('Image geometry provenance changed')
    annotation = json.loads(annotation_path.read_text())
    videos = {r['id']: r['file_name'] for r in annotation['videos']}
    dimensions = {(videos[r['video_id']], int(r['frame_id'])): (r['height'], r['width']) for r in annotation['images']}
    banks = {}
    for video, e in fm['videos'].items():
        directory = feature_root/video
        paths = [directory/'records.json', directory/'valid.npy', crop_root/(video+'.npy'), index_root/(video+'.npz')]
        expected = [e['records_sha256'], e['artifacts']['valid']['sha256'], cm['videos'][video]['sha256'], im['videos'][video]['index_sha256']]
        if any(sha256(p) != value for p, value in zip(paths, expected)):
            raise ValueError('Crop/record/geometry/validity drift: '+video)
        with np.load(paths[3], allow_pickle=False) as a:
            boxes = {view: a[key].copy() for view, key in [('gt', 'gt_boxes'), ('detector', 'detection_boxes')]}
        records = json.loads(paths[0].read_text())
        crops = np.load(paths[2], mmap_mode='r')
        valid = np.load(paths[1], allow_pickle=False)
        if crops.shape != (len(records), 3, 128, 256) or valid.shape != (len(records),):
            raise ValueError('Aligned current crop dimensions required')
        banks[video] = dict(records=records, crops=crops, arrays={'valid': valid}, boxes=boxes)
    return banks, dimensions, fm


def pixel_batch(observations, banks, dimensions, device='cuda'):
    crops, masks = [], []
    for video, i in observations:
        bank = banks[video]
        row = bank['records'][i]
        crops.append(bank['crops'][i])
        if bank['arrays']['valid'][i]:
            x1, y1, x2, y2 = bank['boxes'][row['view']][row['source_index']]
            masks.append(letterbox_visibility(dimensions[(video, row['frame'])], [x1, y1, x2-x1, y2-y1]))
        else:
            masks.append(torch.zeros(1, 128, 256))
    return torch.from_numpy(np.array(crops)).to(device).float(), torch.stack(masks).to(device)


def augment_observations(images, masks, rng, config):
    """Both geometry and observed augmentation support transform together."""
    images, masks = images.clone(), masks.clone()
    flip = rng.random(len(images)) < config['horizontal_flip_probability']
    images[flip] = images[flip].flip(-1)
    masks[flip] = masks[flip].flip(-1)
    h, w = images.shape[-2:]
    for i in range(len(images)):
        if rng.random() < config['erase_probability']:
            area = rng.uniform(*config['erase_area_fraction'])*h*w
            aspect = rng.uniform(.5, 2.)
            eh = min(h, max(1, round((area/aspect)**.5)))
            ew = min(w, max(1, round((area*aspect)**.5)))
            y, x = int(rng.integers(h-eh+1)), int(rng.integers(w-ew+1))
            images[i, :, y:y+eh, x:x+ew] = 0
            masks[i, :, y:y+eh, x:x+ew] = 0
    return images, masks
