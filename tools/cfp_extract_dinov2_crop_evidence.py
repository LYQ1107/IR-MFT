#!/usr/bin/env python3
"""Frozen pretrained CLS/dense/local observations on the exact two-crop pilot."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from cfp.dinov2_identity import load_pinned_small, resized_observations, spatial_observations
from cfp.local_identity_encoder import letterbox_visibility
from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'crop-cache', 'features', 'index', 'reference', 'weights', 'source-proof', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config['state'] != 'FROZEN' or args.output.exists() or args.output.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError('Registered frozen extraction and new outside-Git output required')
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if physical not in ('0', '1', '2', '3'):
        raise ValueError('Exactly one approved idle physical GPU0-3 required')
    cm = json.loads((args.crop_cache / 'manifest.json').read_text())
    fm = json.loads((args.features / 'manifest.json').read_text())
    im = json.loads((args.index / 'manifest.json').read_text())
    if (cm['status'] != 'COMPLETE_ALIGNED_PILOT_CROPS'
            or cm['feature_manifest_sha256'] != sha256(args.features / 'manifest.json')
            or cm['index_manifest_sha256'] != sha256(args.index / 'manifest.json')):
        raise ValueError('Exact completed pilot crops/features/index required')
    annotation_path = Path(im['data_root']) / 'annotations/train.json'
    if sha256(annotation_path) != im['annotation_sha256']:
        raise ValueError('Current-frame dimensions source changed')
    annotation = json.loads(annotation_path.read_text())
    videos = {row['id']: row['file_name'] for row in annotation['videos']}
    dimensions = {(videos[row['video_id']], int(row['frame_id'])): (row['height'], row['width'])
                  for row in annotation['images']}
    per_row_bytes = 162 * 384 * 2 + 2 * 384 * 4 + 12 * 384 * 2 + 174 * 4 + 1
    planned = sum(e['rows'] for e in cm['videos'].values()) * per_row_bytes
    if planned > config['max_feature_bytes'] or shutil.disk_usage(args.output.parent).free - planned < 40 * 2**30:
        raise OSError('Declared artifact budget or40GiB reserve')
    proofs = json.loads(args.source_proof.read_text())['imported_source_proofs']
    weights_manifest = json.loads((args.weights.parent / 'manifest.json').read_text())
    if weights_manifest['status'] != 'COMPLETE' or sha256(args.weights) != weights_manifest['file_sha256']:
        raise ValueError('Verified official pretrained payload required')
    # Validate all large crop inputs before allocating a GPU/output artifact.
    for video, e in cm['videos'].items():
        if sha256(args.crop_cache / (video + '.npy')) != e['sha256']:
            raise ValueError('Current crop bytes changed: ' + video)
    query = subprocess.check_output(['nvidia-smi', '-i', physical,
        '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    memory, utilization = [int(x.strip()) for x in query.split(',')]
    if memory > 64 or utilization:
        raise ValueError('Selected GPU is no longer idle; preserve other users jobs')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_pinned_small(args.reference, args.weights, proofs).cuda()
    args.output.mkdir(parents=True)
    report = dict(status='RUNNING', config=config, config_sha256=sha256(args.config),
        crop_manifest_sha256=sha256(args.crop_cache / 'manifest.json'),
        original_feature_manifest_sha256=sha256(args.features / 'manifest.json'),
        index_manifest_sha256=sha256(args.index / 'manifest.json'), weight_sha256=sha256(args.weights),
        reference_commit=weights_manifest['guard']['reference_commit'], imported_source_proofs=proofs,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        sources=snapshot_sources(args.output / 'sources', [__file__, 'cfp/dinov2_identity.py',
            'cfp/local_identity_encoder.py', args.config]), videos={}, physical_gpu=physical,
        precision='FP32; TF32 disabled', scope=config['scope'])
    write_json(args.output / 'manifest.json', report)
    for video, entry in cm['videos'].items():
        directory = args.output / video
        directory.mkdir()
        record_path = args.features / video / 'records.json'
        valid_path = args.features / video / 'valid.npy'
        index_path = args.index / (video + '.npz')
        if (sha256(record_path) != entry['records_sha256']
                or sha256(valid_path) != fm['videos'][video]['artifacts']['valid']['sha256']
                or sha256(index_path) != im['videos'][video]['index_sha256']):
            raise ValueError('Records/validity/geometry changed')
        shutil.copyfile(record_path, directory / 'records.json')
        shutil.copyfile(valid_path, directory / 'valid.npy')
        records = json.loads(record_path.read_text())
        valid = np.load(valid_path, allow_pickle=False)
        images = np.load(args.crop_cache / (video + '.npy'), mmap_mode='r')
        if images.shape != (len(records), 3, 128, 256) or len(records) != entry['rows']:
            raise ValueError('Input crop shape or indexing differs')
        with np.load(index_path, allow_pickle=False) as archive:
            boxes = {view: archive[key].copy() for view, key in [('gt', 'gt_boxes'), ('detector', 'detection_boxes')]}
        shapes = {'cls': (384,), 'patch_mean': (384,), 'patches': (162, 384),
                  'patch_visibility': (1, 9, 18), 'local4': (4, 384), 'local8': (8, 384),
                  'visibility4': (4,), 'visibility8': (8,)}
        outputs = {name: np.lib.format.open_memmap(directory / (name + '.npy'), mode='w+',
            dtype=np.float16 if name in ('patches', 'local4', 'local8') else np.float32,
            shape=(len(records), *shape)) for name, shape in shapes.items()}
        started = time.perf_counter()
        latencies = []
        with torch.inference_mode():
            for start in range(0, len(images), config['batch_size']):
                rows = records[start:start + config['batch_size']]
                masks = []
                for offset, row in enumerate(rows):
                    if not valid[start + offset]:
                        masks.append(torch.zeros(1, 128, 256))
                    else:
                        x1, y1, x2, y2 = boxes[row['view']][row['source_index']]
                        masks.append(letterbox_visibility(dimensions[(video, row['frame'])], [x1, y1, x2-x1, y2-y1]))
                crops = torch.from_numpy(np.array(images[start:start + len(rows)])).cuda().float()
                crops, visibility = resized_observations(crops, torch.stack(masks).cuda(), tuple(config['input_hw']))
                torch.cuda.synchronize()
                batch_started = time.perf_counter()
                observed = model.forward_features(crops)
                patches = observed['x_norm_patchtokens']
                values = spatial_observations(patches, visibility)
                values.update(cls=F.normalize(observed['x_norm_clstoken'].float(), dim=-1),
                              patches=patches, patch_visibility=visibility)
                torch.cuda.synchronize()
                latencies.append(time.perf_counter() - batch_started)
                available = torch.as_tensor(valid[start:start + len(rows)], device='cuda')
                for name, value in values.items():
                    value[~available] = 0
                    if not torch.isfinite(value).all():
                        raise ValueError('Nonfinite pretrained observation')
                    outputs[name][start:start + len(rows)] = value.cpu().numpy()
                if shutil.disk_usage(args.output).free < 40 * 2**30:
                    raise OSError('40GiB reserve')
        for mapped in outputs.values():
            mapped.flush()
        artifacts = {name: dict(shape=list(mapped.shape), dtype=str(mapped.dtype),
            sha256=sha256(directory / (name + '.npy'))) for name, mapped in outputs.items()}
        artifacts['valid'] = fm['videos'][video]['artifacts']['valid']
        report['videos'][video] = dict(observations=len(records), records_sha256=entry['records_sha256'],
            artifacts=artifacts, extraction_seconds=time.perf_counter()-started,
            encoding_batch_p50_p95_ms=(np.quantile(latencies, [.5, .95])*1000).tolist())
        write_json(args.output / 'manifest.json', report)
        print(video, len(records), 'COMPLETE', flush=True)
    report['status'] = 'COMPLETE_FROZEN_DINOV2_TWO_CROP_EVIDENCE'
    report['latency_scope'] = 'Cache-backed crop encoding; not full raw online detector/encoder/association FPS.'
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
