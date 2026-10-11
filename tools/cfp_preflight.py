#!/usr/bin/env python3
"""Read-only environment, paired-data and migrated-artifact audit."""
import argparse
import collections
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit_data(root):
    reports = {}
    for split in ('train', 'test'):
        annotation = root / 'annotations' / (split + '.json')
        data = json.loads(annotation.read_text())
        images = data['images']
        videos = {v['id']: v['file_name'] for v in data['videos']}
        per_video = collections.defaultdict(list)
        missing, empty, invalid_mapping = [], [], []
        for image in images:
            per_video[image['video_id']].append(image)
            for key in ('vi_file_name', 'ir_file_name'):
                path = root / split / image[key]
                if not path.is_file():
                    missing.append(str(path))
                elif path.stat().st_size == 0:
                    empty.append(str(path))
                if Path(image[key]).parts[0] != videos[image['video_id']]:
                    invalid_mapping.append(image['id'])
            if Path(image['vi_file_name']).stem != Path(image['ir_file_name']).stem:
                invalid_mapping.append(image['id'])
        sequence_reports = []
        for video_id, frames in sorted(per_video.items()):
            name = videos[video_id]
            frame_ids = [im['frame_id'] for im in frames]
            gt_path = root / split / name / 'gt' / 'gt.txt'
            gt_rows = list(csv.reader(gt_path.open())) if gt_path.exists() else []
            valid_gt = [row for row in gt_rows if len(row) >= 7 and float(row[6]) > 0]
            identities = {int(float(row[1])) for row in valid_gt}
            density = collections.Counter(int(float(row[0])) for row in valid_gt)
            sequence_reports.append({
                'sequence': name, 'video_id': video_id, 'frames': len(frames),
                'frame_min': min(frame_ids), 'frame_max': max(frame_ids),
                'unique_frame_ids': len(set(frame_ids)),
                'continuous_frames': sorted(frame_ids) == list(range(1, len(frames) + 1)),
                'annotation_order_causal': frame_ids == sorted(frame_ids),
                'image_dimensions': sorted({(im['height'], im['width']) for im in frames}),
                'gt_exists': gt_path.exists(), 'gt_sha256': sha256(gt_path) if gt_path.exists() else None,
                'seqinfo_exists': (gt_path.parent.parent / 'seqinfo.ini').is_file(),
                'gt_rows': len(gt_rows), 'valid_gt_rows': len(valid_gt), 'identities': len(identities),
                'identity_key': '(sequence, track_id)',
                'mean_fish_per_frame': len(valid_gt) / len(frames),
                'max_fish_per_frame': max(density.values(), default=0),
                'gt_frames_outside_images': sorted(set(density) - set(frame_ids)),
            })
        decoded = []
        try:
            import cv2
            for image in images[::max(1, len(images) // 50)]:
                pair = [cv2.imread(str(root / split / image[key])) for key in ('vi_file_name', 'ir_file_name')]
                valid = all(x is not None for x in pair)
                aligned = valid and pair[0].shape == pair[1].shape and pair[0].shape[:2] == (image['height'], image['width'])
                decoded.append({'image_id': image['id'], 'valid_and_aligned': bool(aligned)})
        except ImportError:
            pass
        reports[split] = {
            'annotation_sha256': sha256(annotation), 'pairs': len(images),
            'annotations': len(data['annotations']), 'missing': missing, 'empty': empty,
            'invalid_mapping': invalid_mapping, 'sample_decode': decoded,
            'sequences': sequence_reports,
        }
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('/data3/liuyeqiang/datasets/mft26-ir'))
    parser.add_argument('--assets', type=Path, default=Path('/data3/liuyeqiang/IR-MFT_assets'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = {'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
              'cpu_count': os.cpu_count(), 'storage': dict(zip(('total', 'used', 'free'), shutil.disk_usage(args.assets))),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    modules = {}
    for name in ('torch', 'torchvision', 'numpy', 'cv2', 'motmetrics', 'pycocotools', 'cython_bbox', 'lap', 'filterpy', 'pandas', 'scipy', 'loguru', 'thop'):
        try:
            module = importlib.import_module(name)
            modules[name] = {'version': getattr(module, '__version__', 'available'), 'path': getattr(module, '__file__', None)}
        except Exception as error:
            modules[name] = {'error': repr(error)}
    output['modules'] = modules
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True)
    output['gpu_snapshot'] = result.stdout.splitlines()
    output['free_memory'] = subprocess.check_output(['free', '-b'], text=True)
    output['data'] = audit_data(args.data_root)
    migrated = json.loads((args.assets / 'recovery_20261011' / 'nas_transfer_verified_20261011.json').read_text())
    output['checkpoints'] = []
    for item in migrated['files']:
        if item['destination'].endswith('.pth.tar'):
            path = Path(item['destination'])
            output['checkpoints'].append({'path': str(path), 'bytes': path.stat().st_size,
                'sha256': sha256(path), 'expected_sha256': item['sha256']})
    output['historical_metrics'] = json.loads((args.assets / 'history' / 'v3_eval_epoch3_tuned_b8' / 'metrics.json').read_text())
    cache_root = args.assets / 'history' / 'v3_dets_epoch3_b8_cache'
    cache = []
    import numpy as np
    for path in sorted(cache_root.glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            keys = list(data.keys())
            frame_ids, offsets, boxes, scores = (data[k] for k in ('frame_ids', 'offsets', 'boxes', 'scores'))
            cache.append({'sequence': path.stem, 'sha256': sha256(path), 'keys': keys,
                'shapes': {k: list(data[k].shape) for k in keys}, 'frames': len(frame_ids),
                'detections': len(scores), 'finite': bool(np.isfinite(boxes).all() and np.isfinite(scores).all()),
                'offsets_valid': len(offsets) == len(frame_ids) + 1 and int(offsets[-1]) == len(scores),
                'causal_order': bool(np.all(np.diff(frame_ids) > 0)),
                'provenance_status': 'historical unversioned; sidecar required before new cache reuse'})
    output['historical_detection_cache'] = cache
    failures = []
    for split, data in output['data'].items():
        if data['missing'] or data['empty'] or data['invalid_mapping']:
            failures.append(split + ': invalid paired files')
        if any(not item['valid_and_aligned'] for item in data['sample_decode']):
            failures.append(split + ': decode or registration mismatch')
    for item in output['checkpoints']:
        if item['sha256'] != item['expected_sha256']:
            failures.append('checkpoint hash mismatch: ' + item['path'])
    output['failures'] = failures
    output['ir_generation_provenance'] = 'unresolved: generator and original source not yet found; no claim of independent thermal sensing'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'failures': failures, 'train_pairs': output['data']['train']['pairs'], 'test_pairs': output['data']['test']['pairs']}, ensure_ascii=False))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
