#!/usr/bin/env python3
"""Stage frozen historical-test inputs without copying the image or detection data.

The canonical causal dataset contains the development/training views, while the
historical-test images and the frozen V3 proposal cache live in the phase-zero
reproduction assets.  This adapter creates only symlinks plus provenance JSON
so the existing feature and external-association runners can consume the
historical split under protocol_v2 without silently mixing protocol versions.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cfp.provenance import sha256, write_json


def link(target: Path, source: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        raise FileExistsError(target)
    target.symlink_to(source, target_is_directory=source.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True,
                        help='Original mft26-ir root containing test/ and annotations/test.json')
    parser.add_argument('--detections', type=Path, required=True,
                        help='Frozen phase-zero V3 detection cache directory')
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    sequences = protocol['historical_test']
    annotation = args.dataset / 'annotations' / 'test.json'
    if sha256(annotation) != protocol['test_annotation_sha256']:
        raise ValueError('Historical annotation does not match protocol_v2')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Output is immutable and non-empty: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)

    data_root = args.output / 'data'
    link(data_root / 'test', args.dataset / 'test')
    link(data_root / 'annotations', args.dataset / 'annotations')
    generation = {
        'schema_version': 1,
        'status': 'COMPLETE',
        'input_regime': 'causal_historical_replay',
        'probe_frames': 0,
        'protocol_sha256': sha256(args.protocol),
        'annotation_sha256': sha256(annotation),
        'source_dataset_root': str(args.dataset),
        'source_test_annotation_sha256': sha256(annotation),
        'scope': 'Read-only symlink staging; original historical RGB/IR frames are unchanged',
    }
    write_json(data_root / 'manifest.json', generation)

    detection_root = args.output / 'detections'
    detection_root.mkdir(parents=True, exist_ok=True)
    caches = {}
    for sequence in sequences:
        source = args.detections / f'{sequence}.npz'
        if not source.is_file():
            raise FileNotFoundError(source)
        link(detection_root / source.name, source)
        provenance = args.detections / f'{sequence}.provenance.json'
        if provenance.is_file():
            link(detection_root / provenance.name, provenance)
        cache = np.load(source)
        frame_ids, offsets, boxes, scores = [cache[k] for k in ('frame_ids', 'offsets', 'boxes', 'scores')]
        if not np.array_equal(frame_ids, np.arange(1, len(frame_ids) + 1)):
            raise ValueError(f'{sequence}: non-contiguous frame ids')
        if len(offsets) != len(frame_ids) + 1 or offsets[0] != 0 or offsets[-1] != len(boxes):
            raise ValueError(f'{sequence}: invalid detection offsets')
        if len(boxes) != len(scores) or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError(f'{sequence}: invalid detection values')
        caches[sequence] = {
            'frames': int(len(frame_ids)),
            'detections': int(len(boxes)),
            'sha256': sha256(source),
        }
    manifest = {
        'schema_version': 1,
        'status': 'COMPLETE',
        'config': {
            'input_regime': 'causal',
            'source_split': 'test',
            'partition': 'historical_test',
            'probe_frames': 0,
            'source_cache_root': str(args.detections),
            'source_cache_scope': 'Frozen V3 reproduction proposals; no detector rerun',
            'generation_manifest_sha256': sha256(data_root / 'manifest.json'),
        },
        'protocol_sha256': sha256(args.protocol),
        'annotation_sha256': sha256(annotation),
        'generation_manifest_sha256': sha256(data_root / 'manifest.json'),
        'protocol_id': protocol.get('protocol_id'),
        'probe': False,
        'caches': caches,
        'scope': 'Historical-test adapter only; all public methods share these unchanged proposals',
    }
    write_json(detection_root / 'manifest.json', manifest)
    write_json(args.output / 'manifest.json', {
        'schema_version': 1,
        'status': 'COMPLETE',
        'protocol_sha256': sha256(args.protocol),
        'annotation_sha256': sha256(annotation),
        'data_root': str(data_root),
        'detections_root': str(detection_root),
        'source_dataset_root': str(args.dataset),
        'source_detection_root': str(args.detections),
        'sequences': sequences,
        'storage': 'symlinks plus small manifests; no image/detection bytes copied',
        'caches': caches,
    })
    print(json.dumps({'output': str(args.output), 'sequences': sequences, 'caches': caches}, indent=2))


if __name__ == '__main__':
    main()
