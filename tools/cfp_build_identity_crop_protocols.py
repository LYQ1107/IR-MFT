#!/usr/bin/env python3
"""Immutable two-crop identity index on all six training videos.

Only metadata/numeric box indices are saved, not duplicate images/crops. This
offline GT retrieval/training index is never consumed by online association.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from cfp.identity_evidence import physical_identity_key
from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('New immutable two-crop protocol index required')
    assets = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    dataset = Path('/data3/liuyeqiang/datasets/mft26-ir-causal-v1')
    protocol_path = Path('configs/cfp/protocol_v2.json')
    protocol = json.loads(protocol_path.read_text())
    annotation_path = dataset / 'annotations/train.json'
    if sha256(annotation_path) != protocol['train_annotation_sha256']:
        raise ValueError('GT source annotation drift')
    if shutil.disk_usage(assets).free < 40 * 2**30:
        raise OSError('40GiB reserve')
    annotation = json.loads(annotation_path.read_text())
    names = {r['id']: r['file_name'] for r in annotation['videos']}
    images = {r['id']: r for r in annotation['images']}
    gt = defaultdict(list)
    excluded = {(r['sequence'], r['frame_id'], r['track_id']) for r in protocol['ambiguous_identity_keys']}
    exclusion_counts = defaultdict(int)
    for row in annotation['annotations']:
        image = images[row['image_id']]
        video = names[image['video_id']]
        if video not in protocol['train']:
            continue
        frame, identity = int(image['frame_id']), int(row['track_id'])
        if (video, frame, identity) in excluded or row.get('conf', 1) <= 0 or row.get('iscrowd', 0):
            exclusion_counts[video] += 1
            continue
        x, y, w, h = map(float, row['bbox'])
        if not np.isfinite([x, y, w, h]).all() or min(w, h) <= 0:
            exclusion_counts[video] += 1
            continue
        physical_identity_key(video, identity)
        gt[video].append((frame, identity, int(image['id']), [x, y, x + w, y + h]))
    physical_root = assets / 'phase5/recovery_train_queries_v1'
    physical_meta = json.loads((physical_root / 'manifest.json').read_text())
    detection_root = assets / 'phase0/causal_v3_train_detections'
    dm = json.loads((detection_root / 'manifest.json').read_text())
    if physical_meta['status'] != 'COMPLETE' or dm['status'] != 'COMPLETE':
        raise ValueError('Complete immutable proposal/physical-label banks required')
    args.output.mkdir(parents=True)
    report = dict(status='RUNNING', videos={}, data_root=str(dataset), source_split='train',
        identity_key=['video_id', 'physical_track_id'], protocol_sha256=sha256(protocol_path),
        annotation_sha256=sha256(annotation_path), detection_manifest_sha256=sha256(detection_root / 'manifest.json'),
        physical_manifest_sha256=sha256(physical_root / 'manifest.json'),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        protocols={'GT_crop': 'All valid nonambiguous physical GT boxes on all six train videos; ideal identity diagnosis only, not online detector input.',
                   'detector_crop': 'Every frozen actual V3 proposal kept in original order, independent physical match label or -1 unknown. Unknown proposals are not labeled wrong.'},
        paired_comparison='Join explicit (video,frame,known physical fish) across GT/detector views; report coverage and multiple proposals, never silently select a GT-favored proposal.',
        temporal_protocol='Query/gallery temporal gaps and same-video/same-frame hard negatives must be declared by evaluators; actual GT turning/occlusion annotations are unavailable, gap/overlap/shape are proxies only.',
        training_protocol='New fish encoder/adapter fitting must exclude both outer-video and distinct inner-calibration-video; frozen old FishReID saw all6 and is an exposed reference. This metadata index does not train a model.',
        pending='Feature extraction, retrieval/same-frame pair metrics, fold-specific training and full-online integration are not complete.',
        source_snapshots=snapshot_sources(args.output / 'sources', [__file__, 'cfp/identity_evidence.py', protocol_path]))
    write_json(args.output / 'manifest.json', report)
    for video in protocol['train']:
        dp, pp = detection_root / (video + '.npz'), physical_root / (video + '.npz')
        if sha256(dp) != dm['caches'][video]['sha256'] or sha256(pp) != physical_meta['videos'][video]['sha256']:
            raise ValueError('Frozen detection/physical supervision drift')
        with np.load(dp, allow_pickle=False) as a:
            frames, offsets, boxes, scores = [a[k].copy() for k in ('frame_ids', 'offsets', 'boxes', 'scores')]
        with np.load(pp, allow_pickle=False) as a:
            labels, det_frames = a['physical_labels'].copy(), a['frame_of_proposal'].copy()
        if len(labels) != len(boxes) or not np.array_equal(det_frames, np.repeat(frames, np.diff(offsets))):
            raise ValueError('Proposal identity/timestamp order drift')
        view_images = sorted([r for r in images.values() if names[r['video_id']] == video], key=lambda r: r['frame_id'])
        if not np.array_equal(np.asarray([r['frame_id'] for r in view_images]), frames):
            raise ValueError('All-frame image/proposal coverage differs')
        for r in view_images:
            for key in ('vi_file_name', 'ir_file_name'):
                path = Path(r[key])
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError('Frame path must stay inside declared dataset split')
        frame_paths = [dict(frame_id=int(r['frame_id']), image_id=int(r['id']), rgb_file=r['vi_file_name'], ir_file=r['ir_file_name']) for r in view_images]
        paths_file = args.output / (video + '.frames.json')
        write_json(paths_file, frame_paths)
        observations = sorted(gt[video], key=lambda r: (r[0], r[1]))
        keys = [(frame, identity) for frame, identity, _, _ in observations]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate GT physical key survived annotation policy')
        gt_frames = np.asarray([r[0] for r in observations], np.int64)
        gt_identities = np.asarray([r[1] for r in observations], np.int64)
        gt_boxes = np.asarray([r[3] for r in observations], np.float32)
        known_pairs = {(int(frame), int(identity)) for frame, identity in zip(det_frames, labels) if identity >= 0}
        truth_pairs = set(keys)
        if not known_pairs <= truth_pairs:
            raise ValueError('Known detector identity lacks matching same-frame GT key')
        output = args.output / (video + '.npz')
        np.savez_compressed(output, gt_frame=gt_frames, gt_identity=gt_identities, gt_boxes=gt_boxes,
            gt_image_id=np.asarray([r[2] for r in observations], np.int64),
            detection_frame=det_frames, detection_identity=labels, detection_boxes=boxes, detection_confidence=scores,
            detection_offsets=offsets, frame_ids=frames, proposal_index=np.arange(len(boxes), dtype=np.int64))
        report['videos'][video] = dict(frames=len(frames), gt_observations=len(observations),
            physical_identities=len(set(gt_identities.tolist())), proposals=len(boxes), unknown_proposals=int((labels < 0).sum()),
            paired_known_frame_identity_keys=len(known_pairs), gt_keys_not_covered=len(truth_pairs - known_pairs),
            extra_known_duplicate_proposals=int((labels >= 0).sum()) - len(known_pairs),
            excluded_gt_rows=exclusion_counts[video], index_sha256=sha256(output), frame_paths_sha256=sha256(paths_file),
            detection_sha256=sha256(dp), physical_sha256=sha256(pp))
        write_json(args.output / 'manifest.json', report)
        print(video, report['videos'][video], flush=True)
    report['status'] = 'COMPLETE_INDEX_ONLY'
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
