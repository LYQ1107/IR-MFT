#!/usr/bin/env python3
"""Retrospective physical labels/age of observed anchors; no tracker decisions."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from cfp.provenance import sha256, snapshot_sources, write_json
from cfp.supervision import independent_proposal_labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queries', type=Path, required=True)
    p.add_argument('--detections', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Anchor diagnoses are immutable')
    protocol = json.loads(args.protocol.read_text())
    qm = json.loads((args.queries / 'manifest.json').read_text())
    dm = json.loads((args.detections / 'manifest.json').read_text())
    if qm['status'] != 'COMPLETE' or qm['partition'] != 'train' or qm['protocol_sha256'] != sha256(args.protocol) or set(qm['videos']) != set(protocol['train']):
        raise ValueError('Use completed training queries only, never test labels')
    if dm['status'] != 'COMPLETE' or dm['protocol_sha256'] != sha256(args.protocol):
        raise ValueError('Matching frozen detections required')
    if sha256(args.data_root / 'annotations/train.json') != protocol['train_annotation_sha256']:
        raise ValueError('Retrospective physical GT changed')
    spec = importlib.util.spec_from_file_location('cfp_anchor_gt_helper', Path(__file__).with_name('cfp_build_utility_queries.py'))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    gt = helper.gt_by_video(json.loads((args.data_root / 'annotations/train.json').read_text()), protocol)
    results = {}
    for sequence, entry in qm['videos'].items():
        qp, dp = args.queries / (sequence + '.npz'), args.detections / (sequence + '.npz')
        if sha256(qp) != entry['sha256'] or sha256(dp) != dm['caches'][sequence]['sha256']:
            raise ValueError('Observed anchor/proposal bytes changed')
        with np.load(qp) as a:
            query = {k: a[k] for k in a.files}
        with np.load(dp) as a:
            frames, offsets, boxes = [a[k] for k in ('frame_ids', 'offsets', 'boxes')]
        first = query['anchor_recent_history_indices'][:, 0]
        first_times = query['anchor_recent_history_frames'][:, 0]
        if np.any(first < 0) or np.any(first_times >= query['frame']):
            raise ValueError('Anchor must be an actually observed strictly earlier proposal')
        labels = np.full(len(boxes), -1, np.int64)
        for i, frame in enumerate(frames):
            ids, gt_boxes = zip(*gt[sequence][int(frame)]) if gt[sequence][int(frame)] else ([], [])
            lo, hi = offsets[i:i + 2]
            labels[lo:hi], _ = independent_proposal_labels(boxes[lo:hi], gt_boxes, ids,
                                                         minimum_iou=qm['config']['physical_matching_min_iou'],
                                                         minimum_gap=qm['config']['physical_matching_ambiguity_gap'])
        anchor_labels = labels[first]
        age = query['frame'] - first_times
        results[sequence] = dict(queries=len(first),
              anchor_known_fraction=float(np.mean(anchor_labels >= 0)),
              anchor_correct_fraction=float(np.mean(anchor_labels == query['anchor_id'])),
              anchor_wrong_known_fraction=float(np.mean((anchor_labels >= 0) & (anchor_labels != query['anchor_id']))),
              anchor_unknown_fraction=float(np.mean(anchor_labels < 0)),
              anchor_age_p50_p95_max_frames=[float(np.median(age)), float(np.quantile(age, .95)), int(age.max())],
              age_encoding_saturated_fraction=float(np.mean(age >= 74)),
              recent_history_mean_purity=float(np.mean(query['history_purity'])),
              recent_fully_contaminated_fraction=float(np.mean(query['history_purity'] == 0)),
              query_sha256=sha256(qp), detections_sha256=sha256(dp))
    report = dict(status='COMPLETE', seed=qm['encoder_seed'], query_manifest_sha256=sha256(args.queries / 'manifest.json'),
                  source_snapshots=snapshot_sources(args.output.with_suffix(''), [__file__, 'cfp/supervision.py', 'tools/cfp_build_utility_queries.py']),
                  videos=results,
                  scope='Training-only retrospective physical proposal labels, no test/dev diagnosis and no changes to online inference. Unknown/ambiguous labels are not proof of an incorrect identity. Timestamp age saturation is descriptive, not a proven cause of tracking errors.')
    write_json(args.output, report)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
