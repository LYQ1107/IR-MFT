#!/usr/bin/env python3
"""Build fixed-candidate OOF utility targets from actual causal tracker histories."""
import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from cfp.association import AppearanceTracker
from cfp.candidate_context import complete_candidate_context
from cfp.provenance import sha256, write_json, snapshot_sources
from cfp.supervision import independent_proposal_labels
from cfp.utility import decision_evidence, counterfactual_utility, EVIDENCE_FIELDS
from yolox.tracker import matching
from yolox.tracker.basetrack import BaseTrack


class QueryRecorder(AppearanceTracker):
    def __init__(self, args, labels, config, frame_of_proposal):
        super().__init__(args, mode='rgb', appearance_weight=.1, ema=.9,
                         proximity_iou=.1, appearance_max_distance=.35)
        self.labels = labels
        self.config = config
        self.frame_of_proposal = frame_of_proposal
        self.anchor = {}
        self.clean = defaultdict(lambda: deque(maxlen=8))
        self.current_ambiguous = set()
        self.rows = []
        self.exclusions = defaultdict(int)

    def association_cost(self, tracks, detections):
        result = super().association_cost(tracks, detections)
        if not tracks or not detections:
            return result
        raw_geometry = matching.iou_distance(tracks, detections)
        for i, track in enumerate(tracks):
            if track.memory_rgb is None:
                continue
            history = list(track.identity_proposals)
            if any(t >= self.frame_id or index is None or self.frame_of_proposal[index] != t for t, index in history):
                raise ValueError('Current/future or invalid source proposal in accepted memory')
            if track.track_id not in self.anchor:
                known = [int(self.labels[index]) for _, index in history if self.labels[index] >= 0]
                if known:
                    self.anchor[track.track_id] = known[0]
            if (self.frame_id - 1) % self.config['query_frame_stride']:
                continue
            if track.track_id not in self.anchor:
                self.exclusions['unlabelled_past_anchor'] += 1
                continue
            anchor = self.anchor[track.track_id]
            if anchor in self.current_ambiguous:
                self.exclusions['ambiguous_current_identity'] += 1
                continue
            rgb = np.array([float(np.clip((1 - track.memory_rgb @ d.current_rgb) / 2, 0, 1))
                            if d.current_rgb is not None else 1. for d in detections], np.float64)
            eligible = np.array([d.current_rgb is not None for d in detections]) & (raw_geometry[i] <= .9) & (rgb <= .35)
            chosen = np.flatnonzero(eligible)
            if not len(chosen):
                self.exclusions['null_only_zero_utility'] += 1
                continue
            costs = rgb[chosen]
            ir, available = costs.copy(), np.zeros(len(chosen), bool)
            indices = np.array([detections[j].current_proposal_index for j in chosen], np.int64)
            if np.any(self.frame_of_proposal[indices] != self.frame_id):
                raise ValueError('Candidate proposal is not from current frame')
            for k, j in enumerate(chosen):
                d = detections[j]
                if track.memory_ir is not None and d.current_ir is not None:
                    ir[k] = float(np.clip((1 - track.memory_ir @ d.current_ir) / 2, 0, 1))
                    available[k] = True
            null_distance = self.config['null_distance']
            if self.config.get('candidate_context_version', 1) == 2:
                geometry = matching.fuse_score(raw_geometry, detections)
                context = complete_candidate_context(track, detections, raw_geometry[i], geometry[i], self.frame_id,
                                  teacher_weight=self.config['teacher_association_weight'])
                if context is None:
                    self.exclusions['no_modifiable_ir'] += 1
                    continue
                chosen = context['pool']
                indices = np.array([detections[j].current_proposal_index for j in chosen], np.int64)
                costs, ir = context['rgb'], context['ir']
                null_distance = context['null_distance']
            correct = np.flatnonzero(self.labels[indices] == anchor)
            if len(correct) > 1:
                raise ValueError('Nonunique physical candidate label')
            target = int(correct[0]) if len(correct) else len(chosen)
            evidence = context['evidence'] if self.config.get('candidate_context_version', 1) == 2 else decision_evidence(costs, ir, raw_geometry[i, chosen],
                       np.array([detections[j].score for j in chosen]), available,
                       self.frame_id - track.start_frame + 1, self.frame_id - track.last_identity_frame,
                       [row[1] for row in track.identity_history],
                       self.config['teacher_temperature'], self.config['null_distance'])
            utility = counterfactual_utility(costs, ir, target, self.config['teacher_temperature'],
                      self.config['teacher_ir_weight'], null_distance)
            if self.config.get('candidate_context_version', 1) == 1 and target == len(chosen):
                all_indices = np.array([d.current_proposal_index for d in detections])
                threshold = self.args.match_thresh if track.is_activated else .7
                if np.any((self.labels[all_indices] == anchor) & (result[i] <= threshold)):
                    self.exclusions['local_null_despite_rgb_matchable_true_candidate'] += 1
            clean = list(self.clean[anchor])
            if any(t >= self.frame_id for t, _ in clean):
                raise ValueError('Teacher-forced current observation leaked into past history')
            self.rows.append(dict(frame=self.frame_id, track_id=track.track_id, anchor_id=anchor,
                             target=target, rgb=costs, ir=ir, candidate_indices=indices, evidence=evidence,
                             utility=utility, null_distance=null_distance, history=history, clean_history=clean))
        return result


def gt_by_video(annotation, protocol):
    names = {v['id']: v['file_name'] for v in annotation['videos']}
    images = {im['id']: im for im in annotation['images']}
    excluded = {(r['sequence'], r['frame_id'], r['track_id']) for r in protocol['ambiguous_identity_keys']}
    result = defaultdict(lambda: defaultdict(list))
    for ann in annotation['annotations']:
        im = images[ann['image_id']];seq = names[im['video_id']];frame = int(im['frame_id']);identity = int(ann['track_id'])
        if (seq, frame, identity) in excluded or ann.get('conf', 1) <= 0 or ann.get('iscrowd', 0):
            continue
        x, y, w, h = ann['bbox']
        if w > 0 and h > 0:
            result[seq][frame].append((identity, [x, y, x + w, y + h]))
    return result


def store_queries(path, rows):
    offsets = np.concatenate([[0], np.cumsum([len(r['rgb']) for r in rows])])
    def histories(key, component):
        out = np.full((len(rows), 8), -1, np.int64)
        for i, r in enumerate(rows):
            values = r[key]
            if values:
                out[i, -len(values):] = [v[component] for v in values]
        return out
    values = dict(offsets=offsets.astype(np.int64),
                  evidence=np.stack([r['evidence'] for r in rows]).astype(np.float32),
                  utility=np.array([r['utility'] for r in rows], np.float32),
                  history_indices=histories('history', 1), history_frames=histories('history', 0),
                  clean_history_indices=histories('clean_history', 1), clean_history_frames=histories('clean_history', 0))
    values['null_distance'] = np.array([r['null_distance'] for r in rows], np.float32)
    for key in ('frame', 'track_id', 'anchor_id', 'target'):
        values[key] = np.array([r[key] for r in rows], np.int64)
    for key in ('rgb', 'ir', 'candidate_indices'):
        values[key] = np.concatenate([r[key] for r in rows])
    np.savez_compressed(path, **values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--detections', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--partition', choices=['train', 'dev'], required=True)
    p.add_argument('--feature-roots', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Supervised utility queries are immutable')
    protocol = json.loads(args.protocol.read_text());config = json.loads(args.config.read_text())
    if protocol['state'] != 'FROZEN' or config['state'] != 'FROZEN':
        raise ValueError('Frozen data and utility protocols required')
    if sorted(config['fit_videos'] + config['calibration_videos']) != sorted(protocol['train']):
        raise ValueError('Gate fit/calibration must partition association training videos')
    if sha256(args.data_root / 'annotations/train.json') != protocol['train_annotation_sha256']:
        raise ValueError('Physical supervision annotation changed')
    detections = json.loads((args.detections / 'manifest.json').read_text())
    if detections['status'] != 'COMPLETE' or detections['probe'] or detections['protocol_sha256'] != sha256(args.protocol):
        raise ValueError('Incomplete/mismatched canonical proposals')
    features = {}
    sources = []
    encoder_seeds = set()
    for root in args.feature_roots:
        meta = json.loads((root / 'manifest.json').read_text())
        if meta['status'] != 'COMPLETE' or meta['protocol_sha256'] != sha256(args.protocol) or meta['embedding_stage'] != 'post_neck' or meta.get('crop_variant', 'original') != 'original':
            raise ValueError('This trial requires complete original-crop post-neck banks')
        if meta['detection_manifest_sha256'] != sha256(args.detections / 'manifest.json'):
            raise ValueError('Features use different proposals')
        encoder_seeds.update([meta['rgb_encoder']['seed'], meta['ir_encoder']['seed']])
        if args.partition == 'train' and meta['encoder_role'] != 'oof_teacher':
            raise ValueError('Utility training rejects in-sample teacher banks')
        if args.partition == 'dev' and meta.get('encoder_role', 'main_baseline') != 'main_baseline':
            raise ValueError('Dev query diagnosis requires heldout main encoders')
        for seq in meta.get('selected_sequences', list(meta['videos'])):
            if seq in features or seq not in protocol[args.partition]:
                raise ValueError('Duplicate/outside-split teacher coverage')
            for branch in ('rgb_encoder', 'ir_encoder'):
                training_meta = meta[branch]['training_metadata']
                if args.partition == 'dev' and training_meta['protocol_sha256'] != sha256(args.protocol):
                    raise ValueError('Main encoder has a different training protocol')
                training = training_meta.get('training_videos', protocol['train'])
                if seq in training or set(training) & set(protocol['dev']):
                    raise ValueError('Teacher has query/dev training exposure')
            features[seq] = root
        sources.append(dict(root=str(root), manifest_sha256=sha256(root / 'manifest.json')))
    if set(features) != set(protocol[args.partition]):
        raise ValueError('Feature banks must cover exactly the complete selected partition')
    if len(encoder_seeds) != 1:
        raise ValueError('One end-to-end seed must use matched RGB/IR teacher seeds across folds')
    annotation = json.loads((args.data_root / 'annotations/train.json').read_text())
    gt = gt_by_video(annotation, protocol)
    policy_ambiguous = defaultdict(set)
    for row in protocol['ambiguous_identity_keys']:
        policy_ambiguous[(row['sequence'], row['frame_id'])].add(row['track_id'])
    args.output.mkdir(parents=True)
    source_snapshots = snapshot_sources(args.output, [__file__, 'cfp/association.py', 'cfp/utility.py',
                                   'cfp/supervision.py', 'cfp/candidate_context.py', 'cfp/provenance.py', args.protocol, args.config])
    manifest = dict(status='RUNNING', partition=args.partition, config=config, config_sha256=sha256(args.config),
                    protocol_sha256=sha256(args.protocol), teacher_sources=sources, evidence_fields=list(EVIDENCE_FIELDS),
                    encoder_seed=next(iter(encoder_seeds)),
                    source_snapshots=source_snapshots,
                    source_sha256=sha256(__file__), tracker_source_sha256=sha256('cfp/association.py'),
                    utility_source_sha256=sha256('cfp/utility.py'), label_source_sha256=sha256('cfp/supervision.py'),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), videos={},
                    scope='GT labels are retrospective supervision only; candidate geometry, scoring and accepted writes use no GT. Query targets use OOF encoders for train and heldout main encoders for dev diagnosis.')
    write_json(args.output / 'manifest.json', manifest)
    for sequence in protocol[args.partition]:
        det_path = args.detections / (sequence + '.npz');feature_path = features[sequence] / (sequence + '.npz')
        provenance = json.loads(feature_path.with_suffix('.provenance.json').read_text())
        if sha256(det_path) != detections['caches'][sequence]['sha256'] or sha256(det_path) != provenance['detection_sha256'] or sha256(feature_path) != provenance['features_sha256']:
            raise ValueError('Frozen teacher/proposal content changed')
        cache = np.load(det_path)
        with np.load(feature_path) as archive:
            bank = {key: archive[key] for key in archive.files}
        frames, offsets, boxes, scores = [cache[k] for k in ['frame_ids','offsets','boxes','scores']]
        if not np.array_equal(frames, np.arange(1, len(frames) + 1)) or len(bank['rgb']) != len(boxes) or bank['ir'].shape != bank['rgb'].shape:
            raise ValueError('Full ordered proposal coverage required')
        labels = np.full(len(boxes), -1, np.int64);ambiguous = {}
        frame_of_proposal = np.repeat(frames, np.diff(offsets))
        for i, frame in enumerate(frames):
            identities, gt_boxes = zip(*gt[sequence][int(frame)]) if gt[sequence][int(frame)] else ([], [])
            lo, hi = offsets[i:i+2]
            labels[lo:hi], ambiguous[int(frame)] = independent_proposal_labels(boxes[lo:hi], gt_boxes, identities,
                   config['physical_proposal_labels']['minimum_iou'], config['physical_proposal_labels']['minimum_top_two_iou_gap'])
            ambiguous[int(frame)].update(policy_ambiguous[(sequence, int(frame))])
        BaseTrack._count = 0
        tracker = QueryRecorder(SimpleNamespace(track_thresh=.58,track_buffer=45,match_thresh=.97,mot20=False), labels, config, frame_of_proposal)
        for i, frame in enumerate(frames):
            lo, hi = offsets[i:i+2];tracker.current_ambiguous = ambiguous[int(frame)]
            tracker.update(np.concatenate([boxes[lo:hi].copy(),scores[lo:hi,None]],1),(1,1),(1,1),bank['rgb'][lo:hi],bank['ir'][lo:hi],np.arange(lo,hi))
            for index in range(lo, hi):
                if labels[index] >= 0 and bank['valid_rgb'][index]:
                    tracker.clean[int(labels[index])].append((int(frame),index))
        if not tracker.rows:
            raise ValueError('No eligible fixed-candidate supervised queries')
        output = args.output / (sequence + '.npz');store_queries(output, tracker.rows)
        u = np.array([r['utility'] for r in tracker.rows]);null = sum(r['target']==len(r['rgb']) for r in tracker.rows)
        report = dict(queries=len(u), null_targets=null, utility_mean=float(u.mean()),utility_std=float(u.std()),
                      positive_fraction=float(np.mean(u>0)),negative_fraction=float(np.mean(u<0)),zero_fraction=float(np.mean(u==0)),
                      candidates_mean=float(np.mean([len(r['rgb']) for r in tracker.rows])), labelled_proposals=int(np.sum(labels>=0)),
                      exclusions=dict(tracker.exclusions),sha256=sha256(output), detections_sha256=sha256(det_path),features_sha256=sha256(feature_path))
        manifest['videos'][sequence]=report;write_json(args.output/'manifest.json',manifest)
        print(sequence,json.dumps(report),flush=True)
    manifest['status']='COMPLETE';write_json(args.output/'manifest.json',manifest)


if __name__ == '__main__':
    main()
