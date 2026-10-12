#!/usr/bin/env python3
"""Join fixed online traces to retrospective physical/official event evidence."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from cfp.provenance import sha256, snapshot_sources, write_json
from cfp.recovery_supervision import physical_bank, classify_row


def overlaps(boxes, truth):
    boxes, truth = np.asarray(boxes, float).reshape(-1, 4), np.asarray(truth, float).reshape(-1, 4)
    if not len(boxes) or not len(truth):
        return np.zeros((len(boxes), len(truth)))
    extent = np.maximum(np.minimum(boxes[:, None, 2:], truth[None, :, 2:]) - np.maximum(boxes[:, None, :2], truth[None, :, :2]), 0)
    area = extent.prod(-1)
    return area / np.maximum((boxes[:, 2:] - boxes[:, :2]).prod(-1)[:, None] + (truth[:, 2:] - truth[:, :2]).prod(-1)[None, :] - area, 1e-8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Offline event report immutable')
    run = json.loads((args.run / 'manifest.json').read_text())
    if run['status'] != 'COMPLETE' or run['probe'] or run['partition'] not in ('train', 'dev'):
        raise ValueError('Full frozen train/dev decision trace required')
    args.output.mkdir(parents=True)
    record = dict(status='RUNNING', partition=run['partition'], run_manifest_sha256=sha256(args.run / 'manifest.json'),
                  source_snapshots=snapshot_sources(args.output, [__file__, 'cfp/recovery_supervision.py', 'cfp/supervision.py']),
                  videos={}, scope='Retrospective operational/overlapping taxonomy, modal known past physical identity reference; no immutable first-anchor truth. Heuristics are proxies, not exclusive proven causes or additive causal HOTA contributions. Unknowns excluded. Official continuity matching labels remain analysis-only.')
    root = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    feature_root = root / ('phase1' if run['partition'] == 'dev' else 'phase2') / f'causal_{run["partition"]}_detection_features_postneck_seed{run["seed"]}'
    evaluation = args.run.with_name(args.run.name + '_evaluation')
    official = None
    if (evaluation / 'metrics.json').exists():
        command = [sys.executable, 'tools/cfp_trace_clear_events.py', '--evaluation-root', str(evaluation), '--derived-gt-root', str(root / 'protocol_v2/gt'),
                   '--trackeval-root', '/data3/liuyeqiang/CFP-FishMOT_references/TrackEval', '--output', str(args.output / 'official_clear.json')]
        with (args.output / 'official_clear.log').open('w') as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
        official = json.loads((args.output / 'official_clear.json').read_text())
        record['official_command'] = command
    write_json(args.output / 'manifest.json', record)
    for seq in run['videos']:
        bank = physical_bank(seq, run['partition'])
        fp = feature_root / (seq + '.npz')
        if sha256(fp) != json.loads(fp.with_suffix('.provenance.json').read_text())['features_sha256']:
            raise ValueError('Feature bank drift')
        with np.load(fp, allow_pickle=False) as archive:
            features = archive['rgb']
        trace = args.run / (seq + '.decisions.jsonl.gz')
        if sha256(trace) != run['videos'][seq]['decisions_sha256']:
            raise ValueError('Decision trace drift')
        events, affected_frames, tracks_by_category = Counter(), defaultdict(set), defaultdict(set)
        for category in ('appearance_similar_swap_proxy', 'crossing_overlap_wrong_match_proxy', 'occlusion_reappearance_new_id',
                         'wrong_reactivation', 'duplicate_output_overlap_proxy', 'premature_termination_proxy',
                         'false_initialization_no_gt_overlap_proxy', 'high_confidence_wrong_match', 'long_term_memory_pollution',
                         'true_candidate_geometry_excluded', 'identity_inconsistent_match'):
            events[category] = 0
        unknown, queries, candidate_recall, known = 0, 0, 0, 0
        visible, full_recall = 0, 0
        seen = set()
        cases, focus = [], []
        evidence = args.output / (seq + '.events.jsonl.gz')
        with gzip.open(trace, 'rt') as source, gzip.open(evidence, 'wt', compresslevel=3) as dest:
            for line in source:
                row = json.loads(line)
                if seq == 'BT-005' and 1044 <= row['frame'] <= 1054:
                    focus.append(row)
                for i, track in enumerate(row['tracks']):
                    assessment = classify_row(row, i, bank['labels'], run['association'], features)
                    queries += 1
                    if assessment['target'] is None:
                        unknown += 1
                        continue
                    known += 1
                    candidate_recall += assessment['true_candidate_recall']
                    frame = row['frame']
                    if assessment['target'] in {identity for identity, _ in bank['gt'][frame]}:
                        visible += 1
                        lo, hi = bank['offsets'][frame - 1:frame + 1]
                        full_recall += bool(np.any(bank['labels'][lo:hi] == assessment['target']))
                    categories = list(assessment['categories'])
                    if row['stage'] == 'unconfirmed' and i in row['unmatched_tracks'] and assessment['true_candidate_recall']:
                        categories.append('premature_termination_proxy')
                    if categories:
                        entry = dict(frame=row['frame'], stage=row['stage'], track_id=track['track_id'], assessment=assessment,
                                     categories=categories, row_position=i, online_trace_sha256=sha256(trace) if not cases else run['videos'][seq]['decisions_sha256'])
                        dest.write(json.dumps(entry, separators=(',', ':')) + '\n')
                        if len(cases) < 24:
                            cases.append(entry)
                    for category in categories:
                        key = (category, row['frame'], track['track_id'])
                        if key not in seen:
                            seen.add(key)
                            events[category] += 1
                            affected_frames[category].add(row['frame'])
                            tracks_by_category[category].add(track['track_id'])
        output_path = args.run / (seq + '.txt')
        rows = np.loadtxt(output_path, delimiter=',', ndmin=2)
        first_frame = {int(identity): int(rows[rows[:, 1] == identity, 0].min()) for identity in np.unique(rows[:, 1])}
        duplicate_frames, false_births, overlaps_official_fp = [], [], []
        matched_outputs = set()
        switches = []
        if official:
            video = official['videos'][seq]
            switches = video['events']
            with np.load(video['matches_path'], allow_pickle=False) as matches:
                matched_outputs = set(zip(matches['frame_ids'].tolist(), matches['tracker_ids'].tolist()))
            for switch in switches:
                if switch['gap_since_previous_gt_match'] > 1 and first_frame.get(switch['new_tracker_id']) == switch['frame']:
                    events['occlusion_reappearance_new_id'] += 1
                    affected_frames['occlusion_reappearance_new_id'].add(switch['frame'])
                    tracks_by_category['occlusion_reappearance_new_id'].add(switch['new_tracker_id'])
        for frame in np.unique(rows[:, 0]).astype(int):
            current = rows[rows[:, 0] == frame]
            pred_boxes = current[:, 2:6].copy()
            pred_boxes[:, 2:] += pred_boxes[:, :2]
            truth = [box for _, box in bank['gt'][frame]]
            iou = overlaps(pred_boxes, truth)
            for i, track_id in enumerate(current[:, 1].astype(int)):
                if first_frame[track_id] == frame and (not len(truth) or iou[i].max() < .5):
                    false_births.append(dict(frame=int(frame), track_id=int(track_id), official_fp=(frame, track_id) not in matched_outputs if official else None))
            if len(truth):
                best = iou.argmax(1)
                for target in np.unique(best):
                    chosen = np.flatnonzero((best == target) & (iou[:, target] >= .5))
                    if len(chosen) > 1:
                        ids = current[chosen, 1].astype(int).tolist()
                        duplicate_frames.append(dict(frame=int(frame), track_ids=ids))
                        overlaps_official_fp.extend((frame, identity) for identity in ids if official and (frame, identity) not in matched_outputs)
        for category, values in [('duplicate_output_overlap_proxy', duplicate_frames), ('false_initialization_no_gt_overlap_proxy', false_births)]:
            events[category] = len(values)
            affected_frames[category].update(x['frame'] for x in values)
            tracks_by_category[category].update(identity for x in values for identity in x.get('track_ids', [x.get('track_id')]))
        if focus:
            write_json(args.output / 'BT005_1049_actual_decisions.json', focus)
        record['videos'][seq] = dict(queries=queries, unknown_reference_queries=unknown, known_reference_queries=known,
                                    current_stage_true_candidate_recall=candidate_recall / known if known else None,
                                    visible_reference_queries=visible, full_proposal_recall_given_visible=full_recall / visible if visible else None,
                                    categories={key: dict(events=int(value), affected_frames=len(affected_frames[key]), track_ids=sorted(tracks_by_category[key])) for key, value in events.items()},
                                    examples=cases, duplicate_examples=duplicate_frames[:12], false_birth_examples=false_births[:12],
                                    duplicate_outputs_official_fp_cooccurrences=len(set(overlaps_official_fp)),
                                    official_switch_count=len(switches) if official else None,
                                    official_switch_events=switches, prediction_sha256=sha256(output_path), events_sha256=sha256(evidence),
                                    detection_sha256=bank['detection_sha256'], annotation_sha256=bank['annotation_sha256'], feature_sha256=sha256(fp))
        write_json(args.output / 'manifest.json', record)
        print(seq, 'known/unknown queries', known, unknown, 'events', dict(events), flush=True)
    record['status'] = 'COMPLETE'
    write_json(args.output / 'manifest.json', record)


if __name__ == '__main__':
    main()
