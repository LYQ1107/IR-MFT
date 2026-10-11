#!/usr/bin/env python3
"""Observe exact upstream CLEAR switches without changing its matching code."""
import argparse
import ast
import configparser
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from cfp.provenance import sha256, write_json


PINNED_COMMIT = '12c8791b303e0a0b50f753af204249e622d0281a'


def capture_clear(metric, data, metric_path, match_observer=None):
    tree = ast.parse(metric_path.read_text())
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For) and
                any(isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Subscript) and
                    isinstance(statement.target.value, ast.Name) and statement.target.value.id == 'res' and
                    isinstance(statement.target.slice, ast.Constant) and statement.target.slice.value == 'IDSW'
                    for statement in node.body))
    increment = next(statement for statement in loop.body if isinstance(statement, ast.AugAssign) and
                     isinstance(statement.target, ast.Subscript) and isinstance(statement.target.value, ast.Name) and
                     statement.target.value.id == 'res' and statement.target.slice.value == 'IDSW')
    observation_line = min(statement.lineno for statement in loop.body if statement.lineno > increment.lineno)
    events, previous_frames = [], {}
    observed = set()
    def observer(frame, event, _):
        if frame.f_code.co_name != 'eval_sequence' or Path(frame.f_code.co_filename).resolve() != metric_path.resolve():
            return None
        if event == 'line' and frame.f_lineno == observation_line:
            values = frame.f_locals
            t = int(values['t'])
            if t in observed:
                raise ValueError('Duplicate CLEAR observation frame')
            observed.add(t)
            if match_observer is not None:
                match_observer(t, values)
            for position, gt in enumerate(values['matched_gt_ids']):
                gt = int(gt)
                if values['is_idsw'][position]:
                    row, col = int(values['match_rows'][position]), int(values['match_cols'][position])
                    events.append(dict(frame=t + 1, gt_id_relabelled=gt,
                                       previous_tracker_id_relabelled=int(values['prev_matched_tracker_ids'][position]),
                                       new_tracker_id_relabelled=int(values['matched_tracker_ids'][position]),
                                       gt_xywh=data['gt_dets'][t][row].tolist(),
                                       prediction_xywh=data['tracker_dets'][t][col].tolist(),
                                       matched_iou=float(data['similarity_scores'][t][row, col]),
                                       gap_since_previous_gt_match=t - previous_frames[gt]))
                previous_frames[gt] = t
        return observer
    if sys.gettrace() is not None:
        raise RuntimeError('Do not replace an existing Python tracer')
    try:
        sys.settrace(observer)
        result = metric.eval_sequence(data)
    finally:
        sys.settrace(None)
    if len(events) != int(result['IDSW']):
        raise ValueError('Observed switches differ from exact upstream CLEAR result')
    return result, events


def preprocess_with_id_maps(dataset, raw, dataset_path):
    maps = {}
    def observer(frame, event, _):
        if frame.f_code.co_name != 'get_preprocessed_seq_data' or Path(frame.f_code.co_filename).resolve() != dataset_path.resolve():
            return None
        if event == 'return':
            maps['gt'] = np.asarray(frame.f_locals['unique_gt_ids'], dtype=int)
            maps['tracker'] = np.asarray(frame.f_locals['unique_tracker_ids'], dtype=int)
        return observer
    if sys.gettrace() is not None:
        raise RuntimeError('Do not replace an existing Python tracer')
    try:
        sys.settrace(observer)
        data = dataset.get_preprocessed_seq_data(raw, 'pedestrian')
    finally:
        sys.settrace(None)
    if len(maps['gt']) != data['num_gt_ids'] or len(maps['tracker']) != data['num_tracker_ids']:
        raise ValueError('Upstream relabelling map capture failed')
    return data, maps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation-root', type=Path, required=True)
    parser.add_argument('--derived-gt-root', type=Path, required=True)
    parser.add_argument('--trackeval-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads((args.evaluation_root / 'metrics.json').read_text())
    if report['source_split'] != 'train':
        raise ValueError('Research event diagnosis excludes historical test')
    commit = subprocess.check_output(['git', '-C', str(args.trackeval_root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != PINNED_COMMIT or report['trackeval_commit'] != commit:
        raise ValueError('Unexpected evaluator version')
    for alias, value in [('float', float), ('int', int), ('bool', bool)]:
        if alias not in np.__dict__:
            setattr(np, alias, value)
    sys.path.insert(0, str(args.trackeval_root))
    import trackeval
    metric_path = args.trackeval_root / 'trackeval/metrics/clear.py'
    dataset_path = args.trackeval_root / 'trackeval/datasets/mot_challenge_2d_box.py'
    if sha256(metric_path) != report['metric_sha256']['clear']:
        raise ValueError('Metric source changed since evaluation')
    seq_info = {}
    for sequence, artifact in report['artifacts'].items():
        directory = args.derived_gt_root / 'train' / sequence
        if sha256(directory / 'gt/gt.txt') != artifact['gt_sha256'] or sha256(args.evaluation_root / 'inputs/cfp/data' / (sequence + '.txt')) != artifact['prediction_sha256']:
            raise ValueError('Evaluation inputs changed')
        info = configparser.ConfigParser()
        info.read(directory / 'seqinfo.ini')
        seq_info[sequence] = info.getint('Sequence', 'seqLength')
    dataset = trackeval.datasets.MotChallenge2DBox(dict(GT_FOLDER=str(args.derived_gt_root / 'train'),
              TRACKERS_FOLDER=str(args.evaluation_root / 'inputs'), TRACKERS_TO_EVAL=['cfp'],
              BENCHMARK='MFT26IR', SPLIT_TO_EVAL='train', SKIP_SPLIT_FOL=True, SEQ_INFO=seq_info,
              GT_LOC_FORMAT='{gt_folder}/{seq}/gt/gt.txt', CLASSES_TO_EVAL=['pedestrian'], PRINT_CONFIG=False, DO_PREPROC=True))
    videos = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for sequence in report['sequences']:
        raw = dataset.get_raw_seq_data('cfp', sequence)
        data, maps = preprocess_with_id_maps(dataset, raw, dataset_path)
        matched = []
        def record_matches(t, values):
            for position, identity in enumerate(values['matched_gt_ids']):
                row, col = int(values['match_rows'][position]), int(values['match_cols'][position])
                matched.append((t + 1, int(maps['gt'][identity]), int(maps['tracker'][values['matched_tracker_ids'][position]]),
                                float(data['similarity_scores'][t][row, col])))
        result, events = capture_clear(trackeval.metrics.CLEAR({'PRINT_CONFIG': False}), data, metric_path, record_matches)
        for event in events:
            event['gt_id'] = int(maps['gt'][event['gt_id_relabelled']])
            event['previous_tracker_id'] = int(maps['tracker'][event['previous_tracker_id_relabelled']])
            event['new_tracker_id'] = int(maps['tracker'][event['new_tracker_id_relabelled']])
        for field in ['IDSW', 'CLR_TP', 'CLR_FP', 'CLR_FN', 'Frag']:
            if result[field] != report['metrics'][sequence][field]:
                raise ValueError(f'Official CLEAR trace parity failed: {sequence} {field}')
        match_path = args.output.with_name(args.output.stem + '_' + sequence + '_matches.npz')
        if match_path.exists():
            raise FileExistsError(match_path)
        rows = np.asarray(matched).reshape(-1, 4)
        if len(rows) != int(result['CLR_TP']):
            raise ValueError('Recorded CLEAR matches differ from official TP')
        np.savez_compressed(match_path, frame_ids=rows[:, 0].astype(np.int32), gt_ids=rows[:, 1].astype(np.int64),
                            tracker_ids=rows[:, 2].astype(np.int64), iou=rows[:, 3].astype(np.float32))
        videos[sequence] = dict(events=events, official_idswitch_count=int(result['IDSW']),
                                matches_path=str(match_path), matches_sha256=sha256(match_path), official_match_count=len(rows),
                                clear_count_parity=True, gt_id_inverse_map=maps['gt'].tolist(), tracker_id_inverse_map=maps['tracker'].tolist())
        print(sequence, 'exact official switch events', len(events), flush=True)
    write_json(args.output, dict(status='COMPLETE', source_sha256=sha256(__file__),
               evaluation_sha256=sha256(args.evaluation_root / 'metrics.json'), trackeval_commit=commit,
               metric_sha256=sha256(metric_path), dataset_sha256=sha256(dataset_path), videos=videos,
               scope='Read-only offline train/dev error diagnosis; GT never supplied to tracking or inference. Exact pinned CLEAR matching observed by Python tracing; not an alternative metric.',
               match_label_policy='Official metric continuity-biased matches, for retrospective error analysis; not independent physical crop identity supervision'))


if __name__ == '__main__':
    main()
