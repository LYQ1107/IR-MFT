#!/usr/bin/env python3
"""Evaluate causal MOT files with pinned official TrackEval; retain all values."""
import argparse
import configparser
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from cfp.provenance import sha256, write_json


PINNED_COMMIT = '12c8791b303e0a0b50f753af204249e622d0281a'


def validate_mot(path, length):
    seen = set()
    previous_frame = 0
    count = 0
    with Path(path).open() as stream:
        for row in csv.reader(stream):
            if len(row) < 7:
                raise ValueError(f'{path}: expected >=7 columns')
            values = np.asarray(row[:7], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f'{path}: nonfinite MOT entry')
            frame, identity = int(values[0]), int(values[1])
            if frame != values[0] or identity != values[1] or identity <= 0:
                raise ValueError(f'{path}: frame/ID must be positive integers')
            if not 1 <= frame <= length or frame < previous_frame:
                raise ValueError(f'{path}: invalid/unsorted frame')
            if (frame, identity) in seen or np.any(values[4:6] <= 0):
                raise ValueError(f'{path}: duplicate frame/ID or degenerate bbox')
            seen.add((frame, identity))
            previous_frame = frame
            count += 1
    return count


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(v) for key, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate(data_root, source_split, sequences, predictions, output, trackeval_root):
    commit = subprocess.check_output(['git', '-C', str(trackeval_root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != PINNED_COMMIT:
        raise ValueError(f'Unexpected TrackEval commit: {commit}')
    if 'trackeval' in sys.modules:
        raise RuntimeError('TrackEval already loaded; use separate evaluation process')
    # Pinned upstream still uses removed builtin aliases in its dataset loader.
    # Preserve upstream source and isolate compatibility to this evaluator process.
    for alias, value in [('float', float), ('int', int), ('bool', bool)]:
        if alias not in np.__dict__:
            setattr(np, alias, value)
    sys.path.insert(0, str(trackeval_root))
    import trackeval
    assert Path(trackeval.__file__).resolve().is_relative_to(Path(trackeval_root).resolve())
    output = Path(output)
    track_dir = output / 'inputs/cfp/data'
    track_dir.mkdir(parents=True, exist_ok=True)
    seq_info, artifacts = {}, {}
    for seq in sequences:
        seq_root = Path(data_root) / source_split / seq
        info = configparser.ConfigParser()
        info.read(seq_root / 'seqinfo.ini')
        seq_info[seq] = info.getint('Sequence', 'seqLength')
        source = Path(predictions) / f'{seq}.txt'
        count = validate_mot(source, seq_info[seq])
        dest = track_dir / source.name
        shutil.copyfile(source, dest)
        artifacts[seq] = {'gt_sha256': sha256(seq_root / 'gt/gt.txt'),
                          'prediction_sha256': sha256(source), 'prediction_rows': count,
                          'frames': seq_info[seq]}
    evaluator = trackeval.Evaluator({'PRINT_RESULTS': False, 'PRINT_CONFIG': False,
                                     'TIME_PROGRESS': False, 'PLOT_CURVES': False})
    dataset = trackeval.datasets.MotChallenge2DBox({
        'GT_FOLDER': str(Path(data_root) / source_split), 'TRACKERS_FOLDER': str(output / 'inputs'),
        'TRACKERS_TO_EVAL': ['cfp'], 'OUTPUT_FOLDER': str(output / 'trackeval_output'),
        'BENCHMARK': 'MFT26IR', 'SPLIT_TO_EVAL': source_split, 'SKIP_SPLIT_FOL': True,
        'SEQ_INFO': seq_info, 'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
        'CLASSES_TO_EVAL': ['pedestrian'], 'PRINT_CONFIG': False, 'DO_PREPROC': True})
    started = time.perf_counter()
    metric_objects = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    results, messages = evaluator.evaluate([dataset], metric_objects)
    raw = results[dataset.get_name()]['cfp']
    flat = {}
    for seq, classes in raw.items():
        flat[seq] = {}
        for metric in metric_objects:
            values = classes['pedestrian'][metric.get_name()]
            for field in metric.fields:
                val = values[field]
                if field in ('FP_per_frame', 'MOTP_sum'):
                    flat[seq][field] = float(val)
                elif field in metric.float_fields or field in metric.float_array_fields:
                    flat[seq][field] = float(np.mean(val) * 100)
                else:
                    flat[seq][field] = jsonable(val)
        flat[seq].update(jsonable(classes['pedestrian']['Count']))
    report = {'trackeval_commit': commit, 'metric_sha256': {m: sha256(Path(trackeval_root) / 'trackeval/metrics' / f'{m}.py') for m in ('hota', 'clear', 'identity')},
              'source_split': source_split, 'sequences': sequences, 'artifacts': artifacts,
              'evaluation_seconds': time.perf_counter() - started,
              'units': 'rate metrics in percent; counts, FP_per_frame, MOTP_sum unchanged; alpha-mean used only for float array fields',
              'metrics': flat, 'raw_official_results': jsonable(raw), 'messages': messages}
    write_json(output / 'metrics.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--partition', choices=['train', 'dev', 'historical_test'], required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trackeval-root', type=Path, required=True)
    parser.add_argument('--derived-gt-root', type=Path, help='Versioned audited GT; hashes must match the selected protocol')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    gt_root = args.data_root
    if protocol.get('derived_gt_sha256'):
        if args.derived_gt_root is None:
            raise ValueError('Audited protocol requires --derived-gt-root; raw duplicate GT is invalid')
        gt_root = args.derived_gt_root
        for seq in protocol[args.partition]:
            split = 'test' if args.partition == 'historical_test' else 'train'
            if sha256(gt_root / split / seq / 'gt/gt.txt') != protocol['derived_gt_sha256'][f'{split}/{seq}']:
                raise ValueError(f'Derived GT hash mismatch for {seq}')
    elif args.derived_gt_root is not None:
        raise ValueError('Cannot change GT root outside the protocol')
    result = evaluate(gt_root, 'test' if args.partition == 'historical_test' else 'train',
                      protocol[args.partition], args.predictions, args.output, args.trackeval_root)
    print(json.dumps(result['metrics']['COMBINED_SEQ'], indent=2))


if __name__ == '__main__':
    main()
