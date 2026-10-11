#!/usr/bin/env python3
"""Create immutable evaluation GT with explicit distractors for ambiguous IDs."""
import argparse
from collections import Counter
import configparser
import csv
import json
from pathlib import Path
import shutil

from cfp.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--base-protocol', type=Path, required=True)
    p.add_argument('--protocol-output', type=Path, required=True)
    p.add_argument('--derived-gt-root', type=Path, required=True)
    p.add_argument('--audit-output', type=Path, required=True)
    args = p.parse_args()
    if args.protocol_output.exists() or args.derived_gt_root.exists():
        raise FileExistsError('Refuse to overwrite an audited protocol/GT')
    protocol = json.loads(args.base_protocol.read_text())
    reports = {}
    all_ambiguous = []
    for split in ('train', 'test'):
        annotation = json.loads((args.data_root / 'annotations' / (split + '.json')).read_text())
        for video in annotation['videos']:
            seq = video['file_name']
            source = args.data_root / split / seq / 'gt/gt.txt'
            rows = list(csv.reader(source.open()))
            counter = Counter((int(float(row[0])), int(float(row[1]))) for row in rows if float(row[6]) > 0 and int(float(row[7])) == 1)
            ambiguous = {key for key, count in counter.items() if count > 1}
            next_id = max(int(float(row[1])) for row in rows) + 1
            derived_rows, ignored = [], []
            for row in rows:
                row = row.copy()
                key = (int(float(row[0])), int(float(row[1])))
                if key in ambiguous and float(row[6]) > 0 and int(float(row[7])) == 1:
                    ignored.append({'frame_id': key[0], 'original_track_id': key[1], 'bbox_xywh': row[2:6], 'replacement_id': next_id})
                    # Official MOT distractor preprocessing removes assigned predictions
                    # and excludes these boxes from evaluated GT. No metric kernel change.
                    row[1], row[7] = str(next_id), '8'
                    next_id += 1
                derived_rows.append(row)
            target = args.derived_gt_root / split / seq / 'gt/gt.txt'
            target.parent.mkdir(parents=True)
            with target.open('w', newline='') as stream:
                csv.writer(stream, lineterminator='\n').writerows(derived_rows)
            shutil.copyfile(source.parent.parent / 'seqinfo.ini', target.parent.parent / 'seqinfo.ini')
            reports[f'{split}/{seq}'] = {'source_sha256': sha256(source), 'derived_sha256': sha256(target),
                                        'ambiguous_identity_keys': sorted(ambiguous), 'distractor_rows': ignored}
            all_ambiguous.extend([dict(source_split=split, sequence=seq, frame_id=frame, track_id=identity) for frame, identity in sorted(ambiguous)])
    if any(x['source_split'] == 'test' for x in all_ambiguous):
        raise ValueError('Historical test has unexpected duplicates; do not silently alter historical reproduction')
    protocol.update(protocol_id='cfp_video_holdout_v2_annotation_audit', supersedes_sha256=sha256(args.base_protocol),
                    revision_reason='Malformed duplicate frame/identity GT detected before any development evaluation; all ambiguous boxes masked equally for every method',
                    identity_annotation_policy='exclude_all_duplicate_keys',
                    ambiguous_identity_keys=all_ambiguous,
                    evaluation_gt_policy='derived duplicate-ID boxes assigned unique dummy IDs and official MOT distractor class 8; all other GT unchanged',
                    derived_gt_sha256={k: v['derived_sha256'] for k, v in reports.items()},
                    derived_gt_root_hint=str(args.derived_gt_root))
    write_json(args.protocol_output, protocol)
    write_json(args.audit_output, {'base_protocol_sha256': sha256(args.base_protocol),
                                   'new_protocol_sha256': sha256(args.protocol_output), 'videos': reports,
                                   'ambiguous_identity_keys': all_ambiguous})
    print(json.dumps({'protocol': str(args.protocol_output), 'ambiguous_identity_keys': all_ambiguous,
                      'ignored_boxes': sum(len(x['distractor_rows']) for x in reports.values())}, indent=2))


if __name__ == '__main__':
    main()
