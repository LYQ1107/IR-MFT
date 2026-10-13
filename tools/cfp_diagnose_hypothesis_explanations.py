#!/usr/bin/env python3
"""Retrospective explanation labels; never called by online tracker inference."""
import argparse
import json
from pathlib import Path

import numpy as np

from cfp.provenance import sha256, write_json
from cfp.explanation_diagnosis import explanations




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Immutable retrospective diagnostic required')
    manifest = json.loads((args.replay / 'manifest.json').read_text())
    if manifest['status'] != 'COMPLETE':
        raise ValueError('Complete causal replay required')
    root = Path('/data3/liuyeqiang/IR-MFT_assets/cfp/phase5/recovery_train_queries_v1')
    physical = json.loads((root / 'manifest.json').read_text())
    report = dict(status='COMPLETE', diagnostic_source_sha256=sha256(__file__), diagnostic_module_sha256=sha256('cfp/explanation_diagnosis.py'), replay_manifest_sha256=sha256(args.replay / 'manifest.json'), videos={},
        scope='Post-hoc diagnostic of internal current explanations before minimum-area output filtering. '
              'Reference identity is the independently labelled observed birth detection, not a trusted global physical-ID oracle. '
              'Unknown birth/current labels excluded from correctness; any retained correct challenger is only a current opportunity, '
              'not proof of successful future recovery. Counts are frame/track explanations and consecutive known-wrong episodes, not official IDSW.')
    for seq, entry in manifest['videos'].items():
        pp = root / (seq + '.npz')
        if sha256(pp) != physical['videos'][seq]['sha256']:
            raise ValueError('Frozen physical-label bank drift')
        with np.load(pp, allow_pickle=False) as archive:
            labels, frames = archive['physical_labels'], archive['frame_of_proposal']
        path = args.replay / (seq + '.hypotheses.jsonl')
        if sha256(path) != entry['hypothesis_trace_sha256']:
            raise ValueError('Causal explanation trace drift')
        counts = dict(frames=0, multiple_explanation_frames=0, selected_known=0, selected_correct=0,
                      selected_wrong=0, selected_unknown=0, wrong_with_known_correct_challenger=0,
                      correct_with_known_wrong_challenger=0, wrong_without_known_correct_challenger=0,
                      consecutive_known_wrong_episodes=0, selected_branch_changes=0)
        previous, last_winner, seen = {}, None, 0
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event['frame'] != seen + 1:
                raise ValueError('Complete consecutive causal frames required')
            seen += 1
            by_branch = explanations(event, labels, frames)
            counts['frames'] += 1
            counts['multiple_explanation_frames'] += int(len(by_branch) > 1)
            counts['selected_branch_changes'] += int(last_winner is not None and event['winner'] != last_winner)
            last_winner = event['winner']
            selected = by_branch[event['winner']]
            for birth, value in selected.items():
                if not value['known']:
                    counts['selected_unknown'] += 1
                    previous[birth] = (event['frame'], False)
                    continue
                counts['selected_known'] += 1
                alternatives = [rows[birth] for bid, rows in by_branch.items() if bid != event['winner'] and birth in rows]
                if value['correct']:
                    counts['selected_correct'] += 1
                    counts['correct_with_known_wrong_challenger'] += int(any(a['known'] and not a['correct'] for a in alternatives))
                else:
                    counts['selected_wrong'] += 1
                    possible = any(a['known'] and a['correct'] for a in alternatives)
                    counts['wrong_with_known_correct_challenger'] += int(possible)
                    counts['wrong_without_known_correct_challenger'] += int(not possible)
                    old = previous.get(birth)
                    counts['consecutive_known_wrong_episodes'] += int(old is None or old != (event['frame'] - 1, True))
                previous[birth] = (event['frame'], not value['correct'])
        if seen != entry['frames']:
            raise ValueError('Missing full-video explanation coverage')
        report['videos'][seq] = dict(counts=counts, physical_bank_sha256=sha256(pp), trace_sha256=sha256(path),
                                   selected_known_accuracy=counts['selected_correct']/max(counts['selected_known'], 1))
    write_json(args.output, report)
    print(json.dumps({k: v['counts'] for k, v in report['videos'].items()}))


if __name__ == '__main__':
    main()
