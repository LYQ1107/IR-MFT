#!/usr/bin/env python3
"""Training-only oracle/episode diagnosis, separate from inference and MOT scores."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from cfp.provenance import sha256, snapshot_sources, write_json
from cfp.recoverability_diagnosis import EarlyReference
from cfp.recoverability_oracle import joint_candidate_oracle
from cfp.recovery_supervision import physical_bank


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('New immutable analysis directory required')
    run = json.loads((args.run / 'manifest.json').read_text())
    if run['status'] != 'COMPLETE' or run['partition'] != 'train' or run['probe']:
        raise ValueError('Oracle intervention limited to complete training traces')
    args.output.mkdir(parents=True)
    report = dict(status='RUNNING', run_manifest_sha256=sha256(args.run / 'manifest.json'), videos={},
        source_snapshots=snapshot_sources(args.output, [__file__, 'cfp/recoverability_oracle.py', 'cfp/recoverability_diagnosis.py']),
        scope='Retrospective joint one-to-one opportunities on fixed actual states and early physical references. '
            'No trajectory replay, metric improvement or unconditional ceiling claimed. '
            'Unknown references retain original columns; every other row has dedicated NULL. '
            'Expanded graph uses all existing current proposals, bypassing stage/cost gating, never GT-added candidates. '
            'Consecutive known-wrong frame/track episodes are operational, not official IDSW.')
    write_json(args.output / 'manifest.json', report)
    for seq in run['videos']:
        bank = physical_bank(seq, 'train')
        trace = args.run / (seq + '.decisions.jsonl.gz')
        if sha256(trace) != run['videos'][seq]['decisions_sha256']:
            raise ValueError('Decision trace drift')
        refs = EarlyReference()
        frame, final = None, {}
        stage_counts, expanded_counts = Counter(), Counter()
        last_wrong, lengths, active = {}, [], {}
        episodes = 0
        def flush():
            nonlocal episodes
            if frame is None:
                return
            lo, hi = bank['offsets'][frame - 1:frame + 1]
            entries = list(final.values())
            for expanded, counter in [(False, stage_counts), (True, expanded_counts)]:
                _, result = joint_candidate_oracle(entries, np.arange(lo, hi), bank['labels'], expanded)
                counter.update(result)
            for key, entry in final.items():
                target, chosen = entry['target'], entry['chosen']
                wrong = target is not None and chosen is not None and bank['labels'][chosen] >= 0 and bank['labels'][chosen] != target
                if wrong:
                    if last_wrong.get(key) != frame - 1:
                        if key in active:
                            lengths.append(active.pop(key))
                        episodes += 1; active[key] = 0
                    active[key] += 1; last_wrong[key] = frame
                elif key in active:
                    lengths.append(active.pop(key))
            final.clear()
        with gzip.open(trace, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                if frame is not None and frame != row['frame']:
                    flush()
                frame = row['frame']
                matching = dict(row['matches'])
                for i, track in enumerate(row['tracks']):
                    target = refs.update(track, bank['labels'])
                    chosen = row['proposals'][matching[i]] if i in matching else None
                    key = int(track['track_id'])
                    if key in final and final[key]['chosen'] is not None:
                        raise ValueError('Accepted row repeated in later stage')
                    final[key] = dict(target=target, chosen=chosen, candidates=row['proposals'])
            flush()
        lengths.extend(active.values())
        report['videos'][seq] = dict(stage_graph=dict(stage_counts), expanded_graph=dict(expanded_counts),
            known_wrong_episodes=episodes, wrong_episode_frames=sum(lengths),
            episode_length_p50_p90_max=[float(np.median(lengths)), float(np.quantile(lengths, .9)), max(lengths)] if lengths else None,
            trace_sha256=sha256(trace), detection_sha256=bank['detection_sha256'])
        write_json(args.output / 'manifest.json', report)
        print(seq, report['videos'][seq], flush=True)
    report['status'] = 'COMPLETE'
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
