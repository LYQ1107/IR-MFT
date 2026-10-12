#!/usr/bin/env python3
"""Analyze actual online decisions offline, retaining unknowns and stage scopes."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from cfp.provenance import sha256, snapshot_sources, write_json
from cfp.recovery_supervision import physical_bank, history_view, modal_identity
from cfp.recoverability_diagnosis import EarlyReference, assess_candidate_row, accumulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--record-events', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Analysis output is immutable')
    run = json.loads((args.run / 'manifest.json').read_text())
    if run['status'] != 'COMPLETE' or run['probe'] or run['partition'] not in ('train', 'dev'):
        raise ValueError('Complete train/dev actual trace required')
    root = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    if shutil.disk_usage(root).free < 40 * 2**30:
        raise OSError('40 GiB reserve')
    args.output.mkdir(parents=True)
    feature_root = root / ('phase2' if run['partition'] == 'train' else 'phase1') / f'causal_{run["partition"]}_detection_features_postneck_seed{run["seed"]}'
    report = dict(status='RUNNING', partition=run['partition'], seed=run['seed'],
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        run_manifest_sha256=sha256(args.run / 'manifest.json'), videos={},
        source_snapshots=snapshot_sources(args.output, [__file__, 'cfp/recoverability_diagnosis.py',
            'cfp/recovery_supervision.py', 'configs/cfp/phase_vi_protocol_v1.json']),
        reference='First >=3 distinct known accepted past observations, >=.6 consensus, '
            'at most first five, frozen per track. Rolling modal reference sensitivity separately reported.',
        geometry_gate='Appearance-blending eligibility, not a hard Hungarian mask. Threshold feasibility separately measured.',
        scope='Retrospective local row opportunity estimates on fixed logged states. '
            'Not a global association oracle, performance bound, causal root-cause allocation or online signal. '
            'Wrong events require known reference AND known assigned physical label. Unknown is never wrong. '
            'Final decision counts deduplicate stage rows per frame/track; no candidate is artificially GT-added.')
    write_json(args.output / 'manifest.json', report)
    for seq in run['videos']:
        bank = physical_bank(seq, run['partition'])
        fp = feature_root / (seq + '.npz')
        if sha256(fp) != json.loads(fp.with_suffix('.provenance.json').read_text())['features_sha256']:
            raise ValueError('Feature drift')
        with np.load(fp, allow_pickle=False) as archive:
            features = archive['rgb']
        trace = args.run / (seq + '.decisions.jsonl.gz')
        if sha256(trace) != run['videos'][seq]['decisions_sha256']:
            raise ValueError('Trace drift')
        references = EarlyReference()
        all_counts, final_counts, rolling_counts = Counter(), Counter(), Counter()
        stages, strata = defaultdict(Counter), defaultdict(Counter)
        final, current_frame, examples = {}, None, []
        margins, separability = defaultdict(list), defaultdict(list)
        event_path = args.output / (seq + '.events.jsonl.gz')
        dest = gzip.open(event_path, 'wt', compresslevel=3) if args.record_events else None

        def flush():
            for (frame, identity), (assessment, stage) in final.items():
                accumulate(final_counts, assessment)
                accumulate(strata[stage], assessment)
                if assessment['reference_known']:
                    group = 'wrong' if assessment['wrong'] else ('correct' if assessment['correct'] else 'other')
                    if assessment['chosen_margin'] is not None:
                        margins[group].append(assessment['chosen_margin'])
                    if assessment['mean_history_candidate_separability'] is not None:
                        separability[group].append(assessment['mean_history_candidate_separability'])
                if assessment.get('wrong') or assessment.get('unmatched') and assessment.get('visible'):
                    entry = dict(frame=frame, track_id=identity, stage=stage, assessment=assessment)
                    if len(examples) < 16:
                        examples.append(entry)
                    if dest:
                        dest.write(json.dumps(entry, separators=(',', ':'), allow_nan=False) + '\n')
            final.clear()

        try:
            with gzip.open(trace, 'rt') as stream:
                for line in stream:
                    row = json.loads(line)
                    if current_frame is not None and current_frame != row['frame']:
                        flush()
                    current_frame = row['frame']
                    for i, track in enumerate(row['tracks']):
                        target = references.update(track, bank['labels'])
                        assessment = assess_candidate_row(row, i, bank, features, target, run['association'])
                        accumulate(all_counts, assessment)
                        accumulate(stages[row['stage']], assessment)
                        rolling, _ = modal_identity([index for _, index in history_view(track)], bank['labels'])
                        rolling_assessment = assess_candidate_row(row, i, bank, features, rolling, run['association'])
                        accumulate(rolling_counts, rolling_assessment)
                        key = (int(row['frame']), int(track['track_id']))
                        if key in final and final[key][0].get('assigned_column') is not None:
                            raise ValueError('Already accepted track appears in a later stage')
                        final[key] = (assessment, row['stage'])
            flush()
        finally:
            if dest:
                dest.close()
        def summary(values):
            return {key: dict(count=len(v), mean=float(np.mean(v)), p10_p50_p90=np.quantile(v, [.1, .5, .9]).tolist()) for key, v in values.items() if v}
        report['videos'][seq] = dict(final_decisions=dict(final_counts), stage_rows=dict(all_counts),
            by_stage={k: dict(v) for k, v in stages.items()}, final_by_stage={k: dict(v) for k, v in strata.items()},
            rolling_reference_stage_sensitivity=dict(rolling_counts), margin=summary(margins),
            mean_history_separability=summary(separability), examples=examples,
            trace_sha256=sha256(trace), feature_sha256=sha256(fp),
            detection_sha256=bank['detection_sha256'], annotation_sha256=bank['annotation_sha256'],
            events_sha256=sha256(event_path) if args.record_events else None)
        write_json(args.output / 'manifest.json', report)
        print(seq, 'final decisions', dict(final_counts), flush=True)
    report.update(status='COMPLETE', storage_free_gib=shutil.disk_usage(root).free / 2**30)
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
