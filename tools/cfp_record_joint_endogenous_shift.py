#!/usr/bin/env python3
"""Inner baseline and fitting warm-label coverage versus fixed native diagnoses."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from cfp.joint_identity_supervision import joint_identity_targets
from cfp.joint_endogenous_diagnosis import classify_history, classify_selection
from cfp.recoverability_diagnosis import EarlyReference
from cfp.recovery_supervision import physical_bank, history_view
from cfp.provenance import sha256, snapshot_sources, write_json


def measure_row(tracks, references, current, labels, selected):
    identities = torch.tensor([-1 if x is None else x for x in references], dtype=torch.int64)
    supervision = joint_identity_targets('offline', identities, torch.from_numpy(labels[current].copy()))
    counts = Counter(detections=len(current), track_slots=len(tracks))
    for track, reference in zip(tracks, references):
        h = classify_history(track, labels, reference)
        counts['track_history_'+h['state']] += 1
        if h['known_wrong'] is not None:
            counts['known_history_observations'] += h['known']
            counts['known_wrong_history_observations'] += h['known_wrong']
    for j, (target, reason) in enumerate(zip(supervision['target'].tolist(), supervision['reason'])):
        counts['target_'+reason] += 1
        if selected is not None:
            counts[classify_selection(selected[j], target, len(tracks))] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    for name in ['root', 'observed', 'output']:
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise ValueError('New immutable offline shift report required')
    diagnosis = json.loads((a.observed/'manifest.json').read_text())
    config_path = Path('configs/cfp/phase_vii_centered_joint_probe_v2.json')
    config = json.loads(config_path.read_text())
    if diagnosis['status'] != 'COMPLETE_ALL_NINE_INNER_ENDOGENOUS_IDENTITY_DIAGNOSES' or diagnosis['video'] != 'SN-001' or len(diagnosis['runs']) != 9 or any(not v['prediction_byte_equal'] or v['online_physical_labels_loaded'] for v in diagnosis['runs'].values()):
        raise ValueError('All nine actual GT-free inner native parities must precede offline comparison')
    assets = a.root.parent
    teacher = assets/'phase5/cross_train_gru_cfp_s0_c1_v1'
    tm = json.loads((teacher/'manifest.json').read_text())
    trace = teacher/'SN-001.decisions.jsonl.gz'
    if tm['status'] != 'COMPLETE' or tm['partition'] != 'train' or tm['seed'] != 0 or sha256(trace) != tm['videos']['SN-001']['decisions_sha256']:
        raise ValueError('Preserved actual baseline decision trace required')
    bank = physical_bank('SN-001', 'train')
    if bank['detection_sha256'] != diagnosis['detection_sha256'] or bank['annotation_sha256'] != diagnosis['offline_annotation_sha256']:
        raise ValueError('Same independent offline physical label source required')
    refs = EarlyReference(); baseline = Counter()
    with gzip.open(trace, 'rt') as stream:
        for row in map(json.loads, stream):
            for track in row['tracks']:
                if any(t >= row['frame'] or bank['frame_of_proposal'][index] != t for t, index in history_view(track)):
                    raise ValueError('Historical baseline trace is not strictly past')
            identities = [refs.update(t, bank['labels']) for t in row['tracks']]
            if row['stage'] != 'high' or row['frame'] == 1:
                continue
            current = np.asarray(row['proposals'], dtype=np.int64)
            matches = {int(j): int(i) for i, j in row['matches']}
            baseline.update(measure_row(row['tracks'], identities, current, bank['labels'], [matches.get(j, -1) for j in range(len(current))]))
            baseline['frames'] += 1
    queries = a.root/'joint_actual_queries_s0_v2'
    qm = json.loads((queries/'manifest.json').read_text())
    if qm['status'] != 'COMPLETE':
        raise ValueError('Actual immutable warm training labels required')
    warm = {}
    for video in config['fit_videos']+config['calibration_videos']:
        path = queries/(video+'.npz')
        if sha256(path) != qm['videos'][video]['sha256']:
            raise ValueError('Warm query data drift')
        physical = bank if video == 'SN-001' else physical_bank(video, 'train')
        with np.load(path, allow_pickle=False) as source:
            q = {k: source[k] for k in source.files}
        counts = Counter()
        for i, frame in enumerate(q['frame']):
            lo, hi = q['track_offsets'][i:i+2]; dl, dh = q['detection_offsets'][i:i+2]
            if not np.array_equal(q['detection_identity'][dl:dh], physical['labels'][q['candidates'][dl:dh]]):
                raise ValueError('Warm current labels differ from independent immutable physical source')
            tracks = []
            for k in range(lo, hi):
                valid = q['indices'][k] >= 0
                indices, times = q['indices'][k][valid], q['times'][k][valid]
                if (times >= frame).any() or not np.array_equal(physical['frame_of_proposal'][indices], times):
                    raise ValueError('Warm source contains current/future history')
                tracks.append(dict(initial_proposal=None, history=list(zip(times.tolist(), indices.tolist()))))
            refs = [None if x < 0 else int(x) for x in q['track_identity'][lo:hi]]
            counts.update(measure_row(tracks, refs, q['candidates'][dl:dh], physical['labels'], None)); counts['frames'] += 1
        warm[video] = dict(counts=counts, query_sha256=sha256(path), annotation_sha256=physical['annotation_sha256'])
    report = dict(status='COMPLETE_INNER_ENDOGENOUS_AND_FIT_WARM_COVERAGE_SHIFT', inner='SN-001', outer_not_read='SN-015',
                  native_observed_manifest_sha256=sha256(a.observed/'manifest.json'), native_runs=diagnosis['runs'],
                  baseline_inner=dict(counts=baseline, trace_sha256=sha256(trace), trace=str(trace)), fitting_and_inner_warm=warm,
                  sources=snapshot_sources(a.output.with_suffix('.sources'), [__file__, 'cfp/joint_endogenous_diagnosis.py']),
                  scope='Offline only after all9native output parities. Same physical proposal labeling. Warm/ordinarybaseline/joint endogenous set membership differs, so descriptive shift, not matched-head performance comparison or isolated causal proof. No outer weights, labels, frames or model-score selection. No new training/native benefit claim.')
    write_json(a.output, report)
    print(json.dumps(dict(status=report['status'], baseline_counts=dict(baseline), warm_videos=len(warm))))


if __name__ == '__main__':
    main()
