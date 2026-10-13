#!/usr/bin/env python3
"""Collect complete identity-coordinate replays and matching baselines.

The identity-seed campaign deliberately changes the frozen RGB coordinate and
the ordinary producer together.  Its disabled baseline therefore lives in the
actual state replays, rather than in the earlier seed-0 baseline directory.
This tool makes one immutable, symlink-only aggregate per identity seed and
evaluates it with the pinned protocol.  It then joins those records with the
four candidate replay families produced by the campaign pipeline.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from cfp.provenance import sha256, write_json


ROOT = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
PHASE6 = ROOT / 'phase6'
SEQUENCES = tuple(json.loads(Path('configs/cfp/protocol_v2.json').read_text())['train'])
FIELDS = ('HOTA', 'AssA', 'DetA', 'IDF1', 'MOTA', 'IDSW')


def _command_output(command):
    marker = '--output'
    if marker not in command:
        raise ValueError('missing --output in command')
    return Path(command[command.index(marker) + 1])


def _completed_json(path, label):
    data = json.loads(Path(path).read_text())
    if data.get('status') != 'COMPLETE':
        raise ValueError(f'{label} is not complete')
    return data


def _read_metrics(path, label):
    data = json.loads(Path(path).read_text())
    if not data.get('metrics') or not data.get('sequences'):
        raise ValueError(f'{label} has no complete metric payload')
    return data


def _state_records(states_summary):
    records = {}
    for run_name, entry in states_summary['runs'].items():
        replay = entry['replay']
        if replay.get('status') != 'COMPLETE':
            raise ValueError(f'incomplete state replay: {run_name}')
        config = replay['config']
        sequence = config['validation_video']
        seed = int(replay['seed'])
        output = _command_output(entry['command'])
        manifest_path = output / 'manifest.json'
        manifest = _completed_json(manifest_path, run_name)
        if sequence not in manifest['videos']:
            raise ValueError(f'{run_name}: validation video missing from replay')
        if config['validation_video'] != sequence:
            raise ValueError(f'{run_name}: state validation mismatch')
        if config.get('partition') != 'train' or sequence not in SEQUENCES:
            raise ValueError(f'{run_name}: unexpected partition/sequence')
        prediction = output / f'{sequence}.txt'
        if sha256(prediction) != manifest['videos'][sequence]['prediction_sha256']:
            raise ValueError(f'{run_name}: prediction drift')
        key = (seed, sequence)
        if key in records:
            raise ValueError(f'duplicate identity state: {key}')
        records[key] = dict(seed=seed, sequence=sequence, run_name=run_name,
                            output=str(output), prediction=str(prediction),
                            replay_manifest_sha256=sha256(manifest_path),
                            prediction_sha256=sha256(prediction),
                            predictor_checkpoint_sha256=manifest['checkpoint_sha256'],
                            producer_excluded_videos=list(manifest['producer_excluded_videos']))
    expected = {(seed, seq) for seed in (1, 2) for seq in SEQUENCES}
    if set(records) != expected:
        raise ValueError(f'identity baseline coverage mismatch: {sorted(set(records) ^ expected)}')
    return records


def _build_baseline(seed, records, states_sha, out):
    aggregate = PHASE6 / f'identity_baseline_seed{seed}_outer_s0_v1'
    if aggregate.exists():
        raise ValueError(f'immutable aggregate already exists: {aggregate}')
    aggregate.mkdir()
    report = dict(status='RUNNING', family=f'identity_baseline_seed{seed}', seed=seed,
                  source_state_summary_sha256=states_sha, videos={},
                  scope='Disabled candidate/risk K1 state replay in the matching identity RGB and ordinary-producer coordinate; not seed-0.')
    for sequence in SEQUENCES:
        rec = records[(seed, sequence)]
        target = aggregate / f'{sequence}.txt'
        target.symlink_to(Path(rec['prediction']).resolve())
        if sha256(target) != rec['prediction_sha256']:
            raise ValueError(f'aggregate symlink drift: {sequence}')
        report['videos'][sequence] = dict(source=rec['prediction'], prediction_sha256=rec['prediction_sha256'],
                                          replay_manifest_sha256=rec['replay_manifest_sha256'],
                                          predictor_checkpoint_sha256=rec['predictor_checkpoint_sha256'],
                                          producer_excluded_videos=rec['producer_excluded_videos'])
    report['status'] = 'COMPLETE'
    write_json(aggregate / 'manifest.json', report)
    return aggregate, report


def _evaluate(aggregate, output):
    if output.exists():
        raise ValueError(f'immutable evaluation already exists: {output}')
    command = [sys.executable, 'tools/cfp_evaluate.py', '--data-root',
               '/data3/liuyeqiang/datasets/mft26-ir-causal-v1', '--derived-gt-root',
               '/data3/liuyeqiang/IR-MFT_assets/cfp/protocol_v2/gt',
               '--protocol', 'configs/cfp/protocol_v2.json', '--partition', 'train',
               '--predictions', str(aggregate), '--output', str(output),
               '--trackeval-root', '/data3/liuyeqiang/CFP-FishMOT_references/TrackEval']
    log = output.with_suffix('.log')
    output.parent.mkdir(parents=True, exist_ok=True)
    with log.open('x') as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                       env=dict(os.environ, PYTHONPATH='.', OPENBLAS_NUM_THREADS='1'))
    result = _read_metrics(output / 'metrics.json', str(output))
    if set(result['sequences']) != set(SEQUENCES):
        raise ValueError('identity baseline evaluation coverage mismatch')
    for sequence, artifact in result['artifacts'].items():
        if sha256(aggregate / f'{sequence}.txt') != artifact['prediction_sha256']:
            raise ValueError(f'evaluation prediction drift: {sequence}')
    return dict(metrics=result['metrics'], evaluation_sha256=sha256(output / 'metrics.json'),
                metric_sha256=result['metric_sha256'], aggregate_manifest_sha256=sha256(aggregate / 'manifest.json'),
                aggregate_command=['identity-state-symlink-aggregate', str(aggregate)],
                evaluation_command=command)


def _verify_replay_official(path):
    report = _completed_json(path, 'identity replay official summary')
    for family, entry in report['families'].items():
        evaluation = _command_output(entry['evaluation_command']) / 'metrics.json'
        if sha256(evaluation) != entry['evaluation_sha256']:
            raise ValueError(f'{family}: evaluation drift')
        record = _read_metrics(evaluation, family)
        if record['metrics'] != entry['metrics']:
            raise ValueError(f'{family}: metric payload drift')
        aggregate = _command_output(entry['aggregate_command'])
        if sha256(aggregate / 'manifest.json') != entry['aggregate_manifest_sha256']:
            raise ValueError(f'{family}: aggregate drift')
        for sequence, artifact in record['artifacts'].items():
            if sha256(aggregate / f'{sequence}.txt') != artifact['prediction_sha256']:
                raise ValueError(f'{family}: prediction drift on {sequence}')
    return report


def _statistics(vectors):
    import numpy as np
    return {field: dict(mean=float(np.mean([v[field] for v in vectors])),
                        std_population=float(np.std([v[field] for v in vectors]))) for field in FIELDS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--states-summary', type=Path, default=PHASE6 / 'identity_states_v1.json')
    parser.add_argument('--replays-summary', type=Path, default=PHASE6 / 'identity_replays_v1.json')
    parser.add_argument('--official-summary', type=Path, default=PHASE6 / 'identity_official_v1.json')
    parser.add_argument('--output', type=Path, default=PHASE6 / 'identity_extended_official_v1.json')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('immutable evidence summary already exists')
    states = _completed_json(args.states_summary, 'identity states')
    replays = _completed_json(args.replays_summary, 'identity replays')
    official = _verify_replay_official(args.official_summary)
    records = _state_records(states)
    families = dict(official['families'])
    baseline_records = {}
    for seed in (1, 2):
        aggregate, aggregate_manifest = _build_baseline(seed, records, sha256(args.states_summary), args.output)
        family = f'identity_baseline_seed{seed}'
        evaluation_name = aggregate.name + '_evaluation'
        evaluation = aggregate.with_name(evaluation_name)
        baseline_records[family] = _evaluate(aggregate, evaluation)
        baseline_records[family]['seed'] = seed
        baseline_records[family]['aggregate_manifest_sha256'] = sha256(aggregate / 'manifest.json')
        baseline_records[family]['state_summary_sha256'] = sha256(args.states_summary)
    families.update(baseline_records)
    report = dict(status='COMPLETE', plan_sha256=replays['plan_sha256'],
                  replay_official_summary_sha256=sha256(args.official_summary),
                  state_summary_sha256=sha256(args.states_summary), families=families,
                  identity_baseline_records=baseline_records, comparisons={})
    for seed in (1, 2):
        base = families[f'identity_baseline_seed{seed}']['metrics']
        for family in ('ordinary_B_identity', 'predictive_B_identity', 'margin_only_identity', 'predictive_margin_identity'):
            key = f'{family}_seed{seed}'
            metrics = families[key]['metrics']
            report['comparisons'][key] = {
                'seed': seed, 'baseline_family': f'identity_baseline_seed{seed}',
                'delta_combined': {field: float(metrics['COMBINED_SEQ'][field] - base['COMBINED_SEQ'][field]) for field in FIELDS},
                'delta_per_video': {seq: {field: float(metrics[seq][field] - base[seq][field]) for field in FIELDS}
                                    for seq in SEQUENCES},
            }
    for family in ('identity_baseline', 'ordinary_B_identity', 'predictive_B_identity',
                   'margin_only_identity', 'predictive_margin_identity'):
        keys = [f'{family}_seed{seed}' for seed in (1, 2)]
        report.setdefault('seed_statistics', {})[family] = {
            'combined': _statistics([families[k]['metrics']['COMBINED_SEQ'] for k in keys]),
            'per_video': {seq: _statistics([families[k]['metrics'][seq] for k in keys]) for seq in SEQUENCES},
        }
    write_json(args.output, report)
    print(json.dumps({'status': report['status'], 'families': sorted(families),
                      'output_sha256': sha256(args.output)}, indent=2))


if __name__ == '__main__':
    main()
