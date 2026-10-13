#!/usr/bin/env python3
"""Collect complete official Phase VI evidence without selecting models."""
import json
from pathlib import Path

import numpy as np

from cfp.provenance import sha256, write_json


ROOT = Path('/data3/liuyeqiang/IR-MFT_assets/cfp/phase6')
DOCS = Path('docs/research')
METRICS = ('HOTA', 'AssA', 'DetA', 'IDF1', 'MOTA', 'IDSW')


def complete(name):
    path = ROOT / name
    result = json.loads(path.read_text())
    if result['status'] != 'COMPLETE':
        raise ValueError('Incomplete evidence: ' + str(path))
    return result


def official(name):
    result = complete(name)
    for entry in result['families'].values():
        command = entry['evaluation_command']
        path = Path(command[command.index('--output') + 1]) / 'metrics.json'
        if sha256(path) != entry['evaluation_sha256']:
            raise ValueError('Official evaluation drift: ' + str(path))
        evaluated = json.loads(path.read_text())
        if evaluated['metrics'] != entry['metrics']:
            raise ValueError('Metric payload drift')
        aggregate = Path(entry['aggregate_command'][-1])
        if sha256(aggregate / 'manifest.json') != entry['aggregate_manifest_sha256']:
            raise ValueError('Aggregate provenance drift')
        manifest = json.loads((aggregate / 'manifest.json').read_text())
        for seq, record in manifest['videos'].items():
            digest = sha256(aggregate / (seq + '.txt'))
            if digest != record['prediction_sha256'] or digest != evaluated['artifacts'][seq]['prediction_sha256']:
                raise ValueError('Prediction drift: ' + seq)
    return result


def statistics(entries):
    # Per-seed official combined metrics, never an average of video metrics.
    return {metric: dict(mean=float(np.mean([e[metric] for e in entries])),
                         std_population=float(np.std([e[metric] for e in entries])))
            for metric in METRICS}


def main():
    summaries = {
        'quota': 'oof_quota_official_v1.json',
        'risk_initializations': 'oof_quota_risk_init_official_v1.json',
        'relation': 'oof_relation_official_v1.json',
        'relation_extra': 'oof_relation_extra_official_v1.json',
        'residual': 'oof_residual_official_v1.json',
    }
    report = dict(status='COMPLETE_STAGE', overall_goal_status='IN_PROGRESS',
                  client_date='2026-10-13', sources={}, evidence={}, statistics={},
                  scope='Full six-video association-module heldout folds with frozen backbone exposure. '
                        'Risk repetitions change initialization/sampling only, not encoder/producer. '
                        'Historic dev references are exposed; no independent test claim.')
    for key, name in summaries.items():
        report['sources'][key] = dict(path=str(ROOT / name), sha256=sha256(ROOT / name))
        report['evidence'][key] = official(name)
    baseline = json.loads((DOCS / 'PHASE_VI_OOF_MEMORY_MEASURED.json').read_text())['runs']['oof_baseline_outer_s0_v1']['metrics']
    report['baseline_metrics'] = baseline
    for family in ('context', 'mlp'):
        seeds = [report['evidence']['quota']['families'][family]['metrics'],
                 *[report['evidence']['risk_initializations']['families'][family + '_init' + str(s)]['metrics'] for s in (1, 2)]]
        report['statistics'][family] = {seq: statistics([s[seq] for s in seeds]) for seq in baseline}
    lines = ['# Phase VI continuation: write quotas and bounded predictive corrections', '',
             'Status: IN_PROGRESS. Collected on 2026-10-13. All numbers below use complete official evaluations. '
             'Six-video OOF means association-module video exclusion; detector and ReID pretraining exposure remains. '
             'This is development evidence, not independent generalization.', '',
             '| Six-video method | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |',
             '|---|---:|---:|---:|---:|---:|---:|']

    def row(name, values):
        lines.append('| ' + name + ' | ' + ' | '.join(f'{values[k]:.4f}' for k in METRICS) + ' |')

    row('Corresponding excluded-video ordinary GRU', baseline['COMBINED_SEQ'])
    for group in ('quota', 'residual', 'relation', 'relation_extra'):
        for family, entry in sorted(report['evidence'][group]['families'].items()):
            row(group + '/' + family, entry['metrics']['COMBINED_SEQ'])
    lines += ['', '## Interpretation and next discriminating experiments', '',
              'At a fixed 10% matched-edge write budget, context HOTA across three initialization/query seeds '
              f"is {report['statistics']['context']['COMBINED_SEQ']['HOTA']['mean']:.4f}±{report['statistics']['context']['COMBINED_SEQ']['HOTA']['std_population']:.4f}; "
              'same-input MLP is '
              f"{report['statistics']['mlp']['COMBINED_SEQ']['HOTA']['mean']:.4f}±{report['statistics']['mlp']['COMBINED_SEQ']['HOTA']['std_population']:.4f}. "
              'The fixed margin rule is 33.8863. Candidate-context complexity has no demonstrated independent advantage. '
              'These repetitions keep ordinary producer and encoder seed0 fixed and must not be called three full pipeline seeds.', '',
              'Replacing ordinary geometry costs causes large IDSW. Original eligibility repairs raw means substantially. '
              'Restricting learned corrections to ±0.05 within native appearance eligibility preserves the ordinary prior. '
              'EMA and noEMA bounded models have almost identical HOTA; the EMA mechanism has no demonstrated independent benefit. '
              'The ordinary residual control is worse than the unchanged producer, so auxiliary prediction deserves further controlled investigation. '
              'IDSW is still above baseline, and a small HOTA increase is not sufficient scientific success.', '',
              'Registered continuation: 36 ordinary/noEMA/EMA fits with initialization/query seeds1/2 and fixed backbone/producer0; '
              '18 corresponding seed0 fits with cue utility disabled. All use frozen epoch8, residual scale0.05, full outer-video replay, '
              'and identical protocol. No outer-video tuning or lucky-seed selection. These distinguish prediction regularization, '
              'auxiliary utility supervision and initialization variability.', '',
              'Complete per-video metric vectors, official prediction hashes and paired statistics are in the adjacent measured JSON. '
              'Full risk deployment online evaluation is still running and its partial video evidence is not reported as a complete result.', '',
              'Validation: 182 project tests passed with the migrated dataset path. Unscoped pytest also collected an unrelated '
              'legacy ctracker tutorial and failed because skimage is absent; the project tests/ suite is the verified scope. '
              'No historical model, cache or prediction was overwritten. Large training artifacts remain local.', '']
    write_json(DOCS / 'PHASE_VI_CONTINUATION_MEASURED.json', report)
    (DOCS / 'PHASE_VI_CONTINUATION_PROGRESS.md').write_text('\n'.join(lines))
    print(json.dumps(report['statistics']['context']['COMBINED_SEQ']))
    print(json.dumps(report['statistics']['mlp']['COMBINED_SEQ']))


if __name__ == '__main__':
    main()
