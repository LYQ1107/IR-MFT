#!/usr/bin/env python3
"""Describe already frozen Test feature activations without tuning any method."""
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pseudo_ir_comparison import dump, sha256


def main():
    plan = json.loads((ROOT / 'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    protocol = json.loads((ROOT / plan['protocol']).read_text())
    state_path = work / 'structural_nocycle_test_state.json'
    state = json.loads(state_path.read_text())
    assert state['status'] == 'COMPLETE' and len(state['jobs']) == 14
    assert all(v['status'] == 'COMPLETE' for v in state['jobs'].values())
    report_path = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261123_STRUCTURAL_NOCYCLE_TEST.json'
    report = json.loads(report_path.read_text())
    groups, provenance, models, per_sequence = defaultdict(list), {}, {}, {}
    observed_names = None
    fields = ['available_fraction', 'teacher_risk_eligible_fraction', 'mean_teacher_risk_factor',
              'nonzero_feature_correction_fraction', 'mean_feature_correction_in_RGB_RMS']
    for seq in protocol['historical_test']:
        path = work / 'nocycle_formal_test/structural_relative_predictions/test' / seq / 'complete.json'
        meta = json.loads(path.read_text())
        assert meta['frames'] == state['producers'][seq]['source_proof']['unavailable_raw_exact_to_fresh']['frames']
        assert not meta['GT_read'] and not meta['future_frame_input']
        observed = {name: info for name, info in meta['models'].items() if info.get('gate_statistics')}
        if observed_names is None:
            observed_names = set(observed)
        assert set(observed) == observed_names
        for name, info in observed.items():
            identity = {k: info[k] for k in ['cue', 'fit_variant', 'checkpoint_sha256']}
            if name in models:
                assert models[name] == identity
            models[name] = identity
            stats = info['gate_statistics']
            assert len(stats) == 6
            for row in stats:
                assert row['frames'] == meta['frames']
                groups[(name, row['level'], row['branch'])].append(row)
        per_sequence[seq] = observed['NT_ir_no_cycle_s0_e06']['gate_statistics']
        provenance[seq] = dict(complete_manifest_sha256=sha256(path), frames=meta['frames'])
    aggregate = []
    for (name, level, branch), rows in sorted(groups.items()):
        frames = sum(r['frames'] for r in rows)
        assert len(rows) == 7 and frames == 15985
        total = sum(r['positions'] for r in rows)
        values = {key: sum(r[key] * r['positions'] for r in rows) / total for key in fields}
        values.update(maximum_feature_correction_in_RGB_RMS=max(r['maximum_feature_correction_in_RGB_RMS'] for r in rows))
        if name.endswith('_unavailable'):
            assert all(values[k] == 0 for k in fields)
        aggregate.append(dict(model=name, level=level, branch=branch, frames=frames,
                              positions=total, **values))
    result = dict(status='COMPLETE_FROZEN_ALL7_TEST_ACTIVATION_DESCRIPTION',
        source_sha256=sha256(__file__), completed_Test_state_sha256=sha256(state_path),
        complete_Test_report_sha256=sha256(report_path), model_provenance=models,
        sequence_provenance=provenance, aggregate=aggregate,
        primary_per_sequence=per_sequence, Test_accuracy_read_only_after_all14_complete=True,
        observer_scope=meta['gate_statistics_scope'],
        interpretation='All spatial positions, heavily dominated by background; these are not foreground-only '
            'gate estimates or correct-correspondence labels. Large eligibility or activation does not prove '
            'beneficial IR use. Current and null branches are reported separately, without summing them as gain. '
            'No Test-based method, threshold, checkpoint or primary reselection.',
        goal_met=False, supported_innovations=0)
    stem = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261123_STRUCTURAL_NOCYCLE_TEST_ACTIVATIONS'
    dump(stem.with_suffix('.json'), result)
    lines = ['# Frozen Test feature activation statistics', '', result['interpretation'], '',
        '| Model | Level | Branch | Teacher eligible fraction | Mean teacher factor | Actual changed fraction | Mean correction / RGB RMS |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for row in aggregate:
        lines.append('| ' + row['model'] + ' | ' + str(row['level']) + ' | ' + row['branch'] + ' | ' +
            ' | '.join(f'{row[key]:.6f}' for key in fields[1:]) + ' |')
    lines += ['', result['observer_scope'], '', 'This report does not change the negative complete Test result or count as an innovation.']
    stem.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=result['status'], observed_models=len(models), aggregate_rows=len(aggregate))))


if __name__ == '__main__':
    main()
