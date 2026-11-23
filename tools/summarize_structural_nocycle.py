#!/usr/bin/env python3
"""Compare completed matched controls and clean removals at the frozen IR choice."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pseudo_ir_comparison import dump, sha256


def main():
    docs = ROOT/'docs/fusion_research'
    new_path = docs/'PERCEPTUAL_MFT_20261123_STRUCTURAL_NOCYCLE_DEV.json'
    main_path = docs/'PERCEPTUAL_MFT_20261123_STRUCTURAL_RELATIVE_MAIN_DEV.json'
    removed_path = docs/'PERCEPTUAL_MFT_20261123_STRUCTURAL_RELATIVE_REMOVALS_DEV.json'
    intervention_path = docs/'PERCEPTUAL_MFT_20261123_STRUCTURAL_RELATIVE_INTERVENTIONS_DEV.json'
    phase_path = ROOT/'configs/perceptual_structural_nocycle_20261123.json'
    phase = json.loads(phase_path.read_text())
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    state_path = Path(plan['work_root'])/'structural_nocycle_dev_state.json'
    state = json.loads(state_path.read_text())
    assert state['status'] == 'COMPLETE' and all(j['status'] == 'COMPLETE' for j in state['jobs'].values())
    new = {r['id']: r for r in json.loads(new_path.read_text())['rows']}
    original = {r['id']: r for r in json.loads(main_path.read_text())['rows']}
    removed = {r['id']: r for r in json.loads(removed_path.read_text())['rows']}
    intervened = {r['id']: r for r in json.loads(intervention_path.read_text())['rows']}
    assert len(new) == 36
    no_cycle = [r for r in removed.values() if r['config']['variant'] == 'no_cycle']
    no_structure = [r for r in removed.values() if r['config']['variant'] == 'no_structure']
    assert len(no_cycle) == len(no_structure) == 8
    primary = removed['SR_ir_no_cycle_s0_e06_raw']
    assert phase['primary_IR']['epoch'] == 6 and phase['primary_IR']['output_policy'] == 'raw'
    assert primary['metrics']['HOTA'] == phase['primary_IR']['Dev_HOTA']
    baseline = original['CG3_source_RGB']
    best_controls = {}
    for cue in ['rgb', 'gray']:
        rows = [r for r in new.values() if r['config']['cue'] == cue]
        assert len(rows) == 8
        best_controls[cue] = max(rows, key=lambda r: r['metrics']['HOTA'])
    strongest = max(best_controls.values(), key=lambda r: r['metrics']['HOTA'])
    removals = []
    for variant, candidates, paired_key in [
        ('structural_weights_validity_cost_confidence', no_structure, 'SR_ir_no_structure_s0_e06_raw'),
        ('semantic_centering', [r for r in new.values() if r['config']['fit_variant'] == 'no_cycle_no_centering'],
         'NC_ir_no_cycle_no_centering_s0_e06_raw'),
        ('teacher_feature_risk_placement', [r for r in new.values() if r['config']['fit_variant'] == 'no_cycle_no_risk'],
         'NC_ir_no_cycle_no_risk_s0_e06_raw')]:
        assert len(candidates) == 8
        paired = next(r for r in candidates if r['id'] == paired_key)
        best = max(candidates, key=lambda r: r['metrics']['HOTA'])
        removals.append(dict(mechanism=variant, paired=paired, best_removal_over4epochs_and_both_policies=best,
            primary_minus_paired_HOTA=primary['metrics']['HOTA']-paired['metrics']['HOTA'],
            primary_minus_best_removal_HOTA=primary['metrics']['HOTA']-best['metrics']['HOTA']))
    controls = []
    for suffix in ['', '_raw']:
        correct = removed['SR_ir_no_cycle_s0_e06'+suffix]
        for key in ['NC_IR_image_roll_s0_e06', 'NC_IR_unavailable_s0_e06']:
            row = new[key+suffix]
            controls.append(dict(id=row['id'], correct_parent=correct['id'], row=row,
                HOTA_change_from_correct_IR=row['metrics']['HOTA']-correct['metrics']['HOTA']))
    # Off packets have already passed exact comparison to the fresh RGB stream;
    # compare to the previous fresh off result before reusing its complete metrics.
    old_off = intervened['SR_IR_unavailable_s0_e24_raw']
    for seq in ['BT-005', 'SN-013']:
        folder = Path(plan['work_root'])/'nocycle_refinement/structural_relative_predictions/dev'/seq
        old_folder = Path(plan['work_root'])/'structural_relative_predictions/dev'/seq
        from tools.correlation_geometry_checks import compare_packets
        compare_packets(folder/'NC_IR_unavailable_s0_e06_raw.npz', old_folder/'SR_IR_unavailable_s0_e24_raw.npz')
    assert new['NC_IR_unavailable_s0_e06_raw']['metrics'] == old_off['metrics']
    retained = [baseline]+no_cycle+no_structure
    report = dict(status='COMPLETE_FIXED_SEED0_STRUCTURAL_NOCYCLE_DEV',
        primary=primary, baseline=baseline, best_matched_controls=best_controls,
        strongest_matched_control=strongest,
        primary_minus_original_RGB_HOTA=primary['metrics']['HOTA']-baseline['metrics']['HOTA'],
        primary_minus_strongest_matched_control_HOTA=primary['metrics']['HOTA']-strongest['metrics']['HOTA'],
        removals=removals, IR_interventions=controls,
        all_new_rows=list(new.values()), all_retained_reference_rows=retained,
        counts=dict(new_accuracy_replays=36, reused_reference_results=17, unique_reported_rows=53),
        provenance=dict(new_family_sha256=sha256(new_path), initial_main_sha256=sha256(main_path),
            initial_removals_sha256=sha256(removed_path), refinement_plan_sha256=sha256(phase_path),
            complete_state_sha256=sha256(state_path)),
        fixed_seed=0, additional_seed_work=False,
        attribution='Hard cycle consistency was harmful and receives no contribution credit. '
            'The structure removal jointly removes local structural weights, validity and absolute-cost confidence. '
            'The risk removal removes only (1-p)*(p<.6), retaining structural confidence and feature/logit bounds. '
            'These task-specific mechanism groups still require fair modality controls, complete Test and prior-art analysis.',
        scope='Complete4834-frame Dev; original RGB pretraining included these scenes. Primary IR epoch6/raw was frozen '
              'before new control accuracy. All4snapshots and bothpolicies retained equally; no new seed runs.',
        Test_selection=False, goal_met=False, supported_innovations=0)
    stem = docs/'PERCEPTUAL_MFT_20261123_STRUCTURAL_NOCYCLE_ANALYSIS'
    dump(stem.with_suffix('.json'), report)
    lines = ['# No-cycle structural refinement: matched controls and clean removals', '', report['scope'], '',
        '| Input | Dev choice | HOTA | DetA | AssA | IDF1 | BT HOTA | SN HOTA |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label, row in [('Original RGB', baseline), ('Matched RGB repeat', best_controls['rgb']),
                       ('Matched gray', best_controls['gray']), ('Frozen primary IR', primary)]:
        values = [row['metrics'][k] for k in ['HOTA', 'DetA', 'AssA', 'IDF1']]
        values += [row['per_sequence'][s]['HOTA'] for s in ['BT-005', 'SN-013']]
        lines.append('| '+label+' | '+row['id']+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines += ['', f"Primary IR minus original RGB: {report['primary_minus_original_RGB_HOTA']:+.3f}; "
        f"minus strongest matched control: {report['primary_minus_strongest_matched_control_HOTA']:+.3f} HOTA.", '',
        '| Removed mechanism | Primary − same6/raw | Primary − best removal curve |',
        '| --- | ---: | ---: |']
    for r in removals:
        lines.append(f"| {r['mechanism']} | {r['primary_minus_paired_HOTA']:+.3f} | {r['primary_minus_best_removal_HOTA']:+.3f} |")
    lines += ['', '| IR intervention | HOTA | Change from correct IR at same policy |',
              '| --- | ---: | ---: |']
    for r in controls:
        lines.append(f"| {r['id']} | {r['row']['metrics']['HOTA']:.3f} | {r['HOTA_change_from_correct_IR']:+.3f} |")
    lines += ['', report['attribution'], '', '36 new evaluations and17 reused references, not53 new runs. '
              'No >=+5 Test gain or2–3 supported innovation claim is established by this Dev report.']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(primary=primary['id'], HOTA=primary['metrics']['HOTA'],
        delta_strongest_control=report['primary_minus_strongest_matched_control_HOTA'],
        fixed_seed=0, Test_selection=False)))


if __name__ == '__main__':
    main()
