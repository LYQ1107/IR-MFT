#!/usr/bin/env python3
"""Freeze all required formal-candidate Test cases after complete matched Dev."""
import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.perceptual_gpu_handoff import POLICY_PATH, future_gpus


def main():
    phase_path = ROOT/'configs/perceptual_structural_nocycle_20261123.json'
    phase = json.loads(phase_path.read_text())
    gate_path = ROOT/'configs/perceptual_structural_nocycle_test_gate_20261123.json'
    gate = json.loads(gate_path.read_text())
    assert gate['status'] == 'REGISTERED_BEFORE_ANY_NEW36DEV_ACCURACY' and gate['fixed_seed'] == 0
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    docs = ROOT/'docs/fusion_research'
    report_path = docs/'PERCEPTUAL_MFT_20261123_STRUCTURAL_NOCYCLE_ANALYSIS.json'
    report = json.loads(report_path.read_text())
    state_path = work/'structural_nocycle_dev_state.json'
    state = json.loads(state_path.read_text())
    assert state['status'] == 'COMPLETE' and len(state['jobs']) == 36
    assert report['status'] == 'COMPLETE_FIXED_SEED0_STRUCTURAL_NOCYCLE_DEV'
    assert report['provenance']['complete_state_sha256'] == sha256(state_path)
    assert report['primary']['id'] == 'SR_ir_no_cycle_s0_e06_raw'
    rolled = next(r for r in report['IR_interventions'] if r['id'] == 'NC_IR_image_roll_s0_e06_raw')
    positive_groups = [r['mechanism'] for r in report['removals']
                       if min(r['primary_minus_paired_HOTA'], r['primary_minus_best_removal_HOTA']) > 0]
    eligible = (report['primary_minus_strongest_matched_control_HOTA'] > 0
                and len(positive_groups) >= 2 and rolled['HOTA_change_from_correct_IR'] < 0)
    decision = dict(status='DEV_COMPLETE_RUN_REQUIRED_FORMAL_TEST',
        Dev_supports_contribution_hypothesis=eligible, formal_Test_required=True,
        complete_Dev_analysis_sha256=sha256(report_path), complete_Dev_state_sha256=sha256(state_path),
        gate_registration_sha256=sha256(gate_path),
        fixed_seed=0, positive_best_curve_mechanism_groups=positive_groups,
        primary_minus_strongest_matched_control_HOTA=report['primary_minus_strongest_matched_control_HOTA'],
        wrong_image_IR_HOTA_change=rolled['HOTA_change_from_correct_IR'],
        eligibility='These Dev criteria assess contribution support, not Test eligibility. The user requires '
                    'every trained formal candidate passing engineering checks to run all7Test sequences. '
                    'All formal candidates are evaluated irrespective of their Dev gains.',
        primary_unchanged_from_pre_CGA14_Test_plan=True, goal_met=False, supported_innovations=0)
    decision_path = docs/'PERCEPTUAL_MFT_20261123_STRUCTURAL_NOCYCLE_TEST_DECISION.json'
    assert not decision_path.exists()
    dump(decision_path, decision)
    spec_path = ROOT/'configs/perceptual_structural_nocycle_test_20261123.json'
    folder = ROOT/'configs/perceptual_structural_nocycle_test_20261123'
    frozen_path = work/'structural_nocycle_test_selection_frozen.json'
    assert not spec_path.exists() and not folder.exists() and not frozen_path.exists()
    models, cases, aliases = {}, {}, {}
    seen = {}

    def add(label, row, intervention=None):
        c = row['config']
        variant = c.get('fit_variant', c['variant'])
        key = f"NT_{c['cue']}_{variant}_s0_e{c['epoch']:02d}"
        if intervention:
            key += '_'+intervention
        if key not in models:
            origin = 'initial_structural_relative' if '/structural_relative_fit/' in c['checkpoint'] else 'nocycle_refinement'
            models[key] = dict(id=key, cue=c['cue'], fit_variant=variant,
                variant='no_centering' if variant.endswith('no_centering') else 'no_cycle',
                seed=0, epoch=c['epoch'], checkpoint=c['checkpoint'], checkpoint_origin=origin)
            if intervention:
                models[key]['intervention'] = intervention
        policy = c['output_policy']
        identity = (key, policy)
        if identity not in seen:
            case_id = 'NT_'+label
            cases[case_id] = dict(id=case_id, model_id=key,
                proposal_key=key+('_raw' if policy == 'raw' else ''), output_policy=policy,
                cue=c['cue'], seed=0, checkpoint=c['checkpoint'])
            seen[identity] = case_id
        aliases[label] = seen[identity]
        return seen[identity]

    primary_id = add('primary_IR', report['primary'])
    original_path = docs/'PERCEPTUAL_MFT_20261123_STRUCTURAL_RELATIVE_ANALYSIS.json'
    assert sha256(original_path) == phase['source_Dev_analysis_sha256']
    original = json.loads(original_path.read_text())
    add('cycle_restored_Dev_selected', original['best_by_cue']['ir'])
    for cue in ['rgb', 'gray']:
        add('matched_'+cue, report['best_matched_controls'][cue])
    removal_roles = []
    for name, row in zip(['structure', 'centering', 'feature_risk'], report['removals']):
        paired_id = add('no_'+name+'_paired6', row['paired'])
        best_id = add('no_'+name+'_Dev_best4', row['best_removal_over4epochs_and_both_policies'])
        removal_roles.append(dict(mechanism=row['mechanism'], paired_id=paired_id, Dev_best4_id=best_id))
    for label, intervention in [('IR_image_roll', 'IR_image_spatial_roll'), ('IR_unavailable', 'unavailable')]:
        add(label, report['primary'], intervention)
    models['CG3_source_RGB'] = dict(id='CG3_source_RGB', cue='rgb', seed=0, baseline=True)
    for label, fresh in [('source_RGB_fresh', True), ('source_RGB_canonical', False)]:
        key = 'NT_'+label
        cases[key] = dict(id=key, model_id='CG3_source_RGB', proposal_key='CG3_source_RGB',
            baseline=True, fresh_RGB=fresh, cue='rgb', seed=0, output_policy='baseline')
        aliases[label] = key
    spec = dict(status='FROZEN_FROM_COMPLETE_DEV_BEFORE_ANY_THIS_FAMILY_TEST_ACCURACY',
        created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        config_sha256=phase['base_config_sha256'], refinement_plan_sha256=sha256(phase_path),
        physical_gpus=list(future_gpus()), gpu_handoff_policy_sha256=sha256(POLICY_PATH),
        single_seed_user_policy_sha256=phase['single_seed_policy_sha256'], fixed_seed=0,
        models=list(models.values()), cases=list(cases.values()), roles=aliases, removals=removal_roles,
        primary_case=primary_id, complete_Dev_analysis_sha256=sha256(report_path),
        gate_registration_sha256=sha256(gate_path),
        complete_Dev_state_sha256=sha256(state_path),
        counts=dict(comparison_roles=14, unique_accuracy_replays=len(cases), models=len(models),
                    produced_streams=2*(len(models)-1)+1),
        coverage=dict(sequences=7, frames=15985),
        primary_IR='No-cycle IR epoch6/raw, frozen before earlierCGA14Test. No bestTest checkpoint or readout.',
        formal_candidates='Both the initial full-cycle structural IR and the refined no-cycle IR are tested, '
                          'selected from complete Dev. Contribution gates cannot remove a formal candidate from Test.',
        numerical_source_protocol='Use the source-only tail audit registered in CGA recovery: matched2px/.005 '
            'bounds, exact matched0.6/0.7memberships, at most1unmatched score<.6 in final4frames. '
            'Separate fresh and canonical RGB controls, unavailable tracks/all39metrics/CSV exact to fresh.',
        previous_Test_exposure='EarlierCGA14Test was negative29.431 vs29.529; this primary was already frozen '
            'before that result. HistoricalTest remains exposed; this is exploratory research, not a fresh blind test.',
        attribution=report['attribution'], goal_met=False, supported_innovations=0)
    assert len(aliases) == 14 and 11 <= len(cases) <= 14
    dump(spec_path, spec)
    folder.mkdir()
    for c in cases.values():
        config = dict(c, fusion_plan_sha256=spec['config_sha256'], specification_sha256=sha256(spec_path),
            refinement_plan_sha256=sha256(phase_path),
            single_seed_user_policy_sha256=phase['single_seed_policy_sha256'],
            producer_directory='nocycle_formal_test/structural_relative_predictions')
        dump(folder/(c['id']+'.json'), config)
    dump(frozen_path, dict(status='COMPLETE_DEV_SELECTION_FROZEN', fixed_seed=0,
        complete_Dev_before_Test=True, complete_Dev_analysis_sha256=sha256(report_path),
        complete_Dev_state_sha256=sha256(state_path), specification_sha256=sha256(spec_path),
        refinement_plan_sha256=sha256(phase_path), primary_case=primary_id, roles=aliases))
    print(json.dumps(dict(status=spec['status'], unique_Test_cases=len(cases), comparison_roles=14, fixed_seed=0)))


if __name__ == '__main__':
    main()
