import copy

import pytest

from tools.spatial_native_rules import NORMAL, VARIANTS, best_dev, budget_decision, complete_curves, select_models
from tools.spatial_native_study import STUDY, computation_key, make_arm, read, release_state_ready, validate_history


def curves(stage=6):
    rows, interventions = [], []
    for variant, cue in NORMAL:
        for epoch in range(1, stage+1):
            row = dict(variant=variant, cue=cue, epoch=epoch, seed=0, split='dev',
                sequences=['BT-005', 'SN-013'], frames=4834, removal=None, intervention='normal',
                selection_precision='original276CSV_COMBINED_full_precision',
                selection_metrics=dict(HOTA=40.+epoch*.01+(cue=='ir'), AssA=30., IDF1=50.))
            rows.append(row)
            if cue == 'ir':
                interventions.extend([dict(row, intervention=i) for i in ['unavailable', 'IR_image_spatial_roll']])
    return rows, interventions


def test_requires_all_nine_complete_normal_curves_even_when_negative():
    rows, _ = curves()
    assert len(complete_curves(rows, 6)) == 54
    with pytest.raises(ValueError):
        complete_curves(rows[:-1], 6)
    with pytest.raises(ValueError):
        complete_curves(rows+[rows[0]], 6)


@pytest.mark.parametrize('field,value', [('split', 'test'), ('frames', 4833), ('seed', 1),
    ('sequences', ['BT-005']), ('selection_precision', 'rounded_headline'), ('intervention', 'unavailable')])
def test_rejects_wrong_scope_or_precision_before_selection(field, value):
    rows, _ = curves()
    rows[0][field] = value
    with pytest.raises(ValueError):
        complete_curves(rows, 6)


def test_full_precision_beats_rounded_display_and_ties_prefer_simple_variant():
    rows, _ = curves()
    candidates = [r for r in rows if r['cue'] == 'ir' and r['epoch'] == 1]
    candidates[-1]['selection_metrics'] = dict(HOTA=41.0100001, AssA=30., IDF1=50.)
    assert best_dev(candidates)['variant'] == VARIANTS[-1]
    candidates[-1]['selection_metrics']['HOTA'] = candidates[0]['selection_metrics']['HOTA']
    assert best_dev(candidates)['variant'] == VARIANTS[0]


def test_budget_waits_for_both_interventions_and_extends_all_nine():
    rows, interventions = curves()
    assert not budget_decision(rows, interventions, 6)['extend_all_nine']
    with pytest.raises(ValueError):
        budget_decision(rows, interventions[:-1], 6)
    row = next(r for r in rows if r['variant'] == VARIANTS[-1] and r['cue'] == 'gray' and r['epoch'] == 6)
    row['selection_metrics']['HOTA'] += 1.
    decision = budget_decision(rows, interventions, 6)
    assert decision['extend_all_nine'] and decision['next_stage'] == 9
    rows12, interventions12 = curves(12)
    rows12[-1]['selection_metrics']['HOTA'] = 90
    assert not budget_decision(rows12, interventions12, 12)['extend_all_nine']


def test_strong_sources_and_same_variant_controls_gate_removals_not_test_coverage():
    rows, _ = curves()
    accepted = select_models(rows, 6, 40, 40)
    assert accepted['fit_enabled_removals'] and len(accepted['all_nine_own_Dev_best']) == 9
    rejected = select_models(rows, 6, 47, 48)
    assert not rejected['fit_enabled_removals'] and rejected['all_nine_still_require_complete_Test']
    assert rejected['Test_accuracy_used'] is False


def test_semantic_reuse_is_exact_computation_not_variant_label_or_score():
    study = read(STUDY)
    spatial = make_arm(study, 'spatial_ir', 'spatial', 'ir')
    removed = make_arm(study, 'quality_spatial_noQ', 'quality_spatial', 'ir', 'no_frame_quality')
    assert computation_key(spatial) == computation_key(removed)
    removed['fit']['learning_rate'] *= 2
    assert computation_key(spatial) != computation_key(removed)
    with pytest.raises(ValueError):
        make_arm(study, 'invalid', 'spatial', 'ir', 'no_redundancy')
    with pytest.raises(ValueError):
        make_arm(study, 'invalid', 'quality_spatial', 'rgb', 'no_frame_quality')


def released_fixture():
    state = dict(status='COMPLETE_FACTORIAL_EXTENSION_NOT_TRIGGERED_BY_DEV', controller_pid=1,
        science_sha256='science', registry_sha256='registry', phases={})
    identity = dict(pid=1, start_ticks=10)
    report = dict(status='COMPLETE_CONDITIONAL_FACTORIAL_NOT_TRIGGERED',
        science_sha256='science', registry_sha256='registry')
    return state, identity, report


def test_release_requires_terminal_export_and_actual_controller_exit():
    state, identity, report = released_fixture()
    assert not release_state_ready(state, identity, report, lambda *a, **k: False)
    assert release_state_ready(state, identity, report, lambda *a, **k: True)
    with pytest.raises(ValueError):
        release_state_ready(state, identity, dict(report, registry_sha256='other'), lambda *a, **k: True)
    with pytest.raises(RuntimeError):
        release_state_ready(dict(state, status='FAILED'), identity, report)


def test_exited_controller_cannot_release_a_live_or_unverified_producer():
    state, identity, _ = released_fixture()
    state['status'] = 'COMPLETE_FACTORIAL_STUDY_REQUIRES_SCIENTIFIC_ASSESSMENT'
    report = dict(status='COMPLETE_FOUR_CELL_FACTORIAL_REQUIRES_SCIENTIFIC_ASSESSMENT',
        science_sha256='science', registry_sha256='registry')
    child = dict(pid=2, start_ticks=11, command=['python', 'worker'], status='COMPLETE', exit_code=0)
    state['phases'] = dict(fit=dict(child=child))
    assert not release_state_ready(state, identity, report, lambda pid, **kw: pid == 1)
    assert release_state_ready(state, identity, report, lambda *a, **kw: True)
    child['exit_code'] = 1
    assert not release_state_ready(state, identity, report, lambda *a, **kw: True)


def test_checkpoint_history_needs_actual_full_train_addresses_and_teacher():
    from tools.native_pyramid_train_data import schedule_proof
    study = read(STUDY)
    arm = make_arm(study, 'spatial_ir', 'spatial', 'ir')
    from tools.run_pseudo_ir_comparison import sha256
    proof = schedule_proof(study['Train_delivery']['lengths'], 1)
    saved = dict(completed_epochs=1, metadata=dict(seed=0, study_sha256=sha256(STUDY),
        source_RGB_sha256=study['source_RGB_sha256'], cue='ir', fit_id='spatial_ir', computation_key=computation_key(arm)),
        history=[dict(epoch=1, frozen_RGB_exact_unchanged=True, delivery=dict(proof,
            actual_address_sha256=proof['address_sha256'], delivered_clips=6814, complete_all_Train_frames=True))])
    assert validate_history(saved, 1, arm)['seed'] == 0
    damaged = copy.deepcopy(saved)
    damaged['history'][0]['delivery']['actual_address_sha256'] = 'wrong'
    with pytest.raises(ValueError):
        validate_history(damaged, 1, arm)
    damaged = copy.deepcopy(saved)
    damaged['metadata']['source_RGB_sha256'] = 'another_teacher'
    with pytest.raises(ValueError):
        validate_history(damaged, 1, arm)
