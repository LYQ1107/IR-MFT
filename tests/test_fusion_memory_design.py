"""Synthetic policy records only: these numbers are not experiment results."""
import copy
import hashlib

import pytest

from tools.fusion_memory_design import (ARCHITECTURES, INTERVENTIONS, PROFILES,
    extra_cases, freeze_case_plan, normal_case_plan, parent_details, profile_cases,
    select_memory, select_visual_parent)


def measures(value):
    return dict(HOTA=float(value), AssA=float(value), IDF1=float(value))


def visual(family, architecture, cue, value, *, baseline=False):
    identifier = family+'_'+architecture+'_'+cue
    return dict(id=identifier, family=family, architecture=architecture, cue=cue,
        epoch=0 if baseline else 6, baseline=baseline,
        checkpoint_sha256=hashlib.sha256(identifier.encode()).hexdigest(),
        split='dev', sequences=['BT-005', 'SN-013'], frames=4834, seed=0,
        intervention='normal', metrics=measures(value))


def parent(family):
    rows = []
    for i, architecture in enumerate(ARCHITECTURES[family]):
        for cue, value in [('ir', 40 if i == 0 else 39-i), ('rgb', 20+i), ('gray', 19+i)]:
            rows.append(visual(family, architecture, cue, value))
    primary = next(r for r in rows if r['cue'] == 'ir')
    fresh = visual('direct', 'fresh', 'rgb', 11, baseline=True)
    canonical = visual('direct', 'canonical', 'rgb', 10, baseline=True)
    interventions = {}
    for kind in INTERVENTIONS:
        row = copy.deepcopy(primary)
        row.update(id=primary['id']+'_'+kind, source_primary_id=primary['id'], intervention=kind,
                   metrics=copy.deepcopy(fresh['metrics']) if kind == 'IR_unavailable' else measures(5))
        interventions[kind] = row
    return dict(family=family, primary_id=primary['id'], original_primary_frozen=True,
        normal_curves_complete=True, interventions_complete=True, normal_best=rows,
        canonical_RGB=canonical, fresh_RGB=fresh, interventions=interventions)


def plan():
    return normal_case_plan(select_visual_parent([parent('native'), parent('spatial')]))


def score_row(case, metrics):
    return dict(id=case['id'], split='dev', sequences=['BT-005', 'SN-013'], frames=4834,
        seed=0, intervention=case['intervention'], metrics=copy.deepcopy(metrics))


def normal_scores(p):
    results = {}
    for case in p['normal_cases']:
        base = case['visual']['metrics']['HOTA']
        w = case['appearance_weight']
        offsets = {'byte': 0., 'rgb_uniform': .5 if w == .05 else 1.,
            'rgb_attention_proxy': .7 if w == .05 else .9,
            'fused_uniform': 1.5 if w == .05 else 2.,
            'fused_attention_proxy': 3. if w == .05 else 2.5}
        results[case['id']] = score_row(case, measures(base+offsets[case['profile']]))
    return results


def additional_scores(p, normal):
    selection = select_memory(p, normal)
    cases = extra_cases(p, selection)
    records = {}
    detail = next(d for d in p['parent']['details'] if d['family'] == p['parent']['chosen_family'])
    for case in cases:
        if case['appearance_weight'] == 0:
            metrics = detail['controls'][case['intervention']]['metrics']
        elif case['intervention'] == 'normal':
            equivalent = next(c for c in p['normal_cases'] if c['visual_key'] == case['visual_key'] and
                c['profile'] == case['profile'] and c['appearance_weight'] == case['appearance_weight'])
            metrics = {k: v+1 for k, v in normal[equivalent['id']]['metrics'].items()}
        else:
            metrics = measures(6)
        records[case['id']] = score_row(case, metrics)
    all_cases = p['normal_cases']+cases
    all_scores = {**normal, **records}
    fresh = p['role_visual_keys']['fresh_RGB']
    for case in cases:
        if case['intervention'] != 'IR_unavailable':
            continue
        control = next(c for c in all_cases if c['visual_key'] == fresh and c['intervention'] == 'normal' and
            c['appearance_weight'] == case['appearance_weight'] and
            (c['profile'] == 'byte' or (c['profile'] == 'rgb_uniform' and c['minimum_reliability'] == case['minimum_reliability'])))
        records[case['id']]['metrics'] = copy.deepcopy(all_scores[control['id']]['metrics'])
    return records


def test_requires_both_original_families_and_complete_matched_arms():
    with pytest.raises(ValueError):
        select_visual_parent([parent('native')])
    value = parent('native')
    value['normal_best'].pop()
    with pytest.raises(ValueError):
        select_visual_parent([value, parent('spatial')])


@pytest.mark.parametrize('flag', ['normal_curves_complete', 'original_primary_frozen', 'interventions_complete'])
def test_incomplete_parent_cannot_qualify(flag):
    value = parent('native')
    value[flag] = False
    with pytest.raises(ValueError):
        select_visual_parent([value, parent('spatial')])


def test_eligibility_uses_strong_and_wrong_missing_controls():
    for control in ['canonical_RGB', 'fresh_RGB', 'IR_image_spatial_roll']:
        values = [parent('native'), parent('spatial')]
        for value in values:
            row = value['interventions'][control] if control in INTERVENTIONS else value[control]
            row['metrics'] = measures(41)
        selected = select_visual_parent(values)
        assert selected['chosen_family'] is None
        with pytest.raises(ValueError):
            normal_case_plan(selected)


def test_parent_tie_order_and_original_primary_are_frozen():
    values = [parent('native'), parent('spatial')]
    assert select_visual_parent(values)['chosen_family'] == 'native'
    # A pure rule must not silently replace an original rounded-Dev frozen snapshot.
    alternative = next(r for r in values[0]['normal_best'] if r['cue'] == 'ir' and r['id'] != values[0]['primary_id'])
    alternative['metrics'] = measures(40.0001)
    selected = select_visual_parent(values)
    assert selected['chosen_primary_id'] == values[0]['primary_id']
    first = next(r for r in values[1]['normal_best'] if r['id'] == values[1]['primary_id'])
    first['metrics'] = measures(41)
    assert select_visual_parent(values)['chosen_family'] == 'spatial'


def test_rejects_another_checkpoint_in_intervention():
    value = parent('native')
    value['interventions']['IR_unavailable']['checkpoint_sha256'] = 'f'*64
    with pytest.raises(ValueError):
        parent_details(value)


@pytest.mark.parametrize('field,value', [('family', 'spatial'), ('architecture', 'temporal'), ('epoch', 9)])
def test_intervention_cannot_change_other_visual_metadata(field, value):
    record = parent('native')
    record['interventions']['IR_unavailable'][field] = value
    with pytest.raises(ValueError):
        parent_details(record)


def test_all48_normal_controls_and_equal_profile_weight_budgets():
    p = plan()
    assert len(p['normal_cases']) == 48 and len(p['visuals']) == 6
    assert p['model_fits_added'] == 0 and not p['formal_launch_ready']
    for key, value in p['visuals'].items():
        profiles = ('rgb_uniform',) if value['baseline'] else PROFILES
        for profile in profiles:
            assert sorted(c['appearance_weight'] for c in profile_cases(p, key, profile)) == [0., .05, .1]


def test_metadata_deduplication_preserves_actual_distinct_visuals():
    values = [parent('native'), parent('spatial')]
    for value in values:
        for row in value['normal_best']:
            if row['cue'] in ['rgb', 'gray']:
                row['metrics'] = measures(10)
    p = normal_case_plan(select_visual_parent(values))
    assert len(p['normal_cases']) == 30  # IR, RGB, gray and direct source.
    assert len(p['visuals']) == 4


@pytest.mark.parametrize('change', ['missing', 'Test', 'seed', 'frames', 'nonfinite'])
def test_normal_selection_rejects_incomplete_or_illegal_scores(change):
    p = plan()
    rows = normal_scores(p)
    key = next(iter(rows))
    if change == 'missing':
        rows.pop(key)
    elif change == 'Test':
        rows[key]['split'] = 'test'
    elif change == 'seed':
        rows[key]['seed'] = 1  # Invalid fixture input, not an additional experiment.
    elif change == 'frames':
        rows[key]['frames'] = 4833
    else:
        rows[key]['metrics']['HOTA'] = float('nan')
    with pytest.raises(ValueError):
        select_memory(p, rows)


def test_zero_must_reproduce_parent_selection_metrics():
    p = plan()
    rows = normal_scores(p)
    zero = next(c for c in p['normal_cases'] if c['profile'] == 'byte')
    rows[zero['id']]['metrics']['HOTA'] += .001
    with pytest.raises(ValueError):
        select_memory(p, rows)


def test_fused_choice_same_weight_same_q_controls_and_floor_gate():
    p = plan()
    selected = select_memory(p, normal_scores(p))
    assert selected['primary_IR']['profile'] == 'fused_attention_proxy'
    assert selected['primary_IR']['appearance_weight'] == .05
    assert selected['RGB_same_q_at_best_fused_weight']['profile'] == 'rgb_attention_proxy'
    assert selected['RGB_same_q_at_best_fused_weight']['appearance_weight'] == .05
    assert selected['fit_reliability_floor_removal']


def test_ordinary_rgb_winner_is_not_labeled_fused_memory():
    p = plan()
    rows = normal_scores(p)
    chosen = next(c for c in p['normal_cases'] if c['visual_key'] == p['role_visual_keys']['primary_IR'] and
                  c['profile'] == 'rgb_uniform' and c['appearance_weight'] == .05)
    rows[chosen['id']]['metrics'] = measures(50)
    selected = select_memory(p, rows)
    assert selected['primary_IR']['id'] == chosen['id'] and not selected['primary_uses_fused_memory']
    assert not selected['fit_reliability_floor_removal'] and selected['supported_innovations'] == 0


def test_no_memory_gain_keeps_zero_but_all_cases_still_require_test():
    p = plan()
    rows = {c['id']: score_row(c, c['visual']['metrics']) for c in p['normal_cases']}
    selected = select_memory(p, rows)
    assert selected['primary_IR']['profile'] == 'byte' and not selected['fit_reliability_floor_removal']
    frozen = freeze_case_plan(p, rows, additional_scores(p, rows))
    assert len(frozen['cases']) == 66 and frozen['all_normal_negative_cases_retained']
    assert frozen['all_cases_require_full7Test15985'] and not frozen['formal_launch_ready']


def test_complete78_case_plan_includes_matching_rgb_floor_controls():
    p = plan()
    rows = normal_scores(p)
    extra = additional_scores(p, rows)
    frozen = freeze_case_plan(p, rows, extra)
    assert len(frozen['cases']) == 78 and frozen['normal_cases'] == 48
    assert frozen['additional_cases'] == 30 and not frozen['Test_accuracy_used']
    assert all(c['score_acceptance_threshold'] == .6 for c in frozen['cases'])
    assert all(c['positive_reliability_and_valid_embedding_still_required'] for c in frozen['cases'])
    policies = frozen['reliability_floor_removal_policies']
    assert policies['paired_best_fused_weight'].endswith('_w005_no_reliability_floor')
    assert frozen['strong_control_roles_including_registered_floor_removals']['fresh_RGB'].endswith('_no_reliability_floor')


def test_missing_fallback_requires_same_floor_and_intervention_zero_source():
    p = plan()
    rows = normal_scores(p)
    extra = additional_scores(p, rows)
    cases = extra_cases(p, select_memory(p, rows))
    missing = next(c for c in cases if c['intervention'] == 'IR_unavailable' and c['minimum_reliability'] == 0.)
    fresh_normal = next(c for c in p['normal_cases'] if c['visual_key'] == p['role_visual_keys']['fresh_RGB'] and
                        c['appearance_weight'] == missing['appearance_weight'])
    extra[missing['id']]['metrics'] = rows[fresh_normal['id']]['metrics']
    with pytest.raises(ValueError, match='same weight AND reliability floor'):
        freeze_case_plan(p, rows, extra)
    extra = additional_scores(p, rows)
    zero = next(c for c in cases if c['appearance_weight'] == 0 and c['intervention'] == 'IR_image_spatial_roll')
    extra[zero['id']]['metrics'] = measures(99)
    with pytest.raises(ValueError, match='Intervention weight0'):
        freeze_case_plan(p, rows, extra)


def test_intervention_metrics_cannot_retune_normal_choice():
    p = plan()
    rows = normal_scores(p)
    extra = additional_scores(p, rows)
    altered = next(c for c in extra_cases(p, select_memory(p, rows)) if
                   c['intervention'] == 'IR_image_spatial_roll' and c['appearance_weight'] > 0)
    extra[altered['id']]['metrics'] = measures(99)
    frozen = freeze_case_plan(p, rows, extra)
    assert frozen['selection']['primary_IR'] == select_memory(p, rows)['primary_IR']


def test_partial_additional_dev_cannot_freeze():
    p = plan()
    rows = normal_scores(p)
    extra = additional_scores(p, rows)
    extra.pop(next(iter(extra)))
    with pytest.raises(ValueError):
        freeze_case_plan(p, rows, extra)
