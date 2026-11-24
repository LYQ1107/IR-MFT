"""Dev-only common-budget extension and fair strong-control selection."""
import copy

import pytest

from tools.native_pyramid_study_rules import NORMAL, budget_decision, select_models


def curves(stage=6):
    return [dict(mode=m, cue=c, epoch=e, split='dev', sequences=['BT-005', 'SN-013'],
        frames=4834, seed=0, intervention='normal', metrics=dict(HOTA=40., AssA=35., IDF1=45.))
        for m, c in NORMAL for e in range(1, stage+1)]


def test_one_late_rgb_control_improvement_extends_every_fit_not_a_subset():
    rows = curves()
    rows[4]['metrics']['HOTA'] = 40.25  # quality/RGB epoch5
    result = budget_decision(rows, 6, True)
    assert result['extend_all_six'] and result['next_stage'] == 9
    assert select_models(rows, 6, 39.)['primary']['cue'] == 'ir'


def test_small_improvement_does_not_extend_or_claim_convergence():
    rows = curves()
    rows[4]['metrics']['HOTA'] = 40.2
    result = budget_decision(rows, 6, True)
    assert not result['extend_all_six'] and not result['numerical_convergence_claim']
    with pytest.raises(ValueError):
        budget_decision(rows, 6, False)
    with pytest.raises(ValueError):
        budget_decision(rows, 3, True)


def test_maximum_budget_stops_even_when_last_epoch_improves():
    rows = curves(12)
    rows[-1]['metrics']['HOTA'] = 55.
    result = budget_decision(rows, 12, True)
    assert not result['extend_all_six'] and result['next_stage'] is None


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'test', 'partial', 'seed', 'nan'])
def test_incomplete_or_test_curves_cannot_select_or_extend(corruption):
    rows = curves()
    if corruption == 'missing': rows.pop()
    elif corruption == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    elif corruption == 'test': rows[0]['split'] = 'test'
    elif corruption == 'partial': rows[0]['frames'] = 3000
    elif corruption == 'seed': rows[0]['seed'] = 1
    elif corruption == 'nan': rows[0]['metrics']['HOTA'] = float('nan')
    with pytest.raises(ValueError): select_models(rows, 6, 47.579)
    with pytest.raises(ValueError): budget_decision(rows, 6, True)


def test_primary_ties_and_stronger_gray_control_prevent_removal_fits():
    rows = curves()
    selection = select_models(rows, 6, 39.)
    assert selection['primary']['mode'] == 'quality' and selection['primary']['epoch'] == 1
    assert not selection['fit_registered_removals']  # IR must strictly beat controls.
    for row in rows:
        if row['mode'] == 'temporal' and row['cue'] == 'ir' and row['epoch'] == 4:
            row['metrics']['HOTA'] = 50.
        if row['mode'] == 'temporal' and row['cue'] == 'gray' and row['epoch'] == 6:
            row['metrics']['HOTA'] = 51.
    selection = select_models(rows, 6, 47.579)
    assert selection['primary']['mode'] == 'temporal' and selection['primary']['epoch'] == 4
    assert selection['primary_mode_gray_control']['epoch'] == 6
    assert not selection['fit_registered_removals']
    assert selection['all_six_still_require_full_Test']


def test_all_controls_negative_still_keep_six_formal_test_models():
    selection = select_models(curves(), 6, 47.579)
    assert len(selection['all_six_own_Dev_best']) == 6
    assert not selection['fit_registered_removals']
    assert selection['supported_innovations'] == 0 and not selection['goal_met']
