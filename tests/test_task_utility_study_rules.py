"""Artificial complete curves only; never actual model/metric qualification."""
import copy

import pytest

from tools.task_utility_study_rules import (VARIANTS,CUES,NORMAL,INTERVENTIONS,valid_dev,
    complete_curves,complete_interventions,best_dev,budget_decision,select_models,test_plan as frozen_test_plan)


def row(variant,cue,epoch,intervention='normal',hota=40.):
    return dict(split='dev',sequences=['BT-005','SN-013'],frames=4834,seed=0,variant=variant,cue=cue,
        epoch=epoch,intervention=intervention,selection_metrics=dict(HOTA=hota,AssA=35.,IDF1=45.),artificial=True)


def curves(through=6):
    rows=[row(v,c,e) for v,c in NORMAL for e in range(1,through+1)]
    rows.extend(row(v,'ir',e,i) for v in VARIANTS for e in range(1,through+1) for i in INTERVENTIONS)
    return rows


def test_test_partial_seed_and_invalid_metrics_cannot_enter_selection():
    reference=row(VARIANTS[0],'ir',1)
    changes=[dict(split='test'),dict(frames=4833),dict(sequences=['BT-005']),dict(seed=1),
        dict(seed=False),dict(epoch=True),dict(variant='unregistered'),
        dict(selection_metrics=dict(HOTA=float('nan'),AssA=35.,IDF1=45.)),dict(intervention='wrong_label')]
    for change in changes:
        with pytest.raises(ValueError):valid_dev(dict(reference,**change))
    with pytest.raises(ValueError):valid_dev(row(VARIANTS[0],'rgb',1,'unavailable'))
    rows=curves();rows.append(dict(reference,split='test',epoch=12))
    with pytest.raises(ValueError):budget_decision(rows,6)


def test_all_twelve_curves_include_negative_models_and_require_every_epoch():
    rows=curves();rows[0]['selection_metrics']['HOTA']=0
    assert len(complete_curves(rows,6))==72
    with pytest.raises(ValueError,match='twelve'):complete_curves(rows[1:],6)
    with pytest.raises(ValueError,match='competing'):complete_curves(rows+[copy.deepcopy(rows[0])],6)


def test_every_IR_snapshot_needs_missing_and_actual_image_roll():
    rows=curves();assert len(complete_interventions(rows,6))==48
    with pytest.raises(ValueError,match='Every IR'):select_models(rows[:-1],6)
    with pytest.raises(ValueError,match='common stage'):select_models(curves(3),3)


def test_exact_point_two_does_not_extend_but_one_RGB_improvement_extends_all():
    rows=curves();chosen=next(r for r in rows if (r['variant'],r['cue'],r['epoch'],r['intervention'])==(VARIANTS[0],'rgb',6,'normal'))
    chosen['selection_metrics']['HOTA']=40.2
    assert not budget_decision(rows,6)['extend_all']
    chosen['selection_metrics']['HOTA']=40.20001
    decision=budget_decision(rows,6)
    assert decision['extend_all'] and decision['through_epoch']==9
    assert decision['improved_conditions']==[dict(variant=VARIANTS[0],cue='rgb',absolute_HOTA_gain='0.20001')]
    assert decision['learning_rate_horizon_complete_Train_epochs']==12


def test_future_Dev_cannot_change_earlier_budget_and_maximum_stops_even_with_gain():
    rows=curves();rows.append(row(VARIANTS[0],'ir',9,hota=100.))
    assert not budget_decision(rows,6)['extend_all']
    all_rows=curves(12)
    next(r for r in all_rows if r['epoch']==12 and r['intervention']=='normal')['selection_metrics']['HOTA']=100.
    assert budget_decision(all_rows,12)==dict(stage=12,extend_all=False,through_epoch=12,reason='fixed_maximum_complete_Train_budget')


def test_earlier_epoch_then_variant_tie_break_and_intervention_rejection():
    first=row(VARIANTS[3],'ir',1);later=row(VARIANTS[0],'ir',2)
    assert best_dev([later,first])==first
    same=row(VARIANTS[0],'ir',1);assert best_dev([first,same])==same
    with pytest.raises(ValueError,match='Interventions'):best_dev([dict(first,intervention='unavailable')])


def test_all_negative_IRs_still_get_full_Test_and_strong_controls_use_Dev():
    rows=curves()
    for r in rows:
        if r['intervention']=='normal':
            r['selection_metrics']['HOTA']=20.+VARIANTS.index(r['variant']) if r['cue']=='ir' else 50.
    next(r for r in rows if (r['variant'],r['cue'],r['epoch'],r['intervention'])==(VARIANTS[0],'rgb',2,'normal'))['selection_metrics']['HOTA']=70.
    selection=select_models(rows,6);assert selection['primary']['variant']==VARIANTS[3]
    assert selection['primary']['selection_metrics']['HOTA']==23.
    assert selection['matched_RGB']['variant']==VARIANTS[3]
    assert selection['strongest_Dev_RGB']['variant']==VARIANTS[0] and selection['strongest_Dev_RGB']['epoch']==2
    cases=frozen_test_plan(selection);assert len(cases)==26
    assert sum(c['kind']=='normal' for c in cases)==12
    assert sum(c['kind']=='input_intervention' for c in cases)==8
    assert sum(c['kind']=='inference_gate_removal' for c in cases)==3
    assert sum(c['kind']=='inference_only_fixed_geometry' for c in cases)==1
    assert all(c['frames']==15985 and len(c['sequences'])==7 and c['complete_Test_required'] for c in cases)
    assert {c['model']['variant'] for c in cases if c['kind']=='normal' and c['model']['cue']=='ir'}==set(VARIANTS)


def test_forged_primary_control_or_truncated_model_set_rejected():
    selection=select_models(curves(),6)
    bad=copy.deepcopy(selection);bad['primary']=bad['matched_RGB']
    with pytest.raises(ValueError):frozen_test_plan(bad)
    bad=copy.deepcopy(selection);bad['normal_models'].pop()
    with pytest.raises(ValueError):frozen_test_plan(bad)
    bad=copy.deepcopy(selection);bad['strongest_Dev_RGB']=bad['matched_gray']
    with pytest.raises(ValueError,match='controls'):frozen_test_plan(bad)
