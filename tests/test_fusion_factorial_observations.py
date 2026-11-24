import copy

import pytest

from tools.evaluate_fusion_factorial import FRESH,validate_observation_contract


def row(cell,epoch):
    return dict(id=f'FQ_{cell}_e{epoch:02d}',cell=cell,epoch=epoch,mode='quality',cue='ir',seed=0,
                geometry_fixed=False,intervention='normal')


def dev():
    return dict(status='FROZEN_FACTORIAL_OBSERVATIONS',split='dev',models=[dict(FRESH)]+[row('00',e) for e in range(1,7)])


def test_complete_new_dev_curve_is_required_without_recomputing_old_three_cells():
    value=dev();auth=dict(mode='quality',stage=6)
    assert len(validate_observation_contract(value,'dev',auth))==6
    value['models'].pop()
    with pytest.raises(ValueError):validate_observation_contract(value,'dev',auth)


@pytest.mark.parametrize('damage',['seed','rgb','gray','geometry','mode','epoch','wrong_cell','duplicate','missing_source','changed_source'])
def test_wrong_modality_or_budget_cannot_masquerade_as_the_missing_joint_cell(damage):
    value=dev();target=value['models'][1]
    if damage=='seed':target['seed']=1
    elif damage in ['rgb','gray']:target['cue']=damage
    elif damage=='geometry':target['geometry_fixed']=True
    elif damage=='mode':target['mode']='temporal'
    elif damage=='epoch':target['epoch']=7
    elif damage=='wrong_cell':target['cell']='10'
    elif damage=='duplicate':value['models'][2]=copy.deepcopy(target)
    elif damage=='missing_source':value['models'].pop(0)
    elif damage=='changed_source':value['models'][0]['seed']=1
    with pytest.raises(ValueError):validate_observation_contract(value,'dev',dict(mode='quality',stage=6))


def test_test_observations_require_full_dev_and_exact_frozen_checkpoint_conditions():
    selected=row('00',3);value=dict(status='FROZEN_FACTORIAL_OBSERVATIONS',split='test',models=[dict(FRESH),selected])
    auth=dict(mode='quality',stage=6)
    freeze=dict(status='FROZEN_FOUR_FACTORIAL_CELLS_FROM_COMPLETE_DEV',complete_all_four_Dev=True,
                Test_accuracy_used_for_selection=False,selected_unique_models=[copy.deepcopy(selected)])
    with pytest.raises(ValueError):validate_observation_contract(value,'test',auth)
    assert validate_observation_contract(value,'test',auth,freeze)==[selected]
    value['models'][1]['epoch']=4
    with pytest.raises(ValueError):validate_observation_contract(value,'test',auth,freeze)


@pytest.mark.parametrize('damage',['incomplete','test_selected','status','empty'])
def test_frozen_flag_does_not_replace_the_required_dev_evidence(damage):
    selected=row('00',3);value=dict(status='FROZEN_FACTORIAL_OBSERVATIONS',split='test',models=[dict(FRESH),selected])
    freeze=dict(status='FROZEN_FOUR_FACTORIAL_CELLS_FROM_COMPLETE_DEV',complete_all_four_Dev=True,
                Test_accuracy_used_for_selection=False,selected_unique_models=[selected])
    if damage=='incomplete':freeze['complete_all_four_Dev']=False
    elif damage=='test_selected':freeze['Test_accuracy_used_for_selection']=True
    elif damage=='status':freeze['status']='PENDING'
    elif damage=='empty':value['models']=[dict(FRESH)]
    with pytest.raises(ValueError):validate_observation_contract(value,'test',dict(mode='quality',stage=6),freeze)
