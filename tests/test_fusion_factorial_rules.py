import copy
import json
from pathlib import Path

import pytest

from tools.fusion_factorial_rules import CELLS, effects, module_kwargs, necessary_numeric_evidence, select_complete_dev


def curves():
    return [dict(cell=c, epoch=e, split='dev', sequences=['BT-005', 'SN-013'], frames=4834,
        seed=0, cue='ir', mode='quality', intervention='normal', executed_fit_budget=6,
        complete_Train_frames_per_epoch=27251, source_RGB_sha256='source',
        initial_native_sha256='initial', Train_schedule_sha256='schedule',
        metrics=dict(HOTA=40.+int(c[0])+int(c[1]), AssA=35., IDF1=45.))
        for c in CELLS for e in range(1, 7)]


def test_all_cells_keep_protection_temporal_and_capacity_configuration_fixed():
    root = Path(__file__).resolve().parents[1]
    study = json.loads((root/'configs/perceptual_native_pyramid_study_20261124.json').read_text())
    for mode in study['modes']:
        values = [module_kwargs(study, mode, c) for c in CELLS]
        assert all(v['protect_rgb'] and v['use_memory'] and v['use_motion'] for v in values)
        assert len({str({k: v for k, v in row.items() if k not in ['spectral', 'use_quality']}) for row in values}) == 1
        assert [(v['spectral'], v['use_quality']) for v in values] == [(False,False),(True,False),(False,True),(True,True)]
    with pytest.raises(ValueError): module_kwargs(study, 'quality', 'missing')


def test_combination_gain_does_not_establish_both_modules():
    result = necessary_numeric_evidence({'00':40.,'10':41.,'01':42.,'11':43.},
                                       {'00':30.,'10':31.,'01':34.,'11':33.})
    assert result['Test']['combined'] == 3.
    assert result['positive_mechanism_effects'] == 1
    assert not result['candidate_mechanism_effects']['contrast']['necessary_four_effects_positive']
    assert result['supported_innovations'] == 0 and not result['goal_met']


def test_test_only_positive_effect_is_not_credited():
    result = necessary_numeric_evidence({'00':40.,'10':39.,'01':42.,'11':41.},
                                       {'00':30.,'10':31.,'01':32.,'11':33.})
    assert result['positive_mechanism_effects'] == 1
    assert not result['candidate_mechanism_effects']['contrast']['positive_alone_on_Dev_and_Test']


def test_interaction_is_reported_without_requiring_positive_interaction():
    result = effects({'00':40.,'10':42.,'01':43.,'11':44.})
    assert result['interaction'] == -1.
    assert result['contrast_given_quality'] == 1.
    assert result['quality_given_contrast'] == 2.
    assert necessary_numeric_evidence({'00':40.,'10':42.,'01':43.,'11':44.},
                                     {'00':30.,'10':32.,'01':33.,'11':34.})['positive_mechanism_effects'] == 2


@pytest.mark.parametrize('bad', [{'00':30.}, {'00':30.,'10':31.,'01':32.,'11':float('nan')},
                                 {'00':-1.,'10':31.,'01':32.,'11':33.}])
def test_missing_or_nonfinite_cells_are_never_treated_as_a_zero_control(bad):
    with pytest.raises(ValueError): effects(bad)


def test_paired_and_own_best_policies_preserve_the_parent_primary():
    rows = curves(); rows[4]['metrics']['HOTA'] = 50.
    result = select_complete_dev(rows, 6, 3, 'quality')
    assert result['paired_parent_primary_epoch']['00']['epoch'] == 3
    assert result['own_Dev_best']['00']['epoch'] == 5
    assert result['fixed_parent_primary_epoch'] == 3 and not result['Test_used_for_selection']
    assert len(result['own_Dev_best']) == 4


@pytest.mark.parametrize('corruption', ['missing','duplicate','test','seed','subset','initial','schedule','budget'])
def test_incomplete_or_unmatched_curves_cannot_freeze_test_choices(corruption):
    rows = curves()
    if corruption == 'missing': rows.pop()
    elif corruption == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    elif corruption == 'test': rows[0]['split'] = 'test'
    elif corruption == 'seed': rows[0]['seed'] = 1
    elif corruption == 'subset': rows[0]['complete_Train_frames_per_epoch'] = 1000
    elif corruption == 'initial': rows[0]['initial_native_sha256'] = 'different'
    elif corruption == 'schedule': rows[0]['Train_schedule_sha256'] = 'different'
    elif corruption == 'budget': rows[0]['executed_fit_budget'] = 12
    with pytest.raises(ValueError): select_complete_dev(rows, 6, 3, 'quality')
