import pytest
from tools.summarize_structural_factorial import interactions


def test_full_component_effect_includes_interactions_instead_of_adding_removals():
    values={}
    for s in [0,1]:
        for c in [0,1]:
            for r in [0,1]:
                values[f'{s}{c}{r}']=10+2*s+3*c+5*r+7*s*c+11*s*r+13*c*r+17*s*c*r
    result=interactions(values)
    assert result['single_component_effects']=={'100':2.,'010':3.,'001':5.}
    assert result['pair_interactions']=={'110':7.,'101':11.,'011':13.}
    assert result['three_component_interaction']==17
    assert result['full_minus_trained_static']==58
    assert sum(values['111']-values[k]for k in ['011','101','110'])!=58


def test_decomposition_rejects_incomplete_component_evidence():
    with pytest.raises(ValueError):interactions({'111':51.329,'000':47.579})
