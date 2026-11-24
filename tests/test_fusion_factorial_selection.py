import copy

import pytest

from tools.fusion_factorial_selection import identity


def condition():
    return dict(id='original',fit_id='quality_ir',checkpoint_sha256='weights',fit_config_sha256='arm',
                epoch=3,mode='quality',cue='ir',seed=0,geometry_fixed=False,intervention='normal')


def test_renamed_condition_or_weight_path_does_not_require_duplicate_inference():
    left=condition();right=dict(left,id='new_name',checkpoint='/different/serialization-path',registration='new prose')
    assert identity(left)==identity(right)


@pytest.mark.parametrize('field,value',[('fit_id','quality_rgb'),('checkpoint_sha256','different'),
    ('fit_config_sha256','different'),('epoch',4),('mode','temporal'),('cue','rgb'),('seed',1),
    ('geometry_fixed',True),('intervention','IR_image_spatial_roll')])
def test_same_case_name_or_equal_metrics_cannot_reuse_a_different_scientific_condition(field,value):
    left=condition();right=copy.deepcopy(left);right[field]=value
    left['HOTA']=right['HOTA']=40.
    assert identity(left)!=identity(right)
