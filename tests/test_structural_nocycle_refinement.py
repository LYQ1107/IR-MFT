import torch

from yolox.models.structural_relative_repair import StructuralRepair
from yolox.models.structural_nocycle_refinement import NoCycleRefinement


def example():
    torch.manual_seed(0)
    rgb=[torch.randn(2,4,5,6)];aux=[torch.randn_like(rgb[0])]
    score=[torch.linspace(.1,.95,30).reshape(1,1,5,6).expand(2,-1,-1,-1)]
    descriptor=torch.rand(2,8,5,6);valid=torch.ones(2,1,5,6,dtype=torch.bool)
    return rgb,aux,score,descriptor,descriptor,valid,valid


def test_refined_main_and_centering_removal_keep_the_original_feature_operator():
    data=example();base=StructuralRepair(channels=(4,),latent=2,radius=1,variant='no_cycle')
    torch.nn.init.normal_(base.levels[0].output.weight)
    for variant in ['no_cycle','no_cycle_no_centering']:
        model=NoCycleRefinement(channels=(4,),latent=2,radius=1,refinement=variant)
        model.load_state_dict(base.state_dict(),strict=True)
        for null in [False,True]:
            assert torch.equal(model(*data,null=null)[0],base(*data,null=null)[0])
        assert model.centered==(variant=='no_cycle')


def test_risk_removal_is_zero_risk_reference_without_mutating_original_scores():
    data=example();model=NoCycleRefinement(channels=(4,),latent=2,radius=1,refinement='no_cycle_no_risk')
    torch.nn.init.normal_(model.levels[0].output.weight)
    base=StructuralRepair(channels=(4,),latent=2,radius=1,variant='no_cycle')
    base.load_state_dict(model.state_dict(),strict=True)
    scores=[x.clone()for x in data[2]];zero_data=list(data);zero_data[2]=[torch.zeros_like(x)for x in scores]
    for null in [False,True]:
        assert torch.equal(model(*data,null=null)[0],base(*zero_data,null=null)[0])
    assert all(torch.equal(x,y)for x,y in zip(data[2],scores))
    assert any(not torch.equal(x,y)for x,y in zip(model(*data),base(*data)))
