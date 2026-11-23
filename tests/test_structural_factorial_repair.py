import torch
from yolox.models.structural_relative_repair import StructuralRepair
from yolox.models.structural_nocycle_refinement import NoCycleRefinement
from yolox.models.structural_factorial_repair import StructuralFactorial, FLAGS


def test_existing_full_and_three_removals_are_preserved_with_nonzero_weights():
    kwargs = dict(channels=(4,),latent=2,radius=1)
    pairs = [('111',NoCycleRefinement(refinement='no_cycle',**kwargs)),
             ('011',StructuralRepair(variant='no_structure',**kwargs)),
             ('101',NoCycleRefinement(refinement='no_cycle_no_centering',**kwargs)),
             ('110',NoCycleRefinement(refinement='no_cycle_no_risk',**kwargs))]
    rgb,aux = (torch.randn(2,4,5,6) for _ in range(2))
    score = torch.rand(2,1,5,6)
    q,k = (torch.randn(2,8,5,6) for _ in range(2))
    valid = torch.ones(2,1,5,6,dtype=torch.bool)
    for bits,original in pairs:
        new = StructuralFactorial(components=bits,**kwargs)
        torch.nn.init.normal_(new.levels[0].output.weight,std=.1)
        torch.nn.init.normal_(new.levels[0].output.bias,std=.1)
        original.load_state_dict(new.state_dict(),strict=True)
        for null in [False,True]:
            args = ([rgb],[aux],[score],q,k,valid,valid)
            torch.testing.assert_close(new(*args,null=null)[0],original(*args,null=null)[0],rtol=0,atol=0)
            assert not torch.equal(new(*args,null=null)[0],rgb)
        assert new.centered == FLAGS[bits]['centered']


def test_all_component_combinations_keep_exact_missing_input_fallback():
    rgb = torch.randn(1,4,5,6)
    for bits in FLAGS:
        model = StructuralFactorial(channels=(4,),latent=2,radius=1,components=bits)
        torch.nn.init.normal_(model.levels[0].output.weight)
        result = model([rgb],None,[torch.ones(1,1,5,6)],None,None,None,None,availability=False)
        assert torch.equal(result[0],rgb)
        assert not model.levels[0].cycle
