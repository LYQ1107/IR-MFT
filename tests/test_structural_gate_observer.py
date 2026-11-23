import torch

from tools.structural_gate_observer import StructuralGateObserver
from yolox.models.structural_nocycle_refinement import NoCycleRefinement


def test_observer_preserves_outputs_and_handles_unavailable_samples():
    torch.manual_seed(0)
    model = NoCycleRefinement(channels=(4,),latent=2,radius=1,refinement='no_cycle')
    torch.nn.init.constant_(model.levels[0].output.weight,.2)
    rgb = (torch.randn(2,4,5,6),)
    auxiliary = (torch.randn_like(rgb[0]),)
    scores = (torch.full((2,1,5,6),.2),)
    descriptor = torch.randn(2,8,5,6)
    validity = torch.ones(2,1,5,6,dtype=torch.bool)
    available = torch.tensor([True,False])
    common = (rgb,auxiliary,scores,descriptor,descriptor,validity,validity,available)
    with torch.no_grad():
        expected = model(*common)[0]
        original_rgb = rgb[0].clone()
        observer = StructuralGateObserver(); observer.attach('IR',model)
        actual = model(*common)[0]
        model(*common,null=True)
        observer.close()
    torch.testing.assert_close(expected,actual,rtol=0,atol=0)
    torch.testing.assert_close(rgb[0],original_rgb,rtol=0,atol=0)
    torch.testing.assert_close(actual[1],rgb[0][1],rtol=0,atol=0)
    rows = observer.summary('IR')
    assert len(rows) == 2 and {r['branch'] for r in rows} == {'current','null_reference'}
    for row in rows:
        assert row['available_fraction'] == .5 and row['teacher_risk_eligible_fraction'] == .5
        assert row['frames'] == 2 and row['positions'] == 60
        assert 0 <= row['nonzero_feature_correction_fraction'] <= .5
