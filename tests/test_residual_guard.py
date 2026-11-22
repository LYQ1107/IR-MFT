import torch

from yolox.tracker.association_calibrator import AssociationCalibrator
from yolox.tracker.guarded_rgb_tracker import ResidualGuard


def test_guard_uses_only_frozen_RGB_current_row_uncertainty_and_retains_zero_exactly():
    torch.manual_seed(4);model=AssociationCalibrator(base_dimensions=26)
    with torch.no_grad():model.net[-1].weight.normal_(std=.3)
    base=torch.rand(4,26);base[:,23:]=torch.tensor([[.95,.85,.85],[.1,-.85,.85],[.55,.1,.1],[.45,-.1,.1]])
    cue=torch.randn(4,17);reference=torch.randn(4,17);available=torch.ones(4,dtype=torch.bool)
    guard=ResidualGuard(model,uncertain_only=True)
    delta=guard.increment(base,cue,reference,available)
    assert torch.all(delta[:2]==0)
    torch.testing.assert_close(delta[2:],model.increment(base,cue,reference,available)[2:],rtol=0,atol=0)
    assert torch.all(ResidualGuard(model,scale=0).increment(base,cue,reference,available)==0)
    torch.testing.assert_close(ResidualGuard(model,scale=.25).increment(base,cue,reference,available),model.increment(base,cue,reference,available)*.25,rtol=0,atol=0)
