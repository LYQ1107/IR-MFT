import torch

from yolox.tracker.association_calibrator import AssociationCalibrator


def test_initialization_and_missing_cue_preserve_RGB_odds_exactly():
    torch.manual_seed(4);base=torch.rand(5,23);cue=torch.randn(5,17);ref=torch.randn(5,17)
    model=AssociationCalibrator().eval();available=torch.ones(5,dtype=torch.bool)
    expected=model.RGB_offset(base)
    torch.testing.assert_close(model(base,cue,ref,available),expected,rtol=0,atol=0)
    with torch.no_grad():model.net[-1].weight.normal_(std=.2)
    torch.testing.assert_close(model(base,cue,ref,~available),expected,rtol=0,atol=0)
    torch.testing.assert_close(model(base,ref,ref,available),expected,rtol=0,atol=0)


def test_signed_geometry_increment_is_bounded_and_has_no_other_sample_context():
    torch.manual_seed(7);model=AssociationCalibrator().eval()
    with torch.no_grad():model.net[-1].weight.normal_(std=2.)
    base=torch.rand(4,23);cue=torch.randn(4,17);ref=torch.randn(4,17);available=torch.ones(4,dtype=torch.bool)
    change=model.increment(base,cue,ref,available)
    assert (change.abs()<=4).all()
    torch.testing.assert_close(model(base,cue,ref,available)[0],model(base[:1],cue[:1],ref[:1],available[:1])[0],rtol=1e-5,atol=1e-6)
