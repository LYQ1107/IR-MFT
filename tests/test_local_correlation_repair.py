import torch

from yolox.models.local_correlation_repair import CorrelationRepair,local_readout


def test_local_correlation_recovers_a_shifted_correspondence():
    query=torch.eye(30).T.reshape(1,30,5,6)
    auxiliary=torch.roll(query,1,dims=3)
    recovered,_=local_readout(query,auxiliary,auxiliary,radius=1,temperature=.01)
    unaligned,_=local_readout(query,auxiliary,auxiliary,radius=0,temperature=.01)
    torch.testing.assert_close(recovered[:,:,1:-1,1:-1],query[:,:,1:-1,1:-1],rtol=0,atol=1e-6)
    assert (unaligned-query).square().mean()>.01


def test_zero_initialized_repair_and_missing_cue_preserve_RGB_exactly():
    torch.manual_seed(3)
    model=CorrelationRepair(channels=(8,12,16),latent=4)
    rgb=[torch.randn(2,c,5,6)for c in [8,12,16]]
    auxiliary=[torch.randn_like(x)for x in rgb]
    for original,value in zip(rgb,model(rgb,auxiliary)):
        assert torch.equal(original,value)
    for original,value in zip(rgb,model(rgb,None,availability=False)):
        assert value is original


def test_gradients_reach_only_the_new_adapter_when_original_features_are_frozen():
    model=CorrelationRepair(channels=(8,12,16),latent=4)
    rgb=[torch.randn(2,c,5,6)for c in [8,12,16]]
    auxiliary=[torch.randn_like(x)for x in rgb]
    loss=sum(x.square().mean()for x in model(rgb,auxiliary))
    loss.backward()
    assert sum(float(level.output.weight.grad.abs().sum())for level in model.levels)>0
    assert all(x.grad is None for x in rgb+auxiliary)


def test_active_readout_is_independent_of_other_images_in_the_batch():
    torch.manual_seed(8)
    model=CorrelationRepair(channels=(8,12,16),latent=4).eval()
    with torch.no_grad():
        for layer in model.levels:
            layer.output.weight.normal_(0,.1)
    rgb=[torch.randn(2,c,5,6)for c in [8,12,16]]
    auxiliary=[torch.randn_like(x)for x in rgb]
    original=model(rgb,auxiliary)
    changed_rgb=[x.clone()for x in rgb];changed_aux=[x.clone()for x in auxiliary]
    for x in changed_rgb+changed_aux:x[1]=torch.randn_like(x[1])*100
    changed=model(changed_rgb,changed_aux)
    for first,second in zip(original,changed):
        assert torch.equal(first[0],second[0])
