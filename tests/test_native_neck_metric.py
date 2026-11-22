import torch

from yolox.tracker.native_neck_metric import native_neck_forward


class Example(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=torch.nn.Identity()
        self.projection=torch.nn.Identity()
        self.neck=torch.nn.BatchNorm1d(3)
        self.classifier=torch.nn.Linear(3,2)


def test_metric_gradient_and_identity_use_the_actual_post_neck_space():
    model=Example().eval()
    with torch.no_grad():
        model.neck.running_mean.copy_(torch.tensor([10.,-4.,2.]))
        model.neck.running_var.copy_(torch.tensor([.1,4.,2.]))
    value=torch.tensor([[10.,-3.,3.],[11.,-5.,2.]],requires_grad=True)
    identity,logits=native_neck_forward(model,value)
    expected=torch.nn.functional.normalize(model.neck(value),dim=-1)
    torch.testing.assert_close(identity,expected)
    assert not torch.allclose(identity,torch.nn.functional.normalize(value,dim=-1))
    (identity[0]@identity[1]).backward()
    assert torch.isfinite(value.grad).all() and value.grad.abs().max()>0
