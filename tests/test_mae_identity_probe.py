import torch
from torch import nn
from cfp.mae_identity_probe import FrozenEncoderIdentityProbe,probe_identity_loss


class Backbone(nn.Module):
    def __init__(self):super().__init__();self.encoder=nn.Linear(3,384)
    def forward_features(self,x):return dict(x_norm_clstoken=self.encoder(x.mean((2,3))))


def test_strict_losses_frozen_encoder_eval_and_projection_identity_information():
    torch.set_num_threads(2);x=torch.randn(4,3,128,256);content=torch.ones(4,1,128,256);labels=torch.tensor([0,0,1,1]);frames=torch.tensor([1,26,1,26]);videos=torch.zeros(4,dtype=torch.long)
    for mode in ['ce_triplet','ce_only','triplet_only']:
        m=FrozenEncoderIdentityProbe(Backbone(),2,mode).train();out=m(x,content);assert not m.backbone.training;assert torch.allclose(out['embedding'],out['raw_global'],atol=1e-6)
        loss,parts=probe_identity_loss(out,labels,frames,videos,mode);loss.backward();assert all(p.grad is None for p in m.backbone.parameters());assert m.projection.weight.grad is not None
        if mode=='ce_only':assert parts['triplet']==0
        if mode=='triplet_only':assert parts['ce']==0 and all(p.grad is None for p in m.classifier.parameters())
