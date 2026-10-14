import torch
from torch import nn
from cfp.visible_token_fish_mae import VisibleTokenFishMAE


class Mix(nn.Module):
    def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(.01))
    def forward(self,x):return x+self.scale*x.mean(1,keepdim=True)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__();self.num_register_tokens=0;self.blocks=nn.ModuleList([Mix() for _ in range(12)]);self.norm=nn.LayerNorm(384);self.patch=nn.Conv2d(3,384,14,14);self.cls=nn.Parameter(torch.randn(1,1,384));self.mask_token=nn.Parameter(torch.zeros(1,384))
    def prepare_tokens_with_masks(self,x):return torch.cat([self.cls.expand(len(x),-1,-1),self.patch(x).flatten(2).transpose(1,2)],1)
    def forward_features(self,x):
        t=self.prepare_tokens_with_masks(x)
        for b in self.blocks:t=b(t)
        return dict(x_norm_clstoken=self.norm(t)[:,0])


def test_hidden_pixel_change_cannot_change_visible_token_prediction_but_changes_target():
    torch.set_num_threads(2);torch.manual_seed(0);m=VisibleTokenFishMAE(Backbone()).eval();x=torch.randn(2,3,126,252);q=torch.zeros(2,162,dtype=torch.bool);q[:,0]=True;y=x.clone();y[:,:,:14,:14]+=5
    with torch.no_grad():assert torch.equal(m.reconstruct_prepared(x,q),m.reconstruct_prepared(y,q))
    z=x.clone();z[:,:,14:28,14:28]+=1
    with torch.no_grad():assert not torch.equal(m.reconstruct_prepared(x,q),m.reconstruct_prepared(z,q))
    assert all(p.requires_grad for n,p in m.backbone.named_parameters() if n!='mask_token') and not m.backbone.mask_token.requires_grad


def test_ragged_visible_groups_and_visible_control_work_under_autocast_with_active_marker():
    torch.set_num_threads(2);torch.manual_seed(2);x=torch.randn(2,3,126,252);content=torch.ones(2,1,126,252);q=torch.zeros(2,162,dtype=torch.bool);q[0,:50]=True;q[1,:100]=True
    parameters=[]
    for mode in ['masked_visible_tokens','visible_target']:
        m=VisibleTokenFishMAE(Backbone(),mode).train();parameters.append(sum(p.numel() for p in m.parameters() if p.requires_grad))
        with torch.autocast('cpu',dtype=torch.bfloat16):out=m.training_observations(x,content,q)
        assert torch.isfinite(out['loss']);out['loss'].backward();assert m.mask_token.grad is not None and m.mask_token.grad.abs().sum()>0
    assert parameters[0]==parameters[1]
