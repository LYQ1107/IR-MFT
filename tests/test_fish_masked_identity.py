import torch
from torch import nn
from cfp.fish_masked_identity import MaskedPixelFishIdentity,masked_patch_identity_loss


class SmallBackbone(nn.Module):
    def __init__(self):
        super().__init__();self.patch=nn.Conv2d(3,384,14,stride=14);self.mask_token=nn.Parameter(torch.zeros(1,384));self.blocks=nn.ModuleList([nn.Identity() for _ in range(10)]+[nn.Linear(384,384),nn.Linear(384,384)]);self.norm=nn.LayerNorm(384)
    def forward_features(self,images,masks=None):
        x=self.patch(images).flatten(2).transpose(1,2)
        if masks is not None:x=torch.where(masks[:,:,None],self.mask_token[None],x)
        x=x+x.mean(1,keepdim=True)
        for block in self.blocks:x=block(x)
        x=self.norm(x);return dict(x_norm_clstoken=x.mean(1),x_norm_patchtokens=x)


def test_pre_attention_mask_discards_hidden_pixel_values_despite_global_mixing():
    torch.manual_seed(1);model=MaskedPixelFishIdentity(SmallBackbone(),2);image=torch.randn(1,3,126,252);mask=torch.zeros(1,162,dtype=torch.bool);mask[:,37]=True
    altered=image.clone();y,x=divmod(37,18);altered[:,:,y*14:(y+1)*14,x*14:(x+1)*14]+=40
    a=model.masked_prepared(image,mask);b=model.masked_prepared(altered,mask);assert torch.equal(a,b)
    assert not torch.equal(model.teacher.forward_features(image)['x_norm_patchtokens'],model.teacher.forward_features(altered)['x_norm_patchtokens'])


def test_real_mask_loss_gradients_independent_teacher_and_online_exclusion():
    torch.manual_seed(2);model=MaskedPixelFishIdentity(SmallBackbone(),2);images=torch.randn(2,3,128,256);content=torch.ones(2,1,128,256);mask=torch.zeros(2,162,dtype=torch.bool);mask[:,::3]=True
    out=model.training_observations(images,content,mask);loss,count=masked_patch_identity_loss(out);(loss+nn.functional.cross_entropy(out['identity']['logits'],torch.tensor([0,1]))).backward()
    assert count==108 and any(p.grad is not None and p.grad.abs().sum()>0 for p in model.predictor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.student.backbone.blocks[-1].parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.teacher.parameters())
    model.update_target(False);assert int(model.teacher_updates)==0
    model.update_target(True);assert int(model.teacher_updates)==1
    model.teacher.forward_features=lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('teacher online'))
    model.predictor.forward=lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('decoder online'))
    assert model.online(images,content)['embedding'].shape==(2,384)
