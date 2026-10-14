import copy
import torch
from torch import nn
from cfp.transreid_dino_jpm import DinoJointPatchReID
from cfp.masked_identity_checkpoint import delta_state, load_delta_state
from third_party.transreid_jpm.shuffle import shuffle_unit


class Mix(nn.Module):
    def __init__(self):
        super().__init__(); self.scale = nn.Parameter(torch.tensor(.01))
    def forward(self, x):
        return x + self.scale*x.mean(1, keepdim=True)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__(); self.embed_dim=384; self.num_register_tokens=0
        self.blocks=nn.ModuleList([Mix() for _ in range(12)]); self.norm=nn.LayerNorm(384)
        self.embed=nn.Linear(3,384); self.cls=nn.Parameter(torch.randn(1,1,384))
    def prepare_tokens_with_masks(self, x):
        patches=nn.functional.avg_pool2d(x,14,14).flatten(2).transpose(1,2)
        return torch.cat([self.cls.expand(len(x),-1,-1),self.embed(patches)],1)


def test_shuffle_matches_explicit_indices_and_remainder_is_declared():
    tokens=torch.arange(163).reshape(1,163,1)
    actual=shuffle_unit(tokens,5,2).flatten()
    shifted=list(range(5,163))+list(range(1,5))
    expected=torch.tensor([i for pair in zip(shifted[:81],shifted[81:]) for i in pair])
    assert torch.equal(actual,expected)
    assert len(actual[:160])==160 and len(actual[160:])==2
    odd=shuffle_unit(tokens[:,:162],5,2)
    assert odd.shape==(1,162,1)


def test_jpm_capacity_bn_eval_zero_content_and_lossless_delta():
    torch.set_num_threads(2);torch.manual_seed(0)
    backbone=Backbone();jpm=DinoJointPatchReID(copy.deepcopy(backbone),3,'jpm')
    torch.manual_seed(0);capacity=DinoJointPatchReID(copy.deepcopy(backbone),3,'cls_capacity')
    assert sum(p.numel() for p in jpm.parameters() if p.requires_grad)==sum(p.numel() for p in capacity.parameters() if p.requires_grad)
    images=torch.randn(4,3,128,256);content=torch.ones(4,1,128,256)
    jpm.train();before=jpm.necks[0].num_batches_tracked.clone();out=jpm(images,content)
    assert jpm.necks[0].num_batches_tracked==before+1 and out['embedding'].shape==(4,1920)
    base=jpm.immutable_base();loss=out['logits'][0].square().sum()+sum(x.square().mean() for x in out['raw_branches']);loss.backward()
    assert all(p.grad is None for p in jpm.backbone.blocks[:10].parameters())
    assert jpm.backbone.blocks[10].scale.grad.abs()>0 and jpm.local_branch[0].scale.grad.abs()>0
    torch.optim.SGD([p for p in jpm.parameters() if p.requires_grad],lr=.01).step()
    fresh=DinoJointPatchReID(copy.deepcopy(backbone),3,'jpm');load_delta_state(fresh,delta_state(jpm,base),fresh.immutable_base())
    assert all(torch.equal(v,fresh.state_dict()[n]) for n,v in jpm.state_dict().items())
    jpm.eval();buffers={n:v.clone() for n,v in jpm.named_buffers()}
    with torch.no_grad():
        assert torch.count_nonzero(jpm(images,content*0)['embedding'])==0
        jpm(images,content)
    assert all(torch.equal(v,buffers[n]) for n,v in jpm.named_buffers())
