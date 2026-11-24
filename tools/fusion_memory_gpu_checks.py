"""Analytic ROI geometry on the explicitly supplied device; no authorization."""
import torch

from tools.fusion_memory_features import pool_p3
from tools.fusion_memory_parent_io import require


@torch.no_grad()
def known_plane_ROI(device):
    x=((torch.arange(12,device=device,dtype=torch.float32)+.5)*8)[None,:].expand(8,12)
    y=((torch.arange(8,device=device,dtype=torch.float32)+.5)*8)[:,None].expand(8,12)
    feature=torch.stack([x,y])[None]
    pooled,valid=pool_p3(feature,[[4.,2.,20.,10.]],(32,48),(64,96),2.,normalize=False)
    expected=torch.tensor([[24.,12.]],device=device)
    require(bool(valid.all()) and torch.equal(pooled,expected),'Actual ROI source/network geometry failed known-plane check')
    return dict(actual_CUDA=feature.is_cuda,stride=8,source_hw=[32,48],input_hw=[64,96],
        resize_ratio=2.,expected_raw_ROI_average=[24.,12.],actual_raw_ROI_average=pooled[0].cpu().tolist(),
        exact=True,authorization_established_here=False)
