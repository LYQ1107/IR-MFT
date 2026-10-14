"""Owned all-block visible-token RGB reconstruction, not official FishMAE.

Targets are current crop pixels; hidden patch embeddings are removed BEFORE
any attention. No identity classifier, EMA teacher, depth or future frames.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .dinov2_identity import resized_observations
from .fish_pixel_reconstruction import patchify


class VisibleTokenFishMAE(nn.Module):
    def __init__(self,backbone,mode='masked_visible_tokens'):
        super().__init__()
        if mode not in ('masked_visible_tokens','visible_target') or len(backbone.blocks)!=12 or backbone.num_register_tokens!=0:raise ValueError('Exact nonregister twelveblock DINO and registeredpretrain mode required')
        self.mode=mode;self.backbone=backbone.requires_grad_(True)
        # Original DINO replacement token is unused; a distinct decoder mask
        # token reconstructs query patches from only visible encoder tokens.
        self.backbone.mask_token.requires_grad_(False)
        # Fixed pretrained spatial positions avoid nondeterministic CUDA
        # bicubic-position backward; all12blocks/patch_embed/CLS/norm train.
        if hasattr(self.backbone,'pos_embed'):self.backbone.pos_embed.requires_grad_(False)
        self.projection=nn.Linear(384,192);self.mask_token=nn.Parameter(torch.zeros(1,1,192));self.position=nn.Parameter(torch.randn(1,163,192)*.02)
        self.decoder=nn.Sequential(*[nn.TransformerEncoderLayer(192,6,768,dropout=0.,batch_first=True,norm_first=True) for _ in range(2)])
        self.decoder_norm=nn.LayerNorm(192);self.pixels=nn.Linear(192,588)
        self.register_buffer('successful_updates',torch.zeros((),dtype=torch.int64))

    def reconstruct_prepared(self,prepared,queries):
        if queries.dtype!=torch.bool or queries.shape!=(len(prepared),162):raise ValueError('Current preparedgrid and declared patchqueries required')
        tokens=self.backbone.prepare_tokens_with_masks(prepared)
        encoder_queries=queries if self.mode=='masked_visible_tokens' else torch.zeros_like(queries)
        keep=torch.cat([torch.ones(len(prepared),1,dtype=torch.bool,device=prepared.device),~encoder_queries],1)
        restored=None
        for count in torch.unique(keep.sum(1)):
            rows=torch.nonzero(keep.sum(1)==count,as_tuple=True)[0]
            selected=tokens[rows][keep[rows]].reshape(len(rows),int(count),384)
            for block in self.backbone.blocks:selected=block(selected)
            encoded=self.projection(self.backbone.norm(selected))
            if restored is None:restored=self.mask_token.to(encoded.dtype).expand(len(prepared),163,192).clone()
            batch,positions=torch.nonzero(keep[rows],as_tuple=True)
            restored[rows[batch],positions]=encoded.reshape(-1,192)
        # In the visible-target control the same query-token parameter is an
        # additive query marker, retaining its active capacity while pixels
        # remain visible. Masked queries already contain that decoder token.
        if self.mode=='visible_target':restored[:,1:]=restored[:,1:]+queries[...,None]*self.mask_token
        decoded=self.decoder_norm(self.decoder(restored+self.position))
        return self.pixels(decoded[:,1:])

    def training_observations(self,images,content,queries):
        prepared,fraction=resized_observations(images,content)
        if (queries&~(fraction.flatten(1)>0)).any():raise ValueError('Onlyactualobservable currentpatchpixels maybereconstruction targets')
        pixel_content=F.interpolate(content.float(),size=(126,252),mode='bilinear',align_corners=False).clamp(0,1)
        weights=patchify(pixel_content.expand(-1,3,-1,-1)).detach();target=patchify(prepared).detach();mass=weights.sum(-1,keepdim=True).clamp_min(1e-6);mean=(target*weights).sum(-1,keepdim=True)/mass;variance=((target-mean).square()*weights).sum(-1,keepdim=True)/mass;target=(target-mean)/(variance+1e-6).sqrt()
        prediction=self.reconstruct_prepared(prepared,queries);selected=queries&(fraction.flatten(1)>0);masked_weight=weights*selected[...,None]
        loss=((prediction.float()-target).square()*masked_weight).sum()/masked_weight.sum().clamp_min(1)
        return dict(loss=loss,prediction=prediction,target=target,queries=selected,observable_patch_fraction=fraction,visible_encoder_patch_counts=(~queries).sum(1) if self.mode=='masked_visible_tokens' else torch.full((len(images),),162,device=images.device))

    def encoder_current_only(self,images,content):
        prepared,fraction=resized_observations(images,content);cls=self.backbone.forward_features(prepared)['x_norm_clstoken'];valid=fraction.flatten(1).sum(-1)>0
        return F.normalize(cls.float(),dim=-1)*valid[:,None]
