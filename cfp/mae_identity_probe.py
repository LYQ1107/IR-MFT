"""Separate identity supervision on a frozen original or fit-only MAE encoder."""
import torch
from torch import nn
from torch.nn import functional as F
from .dinov2_identity import resized_observations
from .reid import temporal_batch_hard


class FrozenEncoderIdentityProbe(nn.Module):
    def __init__(self,backbone,identities,loss_mode='ce_triplet'):
        super().__init__()
        if loss_mode not in ('ce_triplet','ce_only','triplet_only') or identities<2:raise ValueError('Registeredidentityloss andrealfitvocabulary required')
        self.backbone=backbone.eval().requires_grad_(False);self.loss_mode=loss_mode
        self.projection=nn.Linear(384,384,bias=False);nn.init.eye_(self.projection.weight)
        self.classifier=nn.Linear(384,identities,bias=False);nn.init.normal_(self.classifier.weight,std=.001)
        if loss_mode=='triplet_only':self.classifier.requires_grad_(False)

    def train(self,mode=True):
        super().train(mode);self.backbone.eval();return self

    def forward(self,images,content):
        prepared,fraction=resized_observations(images,content)
        with torch.no_grad():cls=self.backbone.forward_features(prepared)['x_norm_clstoken']
        valid=fraction.flatten(1).sum(-1)>0;feature=self.projection(cls.detach().float());embedding=F.normalize(feature,dim=-1)*valid[:,None]
        return dict(embedding=embedding,raw_global=F.normalize(cls.float(),dim=-1)*valid[:,None],logits=self.classifier(feature),crop_valid=valid)

    def head_state(self):return dict(projection=self.projection.state_dict(),classifier=self.classifier.state_dict())
    def load_head(self,state):self.projection.load_state_dict(state['projection'],strict=True);self.classifier.load_state_dict(state['classifier'],strict=True)


def probe_identity_loss(output,labels,frames,videos,mode,gap=25):
    valid=output['crop_valid']
    if not valid.any():raise ValueError('No actualcontent in identitybatch')
    zero=output['embedding'].sum()*0;ce=zero;triplet=zero;coverage=0.
    if mode in ('ce_triplet','ce_only'):ce=F.cross_entropy(output['logits'][valid].float(),labels[valid],label_smoothing=.1)
    if mode in ('ce_triplet','triplet_only'):triplet,coverage=temporal_batch_hard(output['embedding'][valid],labels[valid],frames[valid],videos[valid],minimum_gap=gap)
    if mode not in ('ce_triplet','ce_only','triplet_only'):raise ValueError('No implicitloss aliases')
    return ce+triplet,dict(ce=float(ce.detach()),triplet=float(triplet.detach()),temporal_coverage=coverage)
