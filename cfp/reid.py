"""Small fish ReID baseline and temporal/sequence-aware batch-hard loss."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50


class FishReID(nn.Module):
    def __init__(self, num_identities=98, dimension=512):
        super().__init__()
        self.backbone = resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        self.projection = nn.Linear(2048, dimension, bias=False)
        self.neck = nn.BatchNorm1d(dimension)
        self.neck.bias.requires_grad_(False)
        self.classifier = nn.Linear(dimension, num_identities, bias=False)

    def initialize_imagenet(self, path):
        state = torch.load(path, map_location='cpu')
        # Verify the full original classifier checkpoint before dropping its head.
        original = resnet50(weights=None)
        original.load_state_dict(state, strict=True)
        original.fc = nn.Identity()
        self.backbone.load_state_dict(original.state_dict(), strict=True)

    def forward(self, images):
        feature = self.projection(self.backbone(images))
        normalized = F.normalize(feature.float(), dim=-1)
        logits = self.classifier(self.neck(feature))
        return normalized, logits


def temporal_batch_hard(embedding, labels, frames, sequences, margin=.3, minimum_gap=25):
    distances = torch.cdist(embedding.float(), embedding.float())
    same = labels[:, None] == labels[None, :]
    gap = (frames[:, None] - frames[None, :]).abs()
    positive = same & (gap >= minimum_gap)
    # Different videos can be trivially discriminated by environment/species.
    # Require same-video negatives, omitting anchors lacking valid pairs.
    negative = ~same & (sequences[:, None] == sequences[None, :])
    valid = positive.any(dim=1) & negative.any(dim=1)
    if not valid.any():
        return embedding.sum() * 0, 0.
    hardest_positive = distances.masked_fill(~positive, -torch.inf).amax(dim=1)
    hardest_negative = distances.masked_fill(~negative, torch.inf).amin(dim=1)
    loss = F.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()
    return loss, float(valid.float().mean().detach().cpu())
