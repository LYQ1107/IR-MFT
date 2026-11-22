"""Metric learning in the same normalized neck space used at inference."""
import torch
from torch.nn import functional as F


def native_neck_forward(model, value):
    raw=model.projection(model.backbone(value))
    neck=model.neck(raw)
    return F.normalize(neck.float(),dim=-1),model.classifier(neck)
