#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""Independent RGB and IR YOLOPAFPN branches with V1 average fusion."""

import torch
import torch.nn as nn

from .yolo_pafpn import YOLOPAFPN


class YOLOPAFPNRGBT(nn.Module):
    """Two independent YOLOPAFPN instances followed by FPN-level fusion."""

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=(256, 512, 1024),
        depthwise=False,
        act="silu",
        fusion_mode="avg",
    ):
        super().__init__()
        self.rgb_branch = YOLOPAFPN(
            depth,
            width,
            in_features=in_features,
            in_channels=list(in_channels),
            depthwise=depthwise,
            act=act,
        )
        self.ir_branch = YOLOPAFPN(
            depth,
            width,
            in_features=in_features,
            in_channels=list(in_channels),
            depthwise=depthwise,
            act=act,
        )
        self.fusion_mode = fusion_mode

    def forward(self, x, fusion_mode=None, return_aux=False):
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(
                f"YOLOPAFPNRGBT expects [B, 6, H, W], got {tuple(x.shape)}"
            )

        mode = self.fusion_mode if fusion_mode is None else fusion_mode
        if mode != "avg":
            raise ValueError(f"V1 only supports fusion_mode='avg', got {mode!r}")

        rgb_features = self.rgb_branch(x[:, :3])
        ir_features = self.ir_branch(x[:, 3:6])
        fused_features = tuple(
            0.5 * (rgb_feature + ir_feature)
            for rgb_feature, ir_feature in zip(rgb_features, ir_features)
        )

        if return_aux:
            return fused_features, {
                "rgb_features": rgb_features,
                "ir_features": ir_features,
            }
        return fused_features
