#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""Independent RGB and IR YOLOPAFPN branches with conservative fusion modes."""

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
        freeze_rgb=False,
    ):
        super().__init__()
        self.input_channels = 6
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
        self.freeze_rgb = freeze_rgb
        if fusion_mode == "gated_residual":
            # A small initial IR contribution keeps the source visible model
            # dominant while giving the IR branch a useful training gradient.
            # The sigmoid keeps the fusion convex and prevents an early IR
            # scale explosion from destroying the pretrained detector.
            initial_gate = 0.05
            initial_logit = torch.logit(torch.tensor(initial_gate))
            self.fusion_logits = nn.Parameter(
                initial_logit.repeat(len(in_channels)).float()
            )
        elif fusion_mode != "avg":
            raise ValueError(
                "fusion_mode must be 'avg' or 'gated_residual', "
                f"got {fusion_mode!r}"
            )

        if self.freeze_rgb:
            for parameter in self.rgb_branch.parameters():
                parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        # Prevent frozen RGB BatchNorm running statistics from drifting while
        # the IR branch is adapted to its own modality distribution.
        if mode and self.freeze_rgb:
            self.rgb_branch.eval()
        return self

    def forward(self, x, fusion_mode=None, return_aux=False):
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(
                f"YOLOPAFPNRGBT expects [B, 6, H, W], got {tuple(x.shape)}"
            )

        mode = self.fusion_mode if fusion_mode is None else fusion_mode
        rgb_features = self.rgb_branch(x[:, :3])
        ir_features = self.ir_branch(x[:, 3:6])
        if mode == "avg":
            fused_features = tuple(
                0.5 * (rgb_feature + ir_feature)
                for rgb_feature, ir_feature in zip(rgb_features, ir_features)
            )
        elif mode == "gated_residual":
            gates = torch.sigmoid(self.fusion_logits)
            fused_features = tuple(
                rgb_feature + gates[index] * (ir_feature - rgb_feature)
                for index, (rgb_feature, ir_feature) in enumerate(
                    zip(rgb_features, ir_features)
                )
            )
        else:
            raise ValueError(f"Unsupported fusion mode {mode!r}")

        if return_aux:
            return fused_features, {
                "rgb_features": rgb_features,
                "ir_features": ir_features,
                "fusion_gates": (
                    torch.sigmoid(self.fusion_logits)
                    if hasattr(self, "fusion_logits")
                    else None
                ),
            }
        return fused_features
