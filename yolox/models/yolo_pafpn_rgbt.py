#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""Independent RGB and IR YOLOPAFPN branches with conservative fusion modes."""

import torch
import torch.nn as nn

from .yolo_pafpn import YOLOPAFPN
from .qra_fusion import (
    DynamicQualityFusion,
    QualitySpatialFusion,
    SpatialComplementarityFusion,
    TemporalComplementarityFusion,
)


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
        qra_modes = {
            "dynamic_quality",
            "spatial_complementarity",
            "temporal_complementarity",
            "quality_spatial",
            "quality_spatial_redundancy",
        }
        if fusion_mode in {
            "gated_residual",
            *qra_modes,
        }:
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
            raise ValueError("unsupported fusion mode: %r" % (fusion_mode,))

        self.fusion_modules = nn.ModuleList()
        if fusion_mode in qra_modes:
            channels = [int(channel * width) for channel in in_channels]
            for channel in channels:
                if fusion_mode == "dynamic_quality":
                    module = DynamicQualityFusion(channel)
                elif fusion_mode == "spatial_complementarity":
                    module = SpatialComplementarityFusion(channel)
                elif fusion_mode == "temporal_complementarity":
                    module = TemporalComplementarityFusion(channel)
                elif fusion_mode == "quality_spatial":
                    module = QualitySpatialFusion(channel)
                else:  # quality_spatial_redundancy
                    module = QualitySpatialFusion(channel, use_redundancy=True)
                self.fusion_modules.append(module)

        if self.freeze_rgb:
            for parameter in self.rgb_branch.parameters():
                parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        # Prevent frozen RGB BatchNorm running statistics from drifting while
        # the IR branch is adapted to its own modality distribution.
        if mode and self.freeze_rgb:
            self.rgb_branch.eval()
        for module in self.fusion_modules:
            if isinstance(module, TemporalComplementarityFusion):
                module.set_stateful(not mode)
        return self

    def reset_fusion_state(self):
        """Reset all causal temporal modules at a video boundary."""

        for module in self.fusion_modules:
            if isinstance(module, TemporalComplementarityFusion):
                module.reset_state()

    def forward(self, x, fusion_mode=None, return_aux=False):
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(
                f"YOLOPAFPNRGBT expects [B, 6, H, W], got {tuple(x.shape)}"
            )

        mode = self.fusion_mode if fusion_mode is None else fusion_mode
        rgb_features = self.rgb_branch(x[:, :3])
        ir_features = self.ir_branch(x[:, 3:6])
        fusion_info = {
            "mode": mode,
            "levels": [],
            "rgb_features": rgb_features,
            "aux_features": ir_features,
            # Keep the historical key for downstream analysis scripts.
            "ir_features": ir_features,
        }
        if mode == "avg":
            fused_features = tuple(
                0.5 * (rgb_feature + ir_feature)
                for rgb_feature, ir_feature in zip(rgb_features, ir_features)
            )
            fusion_info["fusion_gates"] = tuple(
                torch.full(
                    (rgb_feature.shape[0], 1, 1, 1),
                    0.5,
                    device=rgb_feature.device,
                    dtype=rgb_feature.dtype,
                )
                for rgb_feature in rgb_features
            )
        elif mode == "gated_residual":
            gates = torch.sigmoid(self.fusion_logits)
            fused_features = tuple(
                rgb_feature + gates[index] * (ir_feature - rgb_feature)
                for index, (rgb_feature, ir_feature) in enumerate(
                    zip(rgb_features, ir_features)
                )
            )
            fusion_info["fusion_gates"] = gates
        elif mode in {
            "dynamic_quality",
            "spatial_complementarity",
            "temporal_complementarity",
            "quality_spatial",
            "quality_spatial_redundancy",
        }:
            fused = []
            gate_values = []
            for index, (rgb_feature, ir_feature, module) in enumerate(
                zip(rgb_features, ir_features, self.fusion_modules)
            ):
                if mode == "quality_spatial":
                    fused_feature, info = module(
                        rgb_feature, ir_feature, self.fusion_logits[index]
                    )
                else:
                    fused_feature, info = module(
                        rgb_feature, ir_feature, self.fusion_logits[index]
                    )
                fused.append(fused_feature)
                gate_values.append(info["effective_weight"])
                fusion_info["levels"].append(info)
            fused_features = tuple(fused)
            fusion_info["fusion_gates"] = tuple(gate_values)
        else:
            raise ValueError(f"Unsupported fusion mode {mode!r}")

        if return_aux:
            return fused_features, fusion_info
        return fused_features
