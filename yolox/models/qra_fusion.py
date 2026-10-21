#!/usr/bin/env python3
"""Lightweight quality/reciprocity-aware multimodal fusion blocks.

The blocks in this file deliberately operate on already computed FPN features.
They do not change the detector head or the two-stream backbone, which keeps
the V3 checkpoint and the RGB-only controls comparable.  Every block accepts
the same ``[B, C, H, W]`` pair and returns a fused feature plus a uniform
diagnostic dictionary.

The last operation that produces a gate correction is zero initialised.  With
the V3 static logit supplied by :class:`YOLOPAFPNRGBT`, a freshly constructed
candidate therefore produces the V3 residual fusion exactly (up to floating
point round-off).  The redundancy estimator is initialised near zero rather
than at an arbitrary learned operating point.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_pair(rgb_feature: torch.Tensor, aux_feature: torch.Tensor) -> None:
    if rgb_feature.ndim != 4 or aux_feature.ndim != 4:
        raise ValueError(
            "fusion features must have shape [B, C, H, W], got "
            f"{tuple(rgb_feature.shape)} and {tuple(aux_feature.shape)}"
        )
    if rgb_feature.shape != aux_feature.shape:
        raise ValueError(
            "RGB and auxiliary feature shapes differ: "
            f"{tuple(rgb_feature.shape)} vs {tuple(aux_feature.shape)}"
        )


def _as_logit(static_logit, feature: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar/one-dimensional FPN logit to a feature map."""

    if not torch.is_tensor(static_logit):
        static_logit = torch.as_tensor(static_logit, device=feature.device)
    return static_logit.to(device=feature.device, dtype=feature.dtype).reshape(
        1, 1, 1, 1
    )


def _info(
    rgb_feature: torch.Tensor,
    aux_feature: torch.Tensor,
    gate: torch.Tensor,
    delta: torch.Tensor,
    redundancy: Optional[torch.Tensor] = None,
    extra_quality: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, object]:
    """Create the common, tensor-valued fusion diagnostics.

    Values are kept as tensors so callers can aggregate them without a second
    forward pass.  The caller may detach them when serialising a manifest.
    """

    if redundancy is None:
        redundancy = torch.zeros_like(gate)
    rgb_quality = rgb_feature.detach().abs().mean(dim=(1, 2, 3))
    aux_quality = aux_feature.detach().abs().mean(dim=(1, 2, 3))
    disagreement = (rgb_feature.detach() - aux_feature.detach()).abs().mean(
        dim=(1, 2, 3)
    )
    quality = {
        "rgb_abs_mean": rgb_quality,
        "auxiliary_abs_mean": aux_quality,
        "disagreement_abs_mean": disagreement,
    }
    if extra_quality:
        quality.update(extra_quality)
    effective = gate
    return {
        "rgb_weight": 1.0 - effective,
        "auxiliary_weight": effective,
        # Alias used by a few downstream analysis scripts.
        "aux_weight": effective,
        "quality_stats": quality,
        "redundancy": redundancy,
        "effective_weight": effective,
        "gate_range": torch.stack((effective.detach().amin(), effective.detach().amax())),
        "delta": delta,
    }


class DynamicQualityFusion(nn.Module):
    """Frame-level/FPN-level quality gate.

    The gate observes pooled RGB, auxiliary and absolute-disagreement vectors;
    it does not use labels, track state or future frames.
    """

    def __init__(self, channels: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = max(channels // 8, 32) if hidden is None else int(hidden)
        self.delta_mlp = nn.Sequential(
            nn.Linear(3 * channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.delta_mlp[-1].weight)
        nn.init.zeros_(self.delta_mlp[-1].bias)

    def forward(
        self,
        rgb_feature: torch.Tensor,
        aux_feature: torch.Tensor,
        static_logit,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        _check_pair(rgb_feature, aux_feature)
        rgb_pool = F.adaptive_avg_pool2d(rgb_feature, 1).flatten(1)
        aux_pool = F.adaptive_avg_pool2d(aux_feature, 1).flatten(1)
        diff_pool = F.adaptive_avg_pool2d(
            (rgb_feature - aux_feature).abs(), 1
        ).flatten(1)
        delta = self.delta_mlp(torch.cat((rgb_pool, aux_pool, diff_pool), dim=1))
        delta = delta[:, :, None, None]
        gate = torch.sigmoid(_as_logit(static_logit, rgb_feature) + 3.0 * torch.tanh(delta))
        fused = rgb_feature + gate * (aux_feature - rgb_feature)
        return fused, _info(rgb_feature, aux_feature, gate, delta)


class RedundancyEstimator(nn.Module):
    """Small local redundancy predictor used by the F2c ablation."""

    def __init__(self, channels: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = max(channels // 8, 32) if hidden is None else int(hidden)
        self.net = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        # A closed gate at construction time: sigmoid(-8) is below 3.4e-4.
        nn.init.constant_(self.net[-1].bias, -8.0)

    def forward(self, rgb_feature: torch.Tensor, aux_feature: torch.Tensor) -> torch.Tensor:
        disagreement = (rgb_feature - aux_feature).abs()
        product = rgb_feature * aux_feature
        return torch.sigmoid(self.net(torch.cat((disagreement, product), dim=1)))


class SpatialComplementarityFusion(nn.Module):
    """Spatially selective complementary fusion with an optional redundancy gate."""

    def __init__(
        self,
        channels: int,
        hidden: Optional[int] = None,
        use_redundancy: bool = False,
    ):
        super().__init__()
        hidden = max(channels // 8, 32) if hidden is None else int(hidden)
        self.use_redundancy = bool(use_redundancy)
        self.spatial_delta = nn.Sequential(
            nn.Conv2d(4 * channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.spatial_delta[-1].weight)
        nn.init.zeros_(self.spatial_delta[-1].bias)
        self.redundancy = (
            RedundancyEstimator(channels, hidden=hidden) if self.use_redundancy else None
        )

    def forward(
        self,
        rgb_feature: torch.Tensor,
        aux_feature: torch.Tensor,
        static_logit,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        _check_pair(rgb_feature, aux_feature)
        difference = (rgb_feature - aux_feature).abs()
        product = rgb_feature * aux_feature
        delta = self.spatial_delta(
            torch.cat((rgb_feature, aux_feature, difference, product), dim=1)
        )
        gate = torch.sigmoid(_as_logit(static_logit, rgb_feature) + delta)
        redundancy = (
            self.redundancy(rgb_feature, aux_feature)
            if self.redundancy is not None
            else torch.zeros_like(gate)
        )
        effective = gate * (1.0 - redundancy)
        fused = rgb_feature + effective * (aux_feature - rgb_feature)
        info = _info(rgb_feature, aux_feature, effective, delta, redundancy)
        info["raw_gate"] = gate
        return fused, info


class QualitySpatialFusion(nn.Module):
    """Composed F1+F2 gate with one bounded effective weight.

    The frame-level correction and spatial correction are summed before the
    sigmoid.  This avoids applying two residual fusions in sequence and keeps
    the zero-correction initialization exactly equivalent to the V3 gate.
    """

    def __init__(self, channels: int, hidden: Optional[int] = None, use_redundancy=False):
        super().__init__()
        hidden = max(channels // 8, 32) if hidden is None else int(hidden)
        self.quality_delta = nn.Sequential(
            nn.Linear(3 * channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.spatial_delta = nn.Sequential(
            nn.Conv2d(4 * channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.quality_delta[-1].weight)
        nn.init.zeros_(self.quality_delta[-1].bias)
        nn.init.zeros_(self.spatial_delta[-1].weight)
        nn.init.zeros_(self.spatial_delta[-1].bias)
        self.redundancy = (
            RedundancyEstimator(channels, hidden=hidden) if use_redundancy else None
        )

    def forward(self, rgb_feature, aux_feature, static_logit):
        _check_pair(rgb_feature, aux_feature)
        difference = (rgb_feature - aux_feature).abs()
        product = rgb_feature * aux_feature
        pooled = torch.cat(
            (
                F.adaptive_avg_pool2d(rgb_feature, 1).flatten(1),
                F.adaptive_avg_pool2d(aux_feature, 1).flatten(1),
                F.adaptive_avg_pool2d(difference, 1).flatten(1),
            ),
            dim=1,
        )
        frame_delta = self.quality_delta(pooled)[:, :, None, None]
        spatial_delta = self.spatial_delta(
            torch.cat((rgb_feature, aux_feature, difference, product), dim=1)
        )
        delta = spatial_delta + 3.0 * torch.tanh(frame_delta)
        gate = torch.sigmoid(_as_logit(static_logit, rgb_feature) + delta)
        redundancy = (
            self.redundancy(rgb_feature, aux_feature)
            if self.redundancy is not None
            else torch.zeros_like(gate)
        )
        effective = gate * (1.0 - redundancy)
        fused = rgb_feature + effective * (aux_feature - rgb_feature)
        info = _info(
            rgb_feature,
            aux_feature,
            effective,
            delta,
            redundancy,
            extra_quality={"frame_delta": frame_delta.detach()},
        )
        info["raw_gate"] = gate
        return fused, info


class TemporalComplementarityFusion(nn.Module):
    """Causal frame-history gate based on pooled multimodal feature evidence.

    Stateful evaluation is opt-in through ``set_stateful(True)`` and is reset
    explicitly at sequence boundaries by ``reset_state``.  Training defaults
    to stateless operation so random image batches cannot accidentally connect
    unrelated videos.  A clip data loader can opt into stateful training later
    without changing this feature interface.
    """

    def __init__(
        self,
        channels: int,
        state_dim: int = 64,
        hidden: Optional[int] = None,
    ):
        super().__init__()
        hidden = max(channels // 8, 32) if hidden is None else int(hidden)
        self.state_dim = int(state_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(3 * channels, self.state_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(self.state_dim, self.state_dim)
        self.delta_mlp = nn.Sequential(
            nn.Linear(self.state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.delta_mlp[-1].weight)
        nn.init.zeros_(self.delta_mlp[-1].bias)
        self.stateful = False
        self._hidden: Optional[torch.Tensor] = None

    def set_stateful(self, stateful: bool) -> None:
        self.stateful = bool(stateful)
        if not self.stateful:
            self.reset_state()

    def reset_state(self) -> None:
        self._hidden = None

    def forward(
        self,
        rgb_feature: torch.Tensor,
        aux_feature: torch.Tensor,
        static_logit,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        _check_pair(rgb_feature, aux_feature)
        rgb_pool = F.adaptive_avg_pool2d(rgb_feature, 1).flatten(1)
        aux_pool = F.adaptive_avg_pool2d(aux_feature, 1).flatten(1)
        diff_pool = F.adaptive_avg_pool2d(
            (rgb_feature - aux_feature).abs(), 1
        ).flatten(1)
        z = self.input_projection(torch.cat((rgb_pool, aux_pool, diff_pool), dim=1))
        hidden = self._hidden
        if (
            hidden is None
            or hidden.shape[0] != z.shape[0]
            or hidden.shape[1] != self.state_dim
            or hidden.device != z.device
            or hidden.dtype != z.dtype
        ):
            hidden = z.new_zeros((z.shape[0], self.state_dim))
        hidden = self.gru(z, hidden)
        if self.stateful:
            self._hidden = hidden.detach()
        delta = self.delta_mlp(hidden)[:, :, None, None]
        gate = torch.sigmoid(_as_logit(static_logit, rgb_feature) + 3.0 * torch.tanh(delta))
        fused = rgb_feature + gate * (aux_feature - rgb_feature)
        info = _info(
            rgb_feature,
            aux_feature,
            gate,
            delta,
            extra_quality={"temporal_state_norm": hidden.detach().norm(dim=1)},
        )
        info["stateful"] = self.stateful
        return fused, info
