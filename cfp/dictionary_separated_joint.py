"""Same-source full-set decoder with ID binding separate from visual evidence.

Owned adaptation of the existing causal decoder, not official MOTIP. Content
conditioning and arbitrary-code conditioning have identical trainable capacity
and standardized activation scale. Slot codes remain bookkeeping only for
content conditioning; they never enter physical visual association scores.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .joint_identity_decoder import JointFishIDDecoder
from .centered_joint_identity import TokenFeedForward
from .predictive import checked_history


class DictionarySeparatedJoint(JointFishIDDecoder):
    def __init__(self, dimension=384, hidden=64, heads=4, capacity=512,
                 architecture='joint', binding_mode='content'):
        if architecture not in ('pair', 'cross_transformer', 'joint') or binding_mode not in ('codes', 'content') or capacity % 2:
            raise ValueError('Registered causal architecture and exact matched binding capacity required')
        super().__init__(dimension, hidden, heads, capacity, architecture='pair' if architecture=='pair' else 'joint')
        self.registered_architecture, self.binding_mode = architecture, binding_mode
        if binding_mode=='content':
            self.dictionary=nn.Sequential(nn.Linear(hidden, capacity//2, bias=False), nn.GELU(), nn.Linear(capacity//2, hidden, bias=False))
        if architecture=='cross_transformer':
            self.track_attention=TokenFeedForward(hidden)
            self.detection_attention=TokenFeedForward(hidden)
        # Only full-ID logits are trained in this registered stage.
        self.residual.requires_grad_(False)
        if architecture!='pair':
            self.pair_score.requires_grad_(False)
        else:
            for module in (self.track_attention, self.detection_attention, self.track_to_detection, self.detection_to_track):
                module.requires_grad_(False)
            self.null_track.requires_grad_(False)
            self.null_detection.requires_grad_(False)

    def forward(self, history, times, valid, frame, detections, detection_valid,
                geometry, appearance, track_state, confidence, codes=None, pair_valid=None):
        source, age, lengths, available, last = checked_history(
            history, times, valid, frame, 'log1p_15000')
        n, length, dimension = history.shape
        m = len(detections)
        if (dimension != self.dimension or detections.shape != (m, dimension)
                or detection_valid.shape != (m,) or detection_valid.dtype != torch.bool
                or geometry.shape != (n, m) or appearance.shape != (n, m)
                or track_state.shape != (n, 3) or confidence.shape != (m,)
                or n > self.capacity or valid.dtype != torch.bool):
            raise ValueError('Aligned whole-frame observable inputs required')
        if not all(torch.isfinite(x).all() for x in (detections, geometry, appearance, track_state, confidence)):
            raise ValueError('Finite whole-frame inputs required')
        if (confidence < 0).any() or (confidence > 1).any():
            raise ValueError('Detection confidence must lie in [0,1]')
        if codes is None:
            codes = torch.arange(n, device=history.device)
        if (codes.shape != (n,) or codes.dtype != torch.int64 or len(codes.unique()) != n
                or (codes < 0).any() or (codes >= self.capacity).any()):
            raise ValueError('Distinct temporary dictionary codes required')
        if pair_valid is None:
            pair_valid = torch.ones(n, m, dtype=torch.bool, device=history.device)
        if pair_valid.shape != (n, m) or pair_valid.dtype != torch.bool:
            raise ValueError('Explicit observable pair eligibility required')
        legal = pair_valid & available[:, None] & detection_valid[None]
        if m == 0:
            return dict(logits=history.new_empty(0, n + 1), residual=history.new_empty(n, 0), legal=legal, slot_codes=codes)
        current = self.visual(F.normalize(detections.detach().float(), dim=-1))
        current = current + self.confidence(confidence[:, None])
        if n:
            memory = self.visual(source) + self.time(age)
            safe = valid.clone()
            safe[~available, 0] = True  # padded empty row cannot cause softmax NaNs.
            if self.temporal_memory:
                track = self.pool(self.pool_query.expand(n, -1, -1), memory, memory,
                                  key_padding_mask=~safe, need_weights=False)[0][:, 0]
            else:
                track = self.visual(last)
            track = track + self.track_state(track_state)
            conditioning = self.dictionary(codes if self.binding_mode == "codes" else track)
            track = track + F.layer_norm(conditioning, (self.hidden,))
            if self.architecture == 'joint' and self.track_competition and available.any():
                track = track + self.track_attention(track[None], track[None], track[None],
                    key_padding_mask=~available[None], need_weights=False)[0][0]
            # Add a valid unmatched token to every attention pool. Entirely
            # occluded/gated frames remain finite without inventing observations.
            det_memory = torch.cat([current[None], self.null_detection], dim=1)
            bias = -(F.softplus(self.geometry_scale) * geometry
                     + F.softplus(self.appearance_scale) * appearance) if self.motion_bias else torch.zeros_like(geometry)
            track_bias = torch.cat([bias.masked_fill(~legal, -torch.inf), bias.new_zeros(n, 1)], dim=1)
            if self.architecture == 'joint':
                track = track + self.track_to_detection(track[None], det_memory, det_memory,
                    attn_mask=track_bias, need_weights=False)[0][0]
            track = self.track_norm(track)
        else:
            track = current.new_empty(0, self.hidden)
            bias = geometry
        if self.architecture == 'joint' and self.detection_competition and detection_valid.any():
            current = current + self.detection_attention(current[None], current[None], current[None],
                key_padding_mask=~detection_valid[None], need_weights=False)[0][0]
        track_memory = torch.cat([track[None], self.null_track], dim=1)
        det_bias = torch.cat([bias.T.masked_fill(~legal.T, -torch.inf), current.new_zeros(m, 1)], dim=1)
        if self.architecture == 'joint':
            current = current + self.detection_to_track(current[None], track_memory, track_memory,
                attn_mask=det_bias, need_weights=False)[0][0]
        current = self.detection_norm(current)
        edge = torch.cat([track[:, None].expand(n, m, -1), current[None].expand(n, m, -1),
                          geometry[..., None], appearance[..., None]], dim=-1)
        id_logits = self.detection_query(current) @ self.track_key(track).T / math.sqrt(self.hidden)
        if self.architecture == 'pair':
            id_logits = id_logits + self.pair_score(edge).squeeze(-1).T
        if self.motion_bias:
            id_logits = id_logits + bias.T
        id_logits = id_logits.masked_fill(~legal.T, -torch.inf)
        null = self.null_score(current)
        null = torch.where(detection_valid[:, None], null, torch.zeros_like(null))
        logits = torch.cat([id_logits, null], dim=1)
        residual = self.residual(edge).squeeze(-1).masked_fill(~legal, 0.)
        return dict(logits=logits, residual=residual, legal=legal, slot_codes=codes)

