"""Strictly current candidate-set interface shared by Phase VI modules.

N track rows, M current detections, T<=8 strictly past accepted observations.
Features use one frozen RGB coordinate space. No label, teacher, or future fields.
An empty/invalid candidate pool always has a feasible NULL action.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CandidateGraph:
    history: torch.Tensor                 # [N,T,D], floating
    history_frames: torch.Tensor          # [N,T], int64
    history_valid: torch.Tensor           # [N,T], bool, packed prefix
    candidates: torch.Tensor              # [M,D], floating, current only
    track_boxes: torch.Tensor             # [N,4] current motion predictions xyxy
    candidate_boxes: torch.Tensor         # [M,4] current measured xyxy
    confidence: torch.Tensor              # [M], [0,1]
    baseline_cost: torch.Tensor           # [N,M], finite original association
    legal: torch.Tensor                   # [N,M], bool explicit candidate mask
    age: torch.Tensor                     # [N], frames since track birth
    gap: torch.Tensor                     # [N], frames since last observation
    frame: int
    stage: int = 0                        # high=0, low=1, unconfirmed=2
    prototype: torch.Tensor = None        # [N,D], frozen ordinary GRU output

    def validate(self):
        if self.history.ndim != 3 or self.candidates.ndim != 2:
            raise ValueError('Expected [N,T,D] history and [M,D] candidates')
        n, t, d = self.history.shape
        m = self.candidates.shape[0]
        shapes = {'history_frames': (n, t), 'history_valid': (n, t),
                  'track_boxes': (n, 4), 'candidate_boxes': (m, 4),
                  'confidence': (m,), 'baseline_cost': (n, m), 'legal': (n, m),
                  'age': (n,), 'gap': (n,)}
        if self.candidates.shape[1] != d or not 0 <= t <= 8 or d <= 0:
            raise ValueError('Inconsistent frozen feature coordinates/history budget')
        for name, shape in shapes.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError('Invalid shape: ' + name)
        if self.history_valid.dtype != torch.bool or self.legal.dtype != torch.bool or self.history_frames.dtype != torch.int64:
            raise ValueError('Masks must be bool and observation frames int64')
        tensors = [v for v in self.__dict__.values() if isinstance(v, torch.Tensor)]
        if any(v.device != self.history.device for v in tensors):
            raise ValueError('Graph tensors must share device')
        if any(v.is_floating_point() and not torch.isfinite(v).all() for v in tensors):
            raise ValueError('Nonfinite input graph')
        if self.frame < 1 or self.stage not in (0, 1, 2):
            raise ValueError('Positive current frame and explicit stage required')
        if ((self.history_frames <= 0) & self.history_valid).any() or ((self.history_frames >= self.frame) & self.history_valid).any():
            raise ValueError('Every accepted observation must be strictly past')
        length = self.history_valid.sum(1)
        if not torch.equal(self.history_valid, torch.arange(t, device=self.history.device)[None] < length[:, None]):
            raise ValueError('History must be packed in temporal order')
        if t > 1 and ((self.history_frames[:, 1:] <= self.history_frames[:, :-1]) & self.history_valid[:, 1:]).any():
            raise ValueError('History timestamps must strictly increase')
        if ((self.confidence < 0) | (self.confidence > 1)).any() or (self.age < 0).any() or (self.gap < 0).any():
            raise ValueError('Out-of-range observable confidence/age/gap')
        for boxes in (self.track_boxes, self.candidate_boxes):
            if (boxes[:, 2:] <= boxes[:, :2]).any():
                raise ValueError('Positive box dimensions required')
        if self.prototype is not None and tuple(self.prototype.shape) != (n, d):
            raise ValueError('Prototype coordinates differ')
        return self


@dataclass(frozen=True)
class CandidateDecision:
    scores: torch.Tensor                  # [N,M+1], larger is better; NULL last
    memory_write_risk: torch.Tensor       # [N,M], [0,1]
    legal: torch.Tensor                   # [N,M], bool

    def validate(self):
        n, m = self.legal.shape
        if self.scores.shape != (n, m + 1) or self.memory_write_risk.shape != (n, m):
            raise ValueError('Candidate output shape mismatch')
        if not torch.isfinite(self.scores[:, -1]).all():
            raise ValueError('NULL must always be finite')
        if not torch.isfinite(self.scores[:, :-1][self.legal]).all():
            raise ValueError('Legal score must be finite')
        if torch.isfinite(self.scores[:, :-1][~self.legal]).any():
            raise ValueError('Illegal candidate must be masked')
        if not torch.isfinite(self.memory_write_risk).all() or ((self.memory_write_risk < 0) | (self.memory_write_risk > 1)).any():
            raise ValueError('Risk must be finite probability')
        return self

