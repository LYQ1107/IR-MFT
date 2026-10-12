"""Optional current/past-only risk scoring on the preserved GRU tracker.

Disabled mode returns the exact original cost object without reading new inputs.
Policy scores are experiments, not calibrated correctness guarantees.
"""
import types

import numpy as np
import torch

from .candidate_graph import CandidateGraph
from .history import anchor_recent_history
from .recoverability import observable_candidate_features


def install_risk_association(tracker, estimator=None, calibration=None, policy='disabled', threshold=.2):
    if policy not in ('disabled', 'risk_gate', 'risk_score', 'margin_gate', 'entropy_gate', 'confidence_gate'):
        raise ValueError('Explicit controlled risk policy required')
    if policy.startswith('risk_') and estimator is None:
        raise ValueError('Risk policy requires trained estimator')
    original = tracker.association_cost
    tracker.phase_vi_risk_counts = dict(scored_edges=0, modified_edges=0, calls=0)
    tracker._phase_vi_risk_stage = 0
    original_update = tracker.update

    def update(self, *args, **kwargs):
        self._phase_vi_risk_stage = 0
        return original_update(*args, **kwargs)

    def association_cost(self, tracks, detections):
        cost = original(tracks, detections)
        stage = self._phase_vi_risk_stage
        self._phase_vi_risk_stage += 1
        if policy == 'disabled' or stage > 0 or not cost.size:
            return cost
        self.phase_vi_risk_counts['calls'] += 1
        if policy == 'confidence_gate':
            result = cost.copy()
            result[:, np.array([d.score for d in detections]) < threshold] = 1.01
        elif policy == 'margin_gate':
            result = cost.copy()
            if cost.shape[1] > 1:
                ranked = np.sort(cost, axis=1)
                result[ranked[:, 1] - ranked[:, 0] < threshold] = 1.01
        else:
            n, m = cost.shape
            past = np.zeros((n, 8, 512), np.float32)
            times = np.zeros((n, 8), np.int64)
            valid = np.zeros((n, 8), bool)
            for i, track in enumerate(tracks):
                anchor = None if track.initial_identity_rgb is None else (track.initial_identity_frame, track.initial_identity_rgb, track.initial_identity_ir)
                history = anchor_recent_history(list(track.identity_history), anchor, 8)
                for j, (frame, rgb, _) in enumerate(history):
                    if rgb is not None:
                        past[i, j], times[i, j], valid[i, j] = rgb, frame, True
            inputs = [torch.from_numpy(v) for v in (past, times, valid)]
            with torch.inference_mode():
                prototype, available = self.predictor(*inputs, torch.full((n,), self.frame_id, dtype=torch.int64))
                if any(d.current_rgb is None for d in detections):
                    return cost
                candidate = np.array([d.current_rgb for d in detections], np.float32)
                if candidate.shape != (m, 512) or not np.isfinite(candidate).all():
                    return cost  # Explicit original fallback for unusable new evidence.
                graph = CandidateGraph(*inputs, torch.from_numpy(candidate),
                    torch.tensor(np.array([t.tlbr for t in tracks]), dtype=torch.float32),
                    torch.tensor(np.array([d.tlbr for d in detections]), dtype=torch.float32),
                    torch.tensor([d.score for d in detections], dtype=torch.float32),
                    torch.from_numpy(cost.astype(np.float32)), torch.ones(n, m, dtype=torch.bool),
                    torch.tensor([self.frame_id - t.start_frame for t in tracks]),
                    torch.tensor([self.frame_id - t.end_frame for t in tracks]), self.frame_id, prototype=prototype)
                evidence = observable_candidate_features(graph)
                if policy == 'entropy_gate':
                    result = cost.copy()
                    result[evidence[:, 0, 8].numpy() > threshold] = 1.01
                else:
                    raw = estimator(evidence, graph.legal)
                    risk = (raw / calibration['temperature'] + calibration['bias']).sigmoid().numpy()
                    result = cost.copy()
                    if policy == 'risk_gate':
                        result[risk > threshold] = 1.01
                    else:
                        result[:] = risk
                result[~available.numpy()] = cost[~available.numpy()]
        self.phase_vi_risk_counts['scored_edges'] += cost.size
        self.phase_vi_risk_counts['modified_edges'] += int(np.count_nonzero(result != cost))
        return result

    tracker.update = types.MethodType(update, tracker)
    tracker.association_cost = types.MethodType(association_cost, tracker)
    return tracker

