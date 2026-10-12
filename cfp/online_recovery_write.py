"""Provisional identity memory without rejecting current geometry/output.

The original cost/Hungarian/lifecycle run unchanged. Observed learned risk only
controls writes to existing identity histories. Newborns and unavailable evidence
fall back to original writes. GT, physical references and future frames are absent.
"""
import types

import numpy as np
import torch

from .candidate_graph import CandidateGraph
from .history import anchor_recent_history
from .recoverability import observable_candidate_features


def install_risk_memory(tracker, estimator, calibration, threshold=.2, constant_risk=None):
    if not 0 <= threshold <= 1 or constant_risk is not None and not 0 <= constant_risk <= 1:
        raise ValueError('Finite risk probability thresholds required')
    original_cost, original_update = tracker.association_cost, tracker.update
    tracker.phase_vi_memory_counts = dict(accepted=0, provisional_rejected=0, unavailable_fallback=0)
    tracker._phase_vi_memory_evidence = {}

    def guard(track):
        if getattr(track, '_phase_vi_memory_wrapped', False):
            return
        original_write = track.write_identity
        def write(frame, ema):
            if not track.pending:
                return False
            risk = tracker._phase_vi_memory_evidence.get((track.track_id, track.current_proposal_index))
            if risk is None:
                tracker.phase_vi_memory_counts['unavailable_fallback'] += 1
                return original_write(frame, ema)
            if risk > threshold:
                track.pending = False
                tracker.phase_vi_memory_counts['provisional_rejected'] += 1
                return False
            tracker.phase_vi_memory_counts['accepted'] += 1
            return original_write(frame, ema)
        track.write_identity = write
        track._phase_vi_memory_wrapped = True

    def cost(self, tracks, detections):
        baseline = original_cost(tracks, detections)
        for track in tracks:
            guard(track)
        if not baseline.size or any(d.current_rgb is None for d in detections):
            return baseline
        n, m = baseline.shape
        if constant_risk is not None:
            risk = np.full((n, m), constant_risk)
            available = np.array([bool(t.identity_history) or t.initial_identity_rgb is not None for t in tracks])
        else:
            past = np.zeros((n, 8, 512), np.float32)
            times = np.zeros((n, 8), np.int64)
            valid = np.zeros((n, 8), bool)
            for i, track in enumerate(tracks):
                anchor = None if track.initial_identity_rgb is None else (track.initial_identity_frame, track.initial_identity_rgb, track.initial_identity_ir)
                history = anchor_recent_history(list(track.identity_history), anchor, 8)
                for j, (frame, rgb, _) in enumerate(history):
                    past[i, j], times[i, j], valid[i, j] = rgb, frame, True
            inputs = [torch.from_numpy(v) for v in (past, times, valid)]
            candidate = np.asarray([d.current_rgb for d in detections], np.float32)
            if candidate.shape != (m, 512) or not np.isfinite(candidate).all():
                return baseline
            with torch.inference_mode():
                prototype, available = self.predictor(*inputs, torch.full((n,), self.frame_id, dtype=torch.int64))
                graph = CandidateGraph(*inputs, torch.from_numpy(candidate),
                    torch.tensor(np.array([t.tlbr for t in tracks]), dtype=torch.float32),
                    torch.tensor(np.array([d.tlbr for d in detections]), dtype=torch.float32),
                    torch.tensor([d.score for d in detections], dtype=torch.float32),
                    torch.from_numpy(baseline.astype(np.float32)), torch.ones(n, m, dtype=torch.bool),
                    torch.tensor([self.frame_id - t.start_frame for t in tracks]),
                    torch.tensor([self.frame_id - t.end_frame for t in tracks]), self.frame_id, prototype=prototype)
                evidence = observable_candidate_features(graph)
                risk = (estimator(evidence, graph.legal) / calibration['temperature'] + calibration['bias']).sigmoid().numpy()
                available = available.numpy()
        for i, track in enumerate(tracks):
            if available[i]:
                for j, detection in enumerate(detections):
                    self._phase_vi_memory_evidence[(track.track_id, detection.current_proposal_index)] = float(risk[i, j])
        return baseline

    def update(self, *args, **kwargs):
        self._phase_vi_memory_evidence = {}
        return original_update(*args, **kwargs)

    tracker.association_cost = types.MethodType(cost, tracker)
    tracker.update = types.MethodType(update, tracker)
    return tracker

