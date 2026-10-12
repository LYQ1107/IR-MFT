"""Clone-safe risk memory: branch state never closes over another tracker/track."""
import numpy as np
import torch

from .association import IdentityTrack
from .candidate_graph import CandidateGraph
from .history import anchor_recent_history
from .recoverability import observable_candidate_features


class RiskMemoryTrack(IdentityTrack):
    def write_identity(self, frame, ema):
        if not self.pending:
            return False
        risk = self.risk_memory_evidence.get(self.current_proposal_index) if self.risk_memory_frame == frame else None
        if risk is None:
            key = 'unavailable_fallback'
        elif risk > self.risk_memory_threshold:
            self.pending = False
            self.risk_memory_counts['provisional_rejected'] += 1
            return False
        else:
            key = 'accepted'
        self.risk_memory_counts[key] += 1
        return super().write_identity(frame, ema)


def install_isolated_risk_memory(tracker, estimator, calibration, threshold=.2, constant_risk=None):
    if tracker.write_policy is not None or not 0 <= threshold <= 1:
        raise ValueError('Independent ordinary write policy and bounded threshold required')
    if constant_risk is not None and not 0 <= constant_risk <= 1:
        raise ValueError('Bounded constant risk required')
    if estimator is None and constant_risk is None:
        raise ValueError('An inference estimator or explicit constant is required')
    base = type(tracker)

    class RiskMemoryTracker(base):
        def association_cost(self, tracks, detections):
            baseline = super().association_cost(tracks, detections)
            for track in tracks:
                if not isinstance(track, RiskMemoryTrack):
                    if type(track) is not IdentityTrack:
                        raise ValueError('Only ordinary identity tracks may be adapted')
                    track.__class__ = RiskMemoryTrack
                    track.risk_memory_counts = dict(accepted=0, provisional_rejected=0, unavailable_fallback=0)
                    track.risk_memory_accounted = dict(track.risk_memory_counts)
                    track.risk_memory_evidence = {}
                    track.risk_memory_frame = -1
                    track.risk_memory_threshold = self.risk_memory_threshold
                if track.risk_memory_frame != self.frame_id:
                    track.risk_memory_evidence = {}
                    track.risk_memory_frame = self.frame_id
            if not baseline.size or any(d.current_rgb is None for d in detections):
                return baseline
            n, m = baseline.shape
            if self.risk_memory_constant is not None:
                risk = np.full((n, m), self.risk_memory_constant)
                available = np.array([bool(t.identity_history) or t.initial_identity_rgb is not None for t in tracks])
            else:
                past = np.zeros((n, 8, 512), np.float32)
                times = np.zeros((n, 8), np.int64)
                valid = np.zeros((n, 8), bool)
                for i, track in enumerate(tracks):
                    anchor = None if track.initial_identity_rgb is None else (track.initial_identity_frame, track.initial_identity_rgb, track.initial_identity_ir)
                    for j, (frame, rgb, _) in enumerate(anchor_recent_history(list(track.identity_history), anchor, 8)):
                        past[i, j], times[i, j], valid[i, j] = rgb, frame, True
                candidates = np.asarray([d.current_rgb for d in detections], np.float32)
                if candidates.shape != (m, 512) or not np.isfinite(candidates).all():
                    return baseline
                inputs = [torch.from_numpy(v) for v in (past, times, valid)]
                with torch.inference_mode():
                    prototype, available = self.predictor(*inputs, torch.full((n,), self.frame_id, dtype=torch.int64))
                    graph = CandidateGraph(*inputs, torch.from_numpy(candidates),
                        torch.tensor(np.array([t.tlbr for t in tracks]), dtype=torch.float32),
                        torch.tensor(np.array([d.tlbr for d in detections]), dtype=torch.float32),
                        torch.tensor([d.score for d in detections], dtype=torch.float32),
                        torch.from_numpy(baseline.astype(np.float32)), torch.ones(n, m, dtype=torch.bool),
                        torch.tensor([self.frame_id - t.start_frame for t in tracks]),
                        torch.tensor([self.frame_id - t.end_frame for t in tracks]), self.frame_id, prototype=prototype)
                    logits = self.risk_memory_estimator(observable_candidate_features(graph), graph.legal)
                    calibration = self.risk_memory_calibration
                    risk = (logits / calibration['temperature'] + calibration['bias']).sigmoid().numpy()
                    available = available.numpy()
            for i, track in enumerate(tracks):
                if available[i]:
                    for j, detection in enumerate(detections):
                        track.risk_memory_evidence[detection.current_proposal_index] = float(risk[i, j])
            return baseline

        def update(self, *args, **kwargs):
            before = self.tracked_stracks + self.lost_stracks + self.removed_stracks
            output = super().update(*args, **kwargs)
            after = self.tracked_stracks + self.lost_stracks + self.removed_stracks
            unique = {id(track): track for track in before + after}
            for track in unique.values():
                if isinstance(track, RiskMemoryTrack):
                    for key, count in track.risk_memory_counts.items():
                        self.phase_vi_memory_counts[key] += count - track.risk_memory_accounted[key]
                        track.risk_memory_accounted[key] = count
            return output

    tracker.__class__ = RiskMemoryTracker
    tracker.risk_memory_estimator = estimator
    tracker.risk_memory_calibration = calibration
    tracker.risk_memory_threshold = threshold
    tracker.risk_memory_constant = constant_risk
    tracker.phase_vi_memory_counts = dict(accepted=0, provisional_rejected=0, unavailable_fallback=0)
    return tracker
