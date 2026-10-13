"""Current/past full-set head on the frozen native GRU lifecycle.

No training labels, physical identity bank or future frame are imported here.
Only high-confidence association is changed; later stages and writes remain
native. NULL means no match to the supplied confirmed/lost high-stage set.
"""
import numpy as np
import torch

from .history import anchor_recent_history
from .joint_identity_decoder import bounded_joint_cost, decode_joint_assignment
from yolox.tracker import matching


def install_joint_identity(tracker, model, route='bounded', maximum_delta=.03, dictionary_seed=0):
    if route not in ('bounded', 'full') or not 0 <= maximum_delta <= .1:
        raise ValueError('Declared bounded/full joint route required')
    if model is None:
        return tracker
    if getattr(tracker, 'history_view', None) != 'anchor_recent' or getattr(tracker, 'predictor', None) is None:
        raise ValueError('Matched ordinary GRU anchor/recent producer required')
    model.eval()
    base = type(tracker)

    class JointTracker(base):
        def association_cost(self, tracks, detections):
            baseline = super().association_cost(tracks, detections)
            if (not baseline.size or any(not t.is_activated for t in tracks)
                    or any(d.score < self.args.track_thresh or d.current_rgb is None for d in detections)):
                return baseline
            n, m = baseline.shape
            if n > self.joint_model.capacity:
                raise ValueError('Full-set capacity exceeded; no silent truncation/fallback')
            dimension = self.joint_model.dimension
            past = np.zeros((n, 8, dimension), np.float32)
            times, valid = np.zeros((n, 8), np.int64), np.zeros((n, 8), bool)
            for i, track in enumerate(tracks):
                anchor = None if track.initial_identity_rgb is None else (
                    track.initial_identity_frame, track.initial_identity_rgb, track.initial_identity_ir)
                for j, (frame, rgb, _) in enumerate(anchor_recent_history(list(track.identity_history), anchor, 8)):
                    past[i, j], times[i, j], valid[i, j] = rgb, frame, True
            cpu_inputs = [torch.from_numpy(v) for v in (past, times, valid, np.full(n, self.frame_id, np.int64))]
            current = torch.tensor(np.asarray([d.current_rgb for d in detections]), dtype=torch.float32)
            raw = matching.iou_distance(tracks, detections).astype(np.float32)
            state = torch.tensor([[np.log1p(self.frame_id - t.start_frame) / np.log(15001),
                                   np.log1p(self.frame_id - t.end_frame) / np.log(15001), float(t.state == 2)]
                                  for t in tracks], dtype=torch.float32)
            device = next(self.joint_model.parameters()).device
            with torch.inference_mode():
                prototype = self.predictor(*cpu_inputs)[0]
                appearance = ((1 - prototype @ current.T) / 2).clamp(0, 1)
                inputs = dict(history=cpu_inputs[0], times=cpu_inputs[1], valid=cpu_inputs[2], frame=cpu_inputs[3],
                    detections=current, detection_valid=torch.ones(m, dtype=torch.bool), geometry=torch.from_numpy(raw),
                    appearance=appearance, track_state=state, confidence=torch.tensor([d.score for d in detections], dtype=torch.float32),
                    codes=torch.from_numpy(self.joint_rng.permutation(self.joint_model.capacity)[:n].astype(np.int64)))
                output = self.joint_model(**{k: v.to(device) for k, v in inputs.items()})
                if self.joint_route == 'bounded':
                    delta = (bounded_joint_cost(torch.zeros(n, m, device=device), output['residual'], output['legal'],
                                               self.joint_maximum_delta)).cpu().numpy().astype(np.float64)
                    result = baseline + delta
                    self.joint_counts['changed_edges'] += int(np.count_nonzero(delta))
                else:
                    selected = decode_joint_assignment(output['logits'], inputs['detection_valid'].to(device)).cpu().numpy()
                    result = np.full_like(baseline, 1.01)
                    for detection, track in enumerate(selected):
                        if track >= 0:
                            result[track, detection] = 0.
                    self.joint_counts['null_detections'] += int((selected < 0).sum())
            self.joint_counts['scored_frames'] += 1
            self.joint_counts['scored_tracks'] += n
            return result

    tracker.__class__ = JointTracker
    tracker.joint_model, tracker.joint_route = model, route
    tracker.joint_maximum_delta = maximum_delta
    tracker.joint_rng = np.random.default_rng(dictionary_seed)
    tracker.joint_counts = dict(scored_frames=0, scored_tracks=0, changed_edges=0, null_detections=0)
    return tracker
