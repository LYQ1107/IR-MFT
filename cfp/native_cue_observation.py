"""Read-only native association observation with no physical labels online."""

import gzip
import json

import numpy as np
import torch

from .predictive import bounded_identity_prototype
from yolox.tracker import matching


def install_native_cue_observer(tracker, path, history_length=4, maximum_history_age=1500):
    """Observe the exact FIFO association graph while preserving its output."""
    if tracker.frame_id != 0 or not 1 <= history_length <= 8 or maximum_history_age <= 0:
        raise ValueError("Observer must be installed before the first frame")
    base = type(tracker)
    stream = gzip.open(path, "wt", compresslevel=3)

    class ObservedTracker(base):
        def association_cost(self, tracks, detections):
            result = super().association_cost(tracks, detections)
            raw = matching.iou_distance(tracks, detections)
            geometry = matching.fuse_score(raw, detections) if not self.args.mot20 else raw
            if result.shape != geometry.shape:
                raise ValueError("Observed baseline cost shape differs from geometry")
            rows = []
            for row, track in enumerate(tracks):
                history = [(int(frame), None if index is None else int(index))
                           for frame, index in track.identity_proposals]
                if any(frame >= self.frame_id for frame, _ in history):
                    raise ValueError("Current identity proposal entered native observation")
                # The registered native FIFO route uses the last four entries of
                # this strictly-past history. Recompute the prototype only for
                # the audit; the tracker has already produced the unchanged cost.
                selected = [(frame, index) for frame, index in history
                            if self.frame_id - maximum_history_age <= frame][-history_length:]
                appearance = np.ones(len(detections), dtype=np.float64)
                available = bool(selected)
                if selected:
                    values = np.zeros((1, 8, len(detections[0].current_rgb)), np.float32) if detections else None
                    if values is None:
                        available = False
                    else:
                        times = np.zeros((1, 8), np.int64)
                        valid = np.zeros((1, 8), bool)
                        for slot, (frame, index) in enumerate(selected):
                            if index is None:
                                available = False
                                break
                            # The actual current-coordinate bank is not available
                            # in this module. The tracker history stores the exact
                            # normalized vectors, so use those values directly.
                            value = None
                            for past_frame, past_rgb, _ in track.identity_history:
                                if int(past_frame) == frame:
                                    value = past_rgb
                            if value is None:
                                available = False
                                break
                            values[0, slot] = value
                            times[0, slot] = frame
                            valid[0, slot] = True
                        if available:
                            with torch.inference_mode():
                                prototype, observed = bounded_identity_prototype(
                                    torch.from_numpy(values), torch.from_numpy(times),
                                    torch.from_numpy(valid), torch.tensor([self.frame_id]), mode="fifo")
                            available = bool(observed[0])
                            if available:
                                p = prototype[0].numpy()
                                for col, detection in enumerate(detections):
                                    if detection.current_rgb is not None:
                                        appearance[col] = float(np.clip((1.0 - p @ detection.current_rgb) / 2.0, 0.0, 1.0))
                rows.append({
                    "track_id": int(track.track_id),
                    "state": int(track.state),
                    "end_frame": int(track.end_frame),
                    "history": history,
                    "available": available,
                    "proposals": [int(d.current_proposal_index) for d in detections],
                    "raw_geometry": raw[row].astype(np.float32).tolist(),
                    "geometry": geometry[row].astype(np.float32).tolist(),
                    "appearance": appearance.astype(np.float32).tolist(),
                    "baseline": result[row].astype(np.float32).tolist(),
                })
            stream.write(json.dumps({"frame": int(self.frame_id), "rows": rows},
                                    separators=(",", ":"), allow_nan=False) + "\n")
            return result

    tracker.__class__ = ObservedTracker
    tracker.native_cue_observation_stream = stream
    tracker.native_cue_observation_rows = 0
    return tracker


def close_native_cue_observer(tracker):
    stream = getattr(tracker, "native_cue_observation_stream", None)
    if stream is not None:
        stream.close()
