"""Original ByteTrack decisions with stream-local IDs and bounded track writes.

No weights, feature cache, image copy, selection, official metrics or GPU launch.
Local counters prevent interleaved model streams from changing each other's IDs.
They start at1 for each video; no emitted row is renumbered later.
"""
import hashlib
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

from tools.run_pseudo_ir_comparison import sha256
from yolox.tracker.byte_tracker import BYTETracker, STrack


class StreamSTrack(STrack):
    def __init__(self, tlwh, score, counter):
        super().__init__(tlwh, score)
        self.identity_counter = counter

    def next_id(self):
        self.identity_counter[0] += 1
        return self.identity_counter[0]


class IndependentByteTrack:
    def __init__(self):
        self.counter = [0]
        self.tracker = BYTETracker(SimpleNamespace(track_thresh=.6, track_buffer=30,
            match_thresh=.9, mot20=False), frame_rate=25,
            track_factory=lambda tlwh, score, evidence: StreamSTrack(tlwh, score, self.counter))

    def update(self, detections, frame):
        if frame != self.tracker.frame_id + 1:
            raise ValueError('Every ordered frame, including empty observations, must be delivered')
        if (detections.ndim != 2 or detections.shape[1] != 5 or
                not np.isfinite(detections).all() or
                ((detections[:, 4] < 0) | (detections[:, 4] > 1)).any()):
            raise ValueError('Finite source-coordinate boxes and probabilities are required')
        # Original tracker divides boxes in place; retain caller-owned packets.
        result = self.tracker.update(detections.copy().astype(np.float32), (1, 1), (1, 1))
        return [(frame, track.tlwh.copy(), track.track_id, track.score) for track in result
                if track.tlwh[2] * track.tlwh[3] > 100]


def mot_line(row):
    frame, tlwh, identity, score = row
    x, y, width, height = map(float, tlwh)
    return (f'{frame},{identity},{x:.3f},{y:.3f},{width:.3f},{height:.3f},'
            f'{float(score):.6f},-1,-1,-1\n')


class StreamTrackWriter:
    """Keep only current packets in RAM; write the final causal track prefix."""
    def __init__(self, directory, sequence, model_ids):
        self.directory, self.sequence = Path(directory), sequence
        ids = list(model_ids)
        if (not ids or len(ids) != len(set(ids)) or
                any(not re.fullmatch(r'[A-Za-z0-9_-]+', value) for value in [sequence, *ids])):
            raise ValueError('Unique safe registered stream and video IDs required')
        self.frames, self.trackers, self.handles, self.paths = 0, {}, {}, {}
        self.packet_hashes, self.detection_counts, self.track_rows = {}, {}, {}
        paths = {name: self.directory / name / 'track_results' / (sequence+'.txt') for name in ids}
        if any(path.exists() for path in paths.values()):
            raise ValueError('Preserve prior causal outputs instead of overwriting them')
        for name, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            self.paths[name], self.handles[name] = path, path.open('x')
            self.trackers[name], self.packet_hashes[name] = IndependentByteTrack(), hashlib.sha256()
            self.detection_counts[name] = self.track_rows[name] = 0

    def append(self, frame, packets):
        if frame != self.frames + 1 or set(packets) != set(self.handles):
            raise ValueError('Each current frame requires all registered model observations')
        for name, values in packets.items():
            rows = self.trackers[name].update(values, frame)
            for row in rows: self.handles[name].write(mot_line(row))
            array = np.asarray(values, dtype='<f4')
            self.packet_hashes[name].update(np.asarray([frame, len(array)], dtype='<i8').tobytes())
            self.packet_hashes[name].update(array.tobytes())
            self.detection_counts[name] += len(array)
            self.track_rows[name] += len(rows)
        self.frames = frame

    def flush(self):
        for handle in self.handles.values(): handle.flush()

    def close(self, expected_frames):
        self.flush()
        for handle in self.handles.values(): handle.close()
        if self.frames != expected_frames:
            raise ValueError('Incomplete video retained; no complete inference manifest may be emitted')
        return {name: dict(sequence=self.sequence, frames=self.frames,
            track_path=str(self.paths[name]), track_sha256=sha256(self.paths[name]),
            detection_packet_sha256=self.packet_hashes[name].hexdigest(),
            detections=self.detection_counts[name], track_rows=self.track_rows[name],
            identity_scope='Sequence-local IDs from1; isolated at emission, no later renumbering.',
            original_ByteTrack_association_unchanged=True, feature_or_detection_disk_cache=False)
            for name in self.handles}

    def abort(self):
        for handle in self.handles.values():
            if not handle.closed: handle.close()
