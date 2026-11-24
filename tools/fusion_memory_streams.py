"""Causal F4 final MOT writer, independent IDs and no proposal/feature disk cache."""
import hashlib
from pathlib import Path
import re

import numpy as np

from tools.native_pyramid_stream_tracks import mot_line


class FusionMemoryStreamWriter:
    def __init__(self, directory, sequence, case_ids):
        ids = list(case_ids)
        if (not ids or len(ids) != len(set(ids)) or
                any(not re.fullmatch(r'[A-Za-z0-9_-]+', s) for s in [sequence, *ids])):
            raise ValueError('Unique safe case/sequence IDs required')
        self.sequence, self.frames = sequence, 0
        self.paths = {name: Path(directory)/name/'track_results'/(sequence+'.txt') for name in ids}
        if any(path.exists() for path in self.paths.values()):
            raise ValueError('Preserve previously emitted tracks')
        self.handles, self.packet_hashes, self.track_hashes = {}, {}, {}
        self.detections, self.rows = {}, {}
        self.failed = self.closed = False
        try:
            for name, path in self.paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                self.handles[name] = path.open('x')
                self.packet_hashes[name], self.track_hashes[name] = hashlib.sha256(), hashlib.sha256()
                self.detections[name] = self.rows[name] = 0
        except Exception:
            self.abort()
            raise

    def append(self, frame, rows, packets):
        if (self.closed or self.failed or frame != self.frames+1 or
                set(rows) != set(self.paths) or set(packets) != set(self.paths)):
            raise ValueError('Each ordered current frame must cover every registered case')
        # Validate the whole current frame before writing any stream.
        lines, arrays = {}, {}
        for name in self.paths:
            array = np.asarray(packets[name], dtype='<f4')
            if (array.ndim != 2 or array.shape[1] != 5 or not np.isfinite(array).all() or
                    ((array[:, 4] < 0) | (array[:, 4] > 1)).any() or
                    (array[:, 2:4] <= array[:, :2]).any()):
                raise ValueError('Finite unchanged current source-coordinate packets required')
            if any(row[0] != frame or row[2] <= 0 or not np.isfinite(row[1]).all() or
                    row[1][2]*row[1][3] <= 100 or not 0 <= row[3] <= 1 for row in rows[name]):
                raise ValueError('Invalid emitted MOT row')
            if len({row[2] for row in rows[name]}) != len(rows[name]):
                raise ValueError('One track identity per emitted frame')
            lines[name], arrays[name] = ''.join(mot_line(row) for row in rows[name]), array
        try:
            for name, array in arrays.items():
                self.handles[name].write(lines[name])
                self.track_hashes[name].update(lines[name].encode())
                self.packet_hashes[name].update(np.asarray([frame, len(array)], dtype='<i8').tobytes())
                self.packet_hashes[name].update(array.tobytes())
                self.detections[name] += len(array)
                self.rows[name] += len(rows[name])
            self.frames = frame
        except Exception:
            self.failed = True
            self.abort()
            raise

    def close(self, expected_frames):
        if self.closed or self.failed:
            raise ValueError('Closed/failed writer cannot emit a complete receipt')
        self.abort()
        if self.frames != expected_frames:
            raise ValueError('Retain incomplete prefix; do not emit a complete receipt')
        for name, path in self.paths.items():
            actual = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    actual.update(block)
            if actual.hexdigest() != self.track_hashes[name].hexdigest():
                raise ValueError('Written MOT bytes differ from the emitted prefix')
        return {name: dict(sequence=self.sequence, frames=self.frames,
            track_path=str(self.paths[name]), track_sha256=self.track_hashes[name].hexdigest(),
            detection_packet_sha256=self.packet_hashes[name].hexdigest(),
            detections=self.detections[name], track_rows=self.rows[name],
            identity_scope='Sequence-local from1, no later renumbering',
            feature_or_detection_disk_cache=False) for name in self.paths}

    def abort(self):
        for handle in self.handles.values():
            if not handle.closed:
                handle.close()
        self.closed = True
