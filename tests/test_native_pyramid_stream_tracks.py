"""Interleaved streams preserve original ByteTrack states and emitted IDs."""
from types import SimpleNamespace

import numpy as np
import pytest

from tools.eval_mft26ir import write_sequence_results
from tools.native_pyramid_stream_tracks import IndependentByteTrack, StreamTrackWriter, mot_line
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.byte_tracker import BYTETracker


def packets():
    return [np.array(v, np.float32).reshape(-1, 5) for v in [
        [[10, 10, 30, 30, .9], [100, 20, 125, 40, .95]],
        [[12, 10, 32, 30, .85], [98, 20, 123, 40, .92]],
        [[14, 10, 34, 30, .3], [96, 20, 121, 40, .9]],
        [],
        [[18, 10, 38, 30, .9], [92, 20, 117, 40, .9]],
        [[20, 10, 40, 30, .9], [90, 20, 115, 40, .9], [50, 80, 70, 100, .9]],
        [[22, 10, 42, 30, .9], [88, 20, 113, 40, .9], [52, 80, 72, 100, .9]],
    ]]


def states(tracker):
    return [(group, t.track_id, t.state, t.is_activated, t.frame_id, t.start_frame,
             t.mean.copy(), t.covariance.copy()) for group in ['tracked_stracks', 'lost_stracks', 'removed_stracks']
            for t in getattr(tracker, group)]


def assert_states_equal(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x[:6] == y[:6]
        np.testing.assert_array_equal(x[6], y[6])
        np.testing.assert_array_equal(x[7], y[7])


def test_interleaving_matches_unmodified_original_states_without_touching_global_ids(tmp_path):
    prior_count = BaseTrack._count
    try:
        BaseTrack._count = 0
        original = BYTETracker(SimpleNamespace(track_thresh=.6, track_buffer=30, match_thresh=.9, mot20=False), frame_rate=25)
        expected_rows, expected_states = [], []
        for frame, packet in enumerate(packets(), 1):
            result = original.update(packet.copy(), (1, 1), (1, 1))
            expected_rows.append([(frame, t.tlwh.copy(), t.track_id, t.score) for t in result
                                  if t.tlwh[2]*t.tlwh[3] > 100])
            expected_states.append(states(original))
        BaseTrack._count = 137
        independent = [IndependentByteTrack(), IndependentByteTrack()]
        for frame, packet in enumerate(packets(), 1):
            original_packet = packet.copy()
            for tracker in independent:
                rows = tracker.update(packet, frame)
                assert [mot_line(row) for row in rows] == [mot_line(row) for row in expected_rows[frame-1]]
                assert_states_equal(states(tracker.tracker), expected_states[frame-1])
            np.testing.assert_array_equal(packet, original_packet)
            assert BaseTrack._count == 137
    finally:
        BaseTrack._count = prior_count


def test_streamed_mot_files_are_byte_exact_to_original_writer(tmp_path):
    writer = StreamTrackWriter(tmp_path / 'streams', 'A', ['RGB', 'missing'])
    original = IndependentByteTrack()
    rows = []
    for frame, packet in enumerate(packets(), 1):
        rows.extend(original.update(packet, frame))
        writer.append(frame, dict(RGB=packet, missing=packet.copy()))
    report = writer.close(7)
    reference = tmp_path / 'original.txt'
    write_sequence_results(str(reference), rows)
    for row in report.values():
        from pathlib import Path
        assert Path(row['track_path']).read_bytes() == reference.read_bytes()
        assert row['frames'] == 7 and not row['feature_or_detection_disk_cache']
    assert report['RGB']['track_sha256'] == report['missing']['track_sha256']


def test_missing_frame_or_stream_cannot_be_silently_skipped(tmp_path):
    writer = StreamTrackWriter(tmp_path, 'A', ['RGB', 'IR'])
    try:
        with pytest.raises(ValueError): writer.append(2, dict(RGB=packets()[0], IR=packets()[0]))
        with pytest.raises(ValueError): writer.append(1, dict(RGB=packets()[0]))
        writer.append(1, dict(RGB=packets()[0], IR=packets()[0]))
        with pytest.raises(ValueError, match='Incomplete'): writer.close(2)
    finally:
        writer.abort()
    with pytest.raises(ValueError, match='overwriting'):
        StreamTrackWriter(tmp_path, 'A', ['RGB'])
