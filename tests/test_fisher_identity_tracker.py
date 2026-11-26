"""Small causal/association regressions; no images, GPU or trained weights."""
import json
from pathlib import Path

import numpy as np
import pytest

from tools.mft_identity_projection import fit_fisher, transform
from tools.native_pyramid_stream_tracks import IndependentByteTrack, mot_line
from yolox.tracker.fisher_identity_tracker import FisherIdentityTracker, FisherTrack


def config():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root/'configs/perceptual_fisher_memory_20261127.json').read_text())['association']


def test_zero_weight_literal_original_and_no_descriptor_read():
    original = IndependentByteTrack()
    candidate = FisherIdentityTracker(0., 25, config())
    class Unreadable:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Zero weight must not read descriptor evidence')
    for frame in range(1, 8):
        packet = np.array([[frame, 0, frame+20, 10, .95], [80-frame, 5, 100-frame, 15, .9]], np.float32)
        untouched = packet.copy()
        expected = ''.join(map(mot_line, original.update(packet, frame)))
        tracks = candidate.update('video', frame, packet, Unreadable())
        actual = [(frame, t.tlwh.copy(), t.track_id, t.score)for t in tracks if t.tlwh[2]*t.tlwh[3] > 100]
        assert ''.join(map(mot_line, actual)) == expected
        np.testing.assert_array_equal(packet, untouched)


def test_discriminative_appearance_reverses_ambiguous_geometric_assignment():
    tracker = FisherIdentityTracker(.5, 250, config())
    tracker.reset('video')
    past = [FisherTrack([0, 0, 20, 10], .95, None, [0], config()),
            FisherTrack([12, 0, 20, 10], .95, None, [0], config())]
    for i, track in enumerate(past):
        track.memory, track.memory_frame = np.eye(2, dtype=np.float32)[i], 1
    current = [FisherTrack([1, 0, 20, 10], .95, dict(vector=np.array([0, 1], np.float32), reliable=True), [0], config()),
               FisherTrack([11, 0, 20, 10], .95, dict(vector=np.array([1, 0], np.float32), reliable=True), [0], config())]
    base = np.array([[.4, .42], [.42, .4]], np.float32)
    changed = tracker.cost(past, current, base, 2, 'primary')
    assert changed[0, 1] < changed[0, 0] and changed[1, 0] < changed[1, 1]
    assert tracker.statistics['actual_float_cost_entries_changed'] == 4
    assert tracker.statistics['frames_with_changed_primary_assignment'] == 1
    np.testing.assert_array_equal(base, np.array([[.4, .42], [.42, .4]], np.float32))
    with pytest.raises(AssertionError, match='past memory'):
        tracker.cost(past, current, base, 1, 'primary')


def test_unreliable_observation_cannot_overwrite_memory():
    track = FisherTrack([0, 0, 20, 10], .95, None, [0], config())
    track.accept(dict(vector=np.array([1, 0], np.float32), reliable=True), 1)
    track.accept(dict(vector=np.array([0, 1], np.float32), reliable=False), 2)
    np.testing.assert_array_equal(track.memory, np.array([1, 0], np.float32))
    assert track.memory_frame == 1


def test_fisher_uses_video_local_identity_classes_and_shared_transform():
    x = np.array([[1, 0, .01], [1, .02, 0], [0, 1, .01], [.02, 1, 0],
                  [1, 0, -.01], [1, -.02, 0], [0, 1, -.01], [-.02, 1, 0]], float)
    videos = np.repeat([0, 1], 4)
    first, details = fit_fisher(x, videos, np.tile([1, 1, 2, 2], 2), dimensions=2)
    renamed, _ = fit_fisher(x, videos, np.array([100, 100, 500, 500, 7, 7, 8, 8]), dimensions=2)
    np.testing.assert_allclose(first['projection'], renamed['projection'])
    assert details['video_local_identity_classes'] == 4
    result = transform(x, first)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-6)
    assert result[0]@result[1] > result[0]@result[2]
