from types import SimpleNamespace

import numpy as np
import pytest

from cfp.association import AppearanceTracker
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.basetrack import BaseTrack


ARGS = SimpleNamespace(track_thresh=.58, track_buffer=45, match_thresh=.97, mot20=False)


def sequence_trace(tracker, frames, features=False):
    result = []
    for frame in frames:
        rgb = np.ones((len(frame), 2), np.float32)
        targets = tracker.update(frame.copy(), (1, 1), (1, 1), rgb) if features else tracker.update(frame.copy(), (1, 1), (1, 1))
        result.append([(target.track_id, *target.tlwh.tolist(), float(target.score)) for target in targets])
    return result


def test_disabled_and_zero_appearance_weight_exactly_preserve_lifecycle():
    frames = [np.array([[t * 2, 0, 30 + t * 2, 20, .9], [80 - t * 2, 0, 110 - t * 2, 20, .8 if t != 2 else .3]], np.float32) for t in range(5)]
    frames.insert(3, np.empty((0, 5), np.float32))
    BaseTrack._count = 0
    reference = sequence_trace(BYTETracker(ARGS, 25), frames)
    BaseTrack._count = 0
    assert reference == sequence_trace(AppearanceTracker(ARGS, mode='off'), frames)
    BaseTrack._count = 0
    assert reference == sequence_trace(AppearanceTracker(ARGS, appearance_weight=0), frames, features=True)


def test_identity_write_follows_scoring_and_missing_ir_is_exact_rgb_fallback():
    BaseTrack._count = 0
    tracker = AppearanceTracker(ARGS, mode='fixed')
    first = np.array([[0, 0, 30, 20, .9]], np.float32)
    targets = tracker.update(first, (1, 1), (1, 1), np.array([[1., 0.]], np.float32))
    track = targets[0]
    assert track.last_identity_frame == 1 and np.array_equal(track.memory_rgb, [1., 0.])
    second = np.array([[1, 0, 31, 20, .9]], np.float32)
    targets = tracker.update(second, (1, 1), (1, 1), np.array([[0., 1.]], np.float32))
    assert targets[0].track_id == track.track_id
    assert track.last_identity_frame == 2 and len(track.identity_history) == 2
    assert np.allclose(track.memory_rgb, np.array([.9, .1]) / np.linalg.norm([.9, .1]))
    with pytest.raises(ValueError, match='once per frame'):
        track.pending = True
        track.write_identity(2, .9)
    # Absent IR produces exactly the same original-coordinate outputs as RGB.
    frames = [first, second]
    BaseTrack._count = 0
    rgb = sequence_trace(AppearanceTracker(ARGS, mode='rgb'), frames, True)
    BaseTrack._count = 0
    assert rgb == sequence_trace(AppearanceTracker(ARGS, mode='fixed'), frames, True)


def test_unextractable_crops_preserve_detector_proposals_and_geometry():
    BaseTrack._count = 0
    tracker = AppearanceTracker(ARGS)
    detection = np.array([[0, 0, 30, 20, .9]], np.float32)
    output = tracker.update(detection.copy(), (1, 1), (1, 1), np.zeros((1, 2), np.float32))
    assert len(output) == 1 and output[0].memory_rgb is None
