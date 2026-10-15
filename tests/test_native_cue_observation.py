"""The native cue observer must preserve outputs and record only strict-past rows."""

import gzip
import json
from types import SimpleNamespace

import numpy as np

from cfp.native_cue_observation import close_native_cue_observer, install_native_cue_observer
from cfp.predictive_association import PredictiveAssociationTracker
from yolox.tracker.basetrack import BaseTrack


ARGS = SimpleNamespace(track_thresh=.58, track_buffer=45, match_thresh=.97, mot20=False)


def _signature(output):
    return [(int(track.track_id), track.tlwh.copy(), float(track.score)) for track in output]


def test_observer_keeps_native_output_and_strict_past_history(tmp_path):
    frames = [
        np.array([[0, 0, 30, 20, .9], [60, 0, 90, 20, .8]], np.float32),
        np.array([[1, 0, 31, 20, .9], [59, 0, 89, 20, .8]], np.float32),
        np.array([[2, 0, 32, 20, .9]], np.float32),
    ]
    features = np.array([[1., 0.], [0., 1.]], np.float32)
    BaseTrack._count = 0
    reference = PredictiveAssociationTracker(ARGS, memory="fifo")
    expected = []
    for frame in frames:
        expected.append(_signature(reference.update(frame, (1, 1), (1, 1),
                                                    features[:len(frame)])))

    BaseTrack._count = 0
    observed = PredictiveAssociationTracker(ARGS, memory="fifo")
    path = tmp_path / "cue.jsonl.gz"
    install_native_cue_observer(observed, path)
    actual = []
    for frame in frames:
        actual.append(_signature(observed.update(frame, (1, 1), (1, 1),
                                                  features[:len(frame)],
                                                  proposal_indices=np.arange(len(frame)))) )
    close_native_cue_observer(observed)
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert g[0] == w[0] and np.array_equal(g[1], w[1]) and g[2] == w[2]
    with gzip.open(path, "rt") as stream:
        events = [json.loads(line) for line in stream]
    assert events and {event["frame"] for event in events} == {1, 2, 3}
    assert all(all(frame < event["frame"] for frame, _ in row["history"])
               for event in events for row in event["rows"])
