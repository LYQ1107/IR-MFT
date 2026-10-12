"""Verify adapters preserve native decisions, temporal order and observer purity."""
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cfp.lifecycle_crossover import FrameFeatures, load_bot_reference, make_bot_lifecycle, make_cfp_lifecycle, DecisionRecorder
from cfp.predictive_association import PredictiveAssociationTracker
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker import matching


ROOT = Path('/data3/liuyeqiang/CFP-FishMOT_references/BoT-SORT')


class Last(torch.nn.Module):
    def forward(self, features, times, valid, frame):
        assert not torch.any(valid & (times >= frame[:, None]))
        return features[torch.arange(len(features)), valid.sum(1) - 1], valid.any(1)


def parameters():
    values = json.loads((Path(__file__).parents[1] / 'configs/cfp/external_association_v1.json').read_text())['bot_sort']
    values.pop('parameter_source')
    values.pop('appearance_thresholds')
    return values


def frames():
    return [np.array([[0, 0, 30, 20, .9]], np.float32),
            np.array([[1, 0, 31, 20, .4], [100, 0, 130, 20, .9]], np.float32),
            np.array([[2, 0, 32, 20, .9]], np.float32),
            np.empty((0, 5), np.float32),
            np.array([[3, 0, 33, 20, .9]], np.float32)]


def signature(output):
    return [(int(t.track_id), t.tlwh.copy(), float(t.score), bool(t.is_activated)) for t in output]


def assert_equal(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x[0] == y[0] and np.array_equal(x[1], y[1]) and x[2:] == y[2:]


def test_frame_feature_mapping_includes_low_proposals_without_reordering_high():
    provider = FrameFeatures()
    proposals = frames()[1]
    provider.set_frame(proposals, np.eye(2, dtype=np.float32), np.array([7, 8]))
    assert provider.lookup([1, 0, 30, 20], proposals[0, 4])[1] == 7
    assert np.array_equal(provider.inference(None, proposals[1:, :4]), np.eye(2, dtype=np.float32)[1:])
    with pytest.raises(ValueError):
        provider.set_frame(np.repeat(proposals[:1], 2, 0), np.eye(2), np.array([7, 8]))


@pytest.mark.skipif(not ROOT.exists(), reason='Pinned reference absent')
def test_bot_diagonal_native_lifecycle_cost_and_low_memory_parity():
    native = load_bot_reference(ROOT)
    values = parameters()
    native_tracker = native.BoTSORT(SimpleNamespace(**values, appearance_thresh=.15), frame_rate=25)
    expected = []
    for frame in frames():
        feature = np.repeat(np.array([[1., 0.]], np.float32), len(frame), 0)
        native_tracker.encoder.set_frame(frame, feature, np.arange(len(frame)))
        expected.append(signature(native_tracker.update(frame.copy(), None)))
    adapted = make_bot_lifecycle(native, values, 'bot')
    for i, frame in enumerate(frames()):
        feature = np.repeat(np.array([[1., 0.]], np.float32), len(frame), 0)
        assert_equal(signature(adapted.update(frame, feature, np.arange(len(frame)))), expected[i])
        if i == 1:
            assert adapted.tracked_stracks[0].last_identity_frame == 2
            assert adapted.tracked_stracks[0].identity_proposals[-1] == (2, 0)


@pytest.mark.skipif(not ROOT.exists(), reason='Pinned reference absent')
def test_cfp_gru_diagonal_and_observer_have_exact_native_outputs(tmp_path):
    native = load_bot_reference(ROOT)
    args = SimpleNamespace(track_thresh=.58, track_buffer=45, match_thresh=.97, mot20=False)
    BaseTrack._count = 0
    original = PredictiveAssociationTracker(args, predictor=Last(), history_view='anchor_recent')
    expected = []
    for i, frame in enumerate(frames()):
        feature = np.repeat(np.array([[1., 0.]], np.float32), len(frame), 0)
        expected.append(signature(original.update(frame.copy(), (1, 1), (1, 1), feature, proposal_indices=np.arange(len(frame)))))
    BaseTrack._count = 0
    adapted = make_cfp_lifecycle(native, 'gru', Last())
    path = tmp_path / 'decisions.gz'
    recorder = DecisionRecorder(path, adapted)
    try:
        for i, frame in enumerate(frames()):
            feature = np.repeat(np.array([[1., 0.]], np.float32), len(frame), 0)
            with recorder.capture([native.matching, matching]):
                output = adapted.update(frame, feature, np.arange(len(frame)))
            assert_equal(signature(output), expected[i])
    finally:
        recorder.close()
    with gzip.open(path, 'rt') as stream:
        records = [json.loads(line) for line in stream]
    assert len(records) == 15
    assert [x['stage'] for x in records[:3]] == ['high', 'low', 'unconfirmed']
    assert all(t['last_identity_frame'] < row['frame'] for row in records for t in row['tracks'])


@pytest.mark.skipif(not ROOT.exists(), reason='Pinned reference absent')
@pytest.mark.parametrize('association,lifecycle', [('bot', 'cfp'), ('gru', 'bot')])
def test_crossed_cells_keep_unique_online_ids_and_strict_past_history(association, lifecycle):
    native = load_bot_reference(ROOT)
    BaseTrack._count = 0
    tracker = make_bot_lifecycle(native, parameters(), association, Last()) if lifecycle == 'bot' else make_cfp_lifecycle(native, association, Last())
    for frame in frames():
        feature = np.repeat(np.array([[1., 0.]], np.float32), len(frame), 0)
        output = tracker.update(frame, feature, np.arange(len(frame)))
        assert len({int(t.track_id) for t in output}) == len(output)
        for track in tracker.tracked_stracks:
            assert track.last_identity_frame <= tracker.frame_id
            assert len(track.identity_history) <= 8

