from types import SimpleNamespace

import numpy as np

from yolox.tracker.basetrack import BaseTrack
from yolox.tracker import matching
from yolox.tracker.semantic_physical_scores import geometry_reference_scores, SemanticPhysicalScoreTracker
from test_foreground_tracker import with_transport, run_original, assert_same


def test_geometry_score_reference_is_current_one_to_one_and_keeps_unmatched():
    boxes = np.array([[0, 0, 100, 100], [1, 0, 101, 100], [200, 0, 300, 100]], np.float32)
    scores = np.array([.9, .95, .85], np.float32)
    original_scores = scores.copy()
    references = np.array([[0, 0, 100, 100], [200, 0, 300, 100]], np.float32)
    values, mask = geometry_reference_scores(boxes, scores, references, np.array([.7, .5], np.float32))
    assert mask.sum() == 1 and not mask[2]
    np.testing.assert_array_equal(values[~mask], scores[~mask])
    np.testing.assert_array_equal(scores, original_scores)
    np.testing.assert_array_equal(boxes, np.array([[0, 0, 100, 100], [1, 0, 101, 100], [200, 0, 300, 100]], np.float32))


def test_disabled_or_equal_scores_preserve_full_original_lifecycle():
    boxes = []; scores = []; offsets = [0]
    for frame in range(1, 66):
        for identity in range(3):
            if identity == 1 and 15 <= frame <= 46:
                continue
            if identity == 2 and 20 <= frame <= 23:
                continue
            x = 20+identity*60+.3*frame
            boxes.append([x, 20, x+25, 40]); scores.append(.4 if frame % 7 == 0 else .95)
        offsets.append(len(boxes))
    data = with_transport(boxes, scores, offsets); expected = run_original(data, 65)
    for enabled in [False, True]:
        data['association_scores'] = data['scores'].copy() if enabled else np.zeros_like(data['scores'])
        BaseTrack._count = 0
        tracker = SemanticPhysicalScoreTracker(dict(separate_association_score=enabled))
        actual = [tracker.update(frame, data) for frame in range(1, 66)]
        assert_same(expected, actual)


def test_physical_reference_resolves_semantic_score_reversal_without_mutation():
    tracks = [SimpleNamespace(tlbr=np.array(b, np.float32)) for b in [[0, 0, 100, 100], [10, 0, 110, 100]]]
    observations = [SimpleNamespace(tlbr=np.array(b, np.float32), score=score, evidence=dict(association_score=ref))
                    for b, score, ref in [([-5, 0, 95, 100], .65, .95), ([-20, 0, 80, 100], .95, .65)]]
    base = matching.fuse_score(matching.iou_distance(tracks, observations), observations); original = base.copy()
    assert base.trace() > base[0, 1]+base[1, 0]
    tracker = SemanticPhysicalScoreTracker({})
    result = tracker.rank_cost(tracks, observations, base, 2, 'primary')
    assert result.trace() < result[0, 1]+result[1, 0]
    np.testing.assert_array_equal(base, original)
    assert tracker.rank_cost(tracks, observations, base, 2, 'secondary') is base


def test_future_scores_and_boxes_do_not_change_active_prefix():
    boxes = [[x+f, 20, x+f+30, 40] for f in range(6) for x in [20, 100]]
    data = with_transport(boxes, [.8]*12, np.arange(0, 13, 2))
    data['association_scores'] = np.full(12, .95, np.float32)
    other = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in data.items()}
    other['boxes'][8:] = -1000; other['scores'][8:] = .1; other['association_scores'][8:] = 0.
    outputs = []
    for packet in [data, other]:
        BaseTrack._count = 0; tracker = SemanticPhysicalScoreTracker({})
        outputs.append([tracker.update(frame, packet) for frame in range(1, 5)])
        assert tracker.statistics['physical_score_columns'] > 0
    assert_same(*outputs)
