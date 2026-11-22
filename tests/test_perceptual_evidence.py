import cv2
import numpy as np

from yolox.tracker.perceptual_features import (
    foreground, hog, letterbox_tensor_array, local_transport,
)
from yolox.tracker.perceptual_tracker import PerceptualTracker


def test_local_transport_recovers_known_translation_and_rejects_empty_frame():
    rng = np.random.default_rng(9)
    previous = np.zeros((120, 160), np.uint8)
    previous[25:85, 35:115] = rng.integers(40, 250, (60, 80), dtype=np.uint8)
    current = cv2.warpAffine(previous, np.float32([[1, 0, 5], [0, 1, 3]]), (160, 120))
    boxes = np.array([[35, 25, 115, 85]], np.float32)
    result = local_transport(previous, current, boxes, previous.shape)
    np.testing.assert_allclose(result[0, :2], [5, 3], atol=.4)
    assert result[0, 2] > .7
    missing = local_transport(np.zeros_like(previous), np.zeros_like(previous), boxes, previous.shape)
    assert not missing.any()


def test_shape_contrast_is_polarity_independent_and_finite():
    gray = np.full((32, 64), 30, np.uint8)
    cv2.ellipse(gray, (32, 16), (19, 7), 15, 0, 360, 180, -1)
    first, quality = foreground(gray)
    inverse, inverse_quality = foreground(255 - gray)
    np.testing.assert_allclose(first, inverse, atol=1e-5)
    np.testing.assert_allclose(quality, inverse_quality, atol=1e-5)
    assert first[16, 32] > first[0, 0]
    assert np.isfinite(hog(gray)).all()


def test_empty_saliency_retains_finite_mean_padded_crop():
    image = np.full((50, 100, 3), 100, np.uint8)
    empty, quality = foreground(np.full((25, 50), 80, np.uint8))
    assert quality == 0
    crop = letterbox_tensor_array(image, [-20, 10, 80, 40], empty)
    assert crop.shape == (3, 128, 256)
    assert np.isfinite(crop).all()


def crossing_data():
    centers = [(30, 70), (45, 55), (60, 40), (75, 25)]
    boxes = np.array([[x-10, 20, x+10, 40] for frame in centers for x in frame], np.float32)
    appearance = np.tile(np.eye(2, dtype=np.float32), (4, 1))
    return dict(boxes=boxes, scores=np.full(8, .95), offsets=np.arange(0,9,2),
                embedding=appearance, ir_mask_embedding=appearance,
                rgb_mask_embedding=appearance, ir_hog=appearance,
                rgb_hog=appearance, color=appearance,
                flow_rgb=np.tile([[15,0,1,0],[-15,0,1,0]],(4,1)),
                flow_ir=np.tile([[15,0,1,0],[-15,0,1,0]],(4,1)))


def test_appearance_keeps_unique_identity_through_crossing():
    data = crossing_data()
    tracker = PerceptualTracker(dict(appearance_weight=.65, color_weight=0,
                                    appearance_scale=.2, maximum_center_distance=3))
    outputs = [tracker.update(f, data) for f in range(1,5)]
    assert all(len({r[2] for r in out})==2 for out in outputs)
    last = {row[2]:row[1][0]+row[1][2]/2 for row in outputs[-1]}
    assert last[1] > last[2]


def test_future_evidence_does_not_change_already_emitted_prefix():
    first = crossing_data()
    second = {k:v.copy() for k,v in first.items()}
    # At frame 2 only flow from source frame 1 may be used. Flow from
    # source frame 2 describes frame 3, and is deliberately corrupted.
    for key in ['flow_rgb','flow_ir']:
        second[key][2:] = 1000
    for key in ['embedding','ir_mask_embedding','rgb_mask_embedding','boxes']:
        second[key][4:] = -1000
    cfg = dict(flow='consensus',appearance_weight=.5,protected_memory=True,
               maximum_center_distance=3)
    a,b=PerceptualTracker(cfg),PerceptualTracker(cfg)
    for frame in [1,2]:
        out_a,out_b=a.update(frame,first),b.update(frame,second)
        for ra,rb in zip(out_a,out_b):
            np.testing.assert_array_equal(ra[1],rb[1])
            assert ra[2:]==rb[2:]
    assert a.statistics==b.statistics
