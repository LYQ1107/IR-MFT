"""Current-image classical photometric diagnostics; no fitting or labels."""
import cv2
import numpy as np


def histogram_match_uint8(source, reference):
    if source.dtype != np.uint8 or reference.dtype != np.uint8:
        raise ValueError('Original uint8 scalar images required')
    if source.ndim != 2 or source.shape != reference.shape:
        raise ValueError('Aligned scalar image shapes required')
    source_hist = np.bincount(source.ravel(), minlength=256)
    target_hist = np.bincount(reference.ravel(), minlength=256)
    occupied = np.flatnonzero(target_hist)
    source_cdf = source_hist.cumsum() / source.size
    target_cdf = target_hist.cumsum()[occupied] / reference.size
    lut = np.rint(np.interp(source_cdf, target_cdf, occupied)).astype(np.uint8)
    return lut[source], lut


def contrast_polarity(rgb, infrared, boxes, scores):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    exclusion = np.zeros_like(gray, dtype=bool)

    def rectangle(box, scale):
        box = np.asarray(box, dtype=float)
        center = (box[:2] + box[2:]) / 2
        extent = (box[2:] - box[:2]) * scale / 2
        lo = np.floor(center - extent).astype(int)
        hi = np.ceil(center + extent).astype(int)
        return np.clip([lo[0], hi[0]], 0, width), np.clip([lo[1], hi[1]], 0, height)

    for box in boxes:
        (x1, x2), (y1, y2) = rectangle(box, 1.)
        exclusion[y1:y2, x1:x2] = True
    votes = []
    for box, score in zip(boxes, scores):
        if score < .7:
            continue
        (x1, x2), (y1, y2) = rectangle(box, .5)
        if (x2 - x1) * (y2 - y1) < 16:
            continue
        (a, b), (c, d) = rectangle(box, 1.25)
        ring = ~exclusion[c:d, a:b]
        if ring.sum() < 16:
            continue
        rgb_contrast = float(np.median(gray[y1:y2, x1:x2])) - float(np.median(gray[c:d, a:b][ring]))
        ir_contrast = float(np.median(infrared[y1:y2, x1:x2])) - float(np.median(infrared[c:d, a:b][ring]))
        if min(abs(rgb_contrast), abs(ir_contrast)) < 8:
            continue
        votes.append(dict(score=float(score), rgb_contrast=rgb_contrast, ir_contrast=ir_contrast,
                          agreement=1 if rgb_contrast * ir_contrast > 0 else -1))
    signed_weight = sum(v['score'] * v['agreement'] for v in votes)
    return signed_weight < 0, dict(votes=votes, signed_weight=signed_weight, valid_votes=len(votes))


def photometric_views(rgb, infrared, boxes, scores):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    identity, _ = histogram_match_uint8(gray, gray)
    np.testing.assert_array_equal(identity, gray)
    flip, diagnostics = contrast_polarity(rgb, infrared, boxes, scores)
    mapped, _ = histogram_match_uint8(infrared, gray)
    corrected, lut = histogram_match_uint8(255 - infrared if flip else infrared, gray)
    return dict(gray=gray, ir_raw=infrared, ir_gray_norm=infrared, ir_cdf=mapped,
                ir_polarity_cdf=corrected), dict(flip=flip, gray_self_CDF_exact=True,
                                               corrected_LUT=lut.tolist(), **diagnostics)
