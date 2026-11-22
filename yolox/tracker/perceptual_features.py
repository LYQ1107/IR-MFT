"""Current-frame shape descriptors and strictly adjacent-frame local transport.

Independent implementation; classical HOG, saliency and pyramidal LK are
controls/primitives, not claims of new algorithms. No annotation is accepted.
"""
from __future__ import annotations

import cv2
import numpy as np


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def clipped_crop(image, box, margin=0.0):
    x1, y1, x2, y2 = map(float, box)
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    left = max(0, min(image.shape[1] - 1, int(np.floor(x1 - margin * w))))
    top = max(0, min(image.shape[0] - 1, int(np.floor(y1 - margin * h))))
    right = max(left + 1, min(image.shape[1], int(np.ceil(x2 + margin * w))))
    bottom = max(top + 1, min(image.shape[0], int(np.ceil(y2 + margin * h))))
    return image[top:bottom, left:right]


def foreground(gray):
    """Polarity-independent local contrast, with a soft spatial support prior."""
    gray = cv2.resize(gray, (64, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0.7)
    border = np.r_[smooth[:3].ravel(), smooth[-3:].ravel(),
                   smooth[:, :3].ravel(), smooth[:, -3:].ravel()]
    contrast = np.abs(smooth - np.median(border))
    yy, xx = np.mgrid[-1:1:32j, -1:1:64j]
    support = np.exp(-(xx * xx + yy * yy) / 1.1).astype(np.float32)
    strength = max(float(np.percentile(contrast, 85)), 3.0)
    mask = np.clip(contrast / strength, 0, 1) * np.sqrt(support)
    quality = float(np.clip((strength - 3) / 25, 0, 1))
    occupancy = float((mask > .35).mean())
    quality *= min(occupancy / .12, 1) * min((1 - occupancy) / .15, 1)
    return mask.astype(np.float32), quality


def hog(gray):
    gray = cv2.resize(gray, (64, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag, angle = cv2.cartToPolar(gx, gy)
    bins = np.minimum(((angle % np.pi) * (9 / np.pi)).astype(int), 8)
    descriptor = []
    for y in range(4):
        for x in range(4):
            sl = np.s_[y * 8:(y + 1) * 8, x * 16:(x + 1) * 16]
            hist = np.bincount(bins[sl].ravel(), weights=mag[sl].ravel(), minlength=9)
            descriptor.extend(unit(hist))
    return unit(descriptor)


def color_histogram(bgr):
    patch = cv2.resize(bgr, (64, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    parts = []
    for y in range(2):
        for x in range(2):
            cell = hsv[y * 16:(y + 1) * 16, x * 32:(x + 1) * 32]
            for channel, bins, end in [(0, 16, 180), (1, 8, 256), (2, 8, 256)]:
                hist = np.histogram(cell[:, :, channel], bins=bins, range=(0, end))[0]
                parts.extend(np.sqrt(hist / max(hist.sum(), 1)))
    return unit(parts)


def letterbox_tensor_array(image, box, mask=None, size=(128, 256)):
    """Same 5%-margin RGB letterbox as the preserved train-only fish ReID."""
    patch = clipped_crop(image, box, .05)
    mean = np.array([.485, .456, .406], np.float32)
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    if mask is not None:
        soft = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]))[:, :, None]
        # Retain context conservatively instead of cutting a binary silhouette.
        soft = .2 + .8 * soft
        rgb = rgb * soft + mean * (1 - soft)
    height, width = size
    ratio = min(height / rgb.shape[0], width / rgb.shape[1])
    resized = cv2.resize(rgb, (max(1, round(rgb.shape[1] * ratio)),
                              max(1, round(rgb.shape[0] * ratio))))
    canvas = np.broadcast_to(mean, (height, width, 3)).copy()
    dy, dx = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[dy:dy + resized.shape[0], dx:dx + resized.shape[1]] = resized
    return ((canvas - mean) / np.array([.229, .224, .225], np.float32)).transpose(2, 0, 1)


def local_transport(previous, current, boxes, full_size):
    """Robust LK translation per previous observation, FB checked using t-1,t.

    Frames are downsampled gray images. Returned displacement is expressed in
    the original image coordinates; no t+1 input is read for an update at t.
    """
    result = np.zeros((len(boxes), 4), np.float32)
    if not len(boxes):
        return result
    sx, sy = previous.shape[1] / full_size[1], previous.shape[0] / full_size[0]
    points, owner = [], []
    for index, box in enumerate(boxes):
        b = np.asarray(box) * np.array([sx, sy, sx, sy])
        x1, y1 = np.maximum(b[:2].astype(int), 0)
        x2, y2 = np.minimum(np.ceil(b[2:]).astype(int), [previous.shape[1], previous.shape[0]])
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        patch = previous[y1:y2, x1:x2]
        p = cv2.goodFeaturesToTrack(patch, maxCorners=20, qualityLevel=.005,
                                    minDistance=2, blockSize=3)
        if p is None:
            continue
        p = p.reshape(-1, 2) + [x1, y1]
        points.extend(p)
        owner.extend([index] * len(p))
    if not points:
        return result
    p0 = np.asarray(points, np.float32).reshape(-1, 1, 2)
    opts = dict(winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .02))
    p1, ok, err = cv2.calcOpticalFlowPyrLK(previous, current, p0, None, **opts)
    back, ok_back, _ = cv2.calcOpticalFlowPyrLK(current, previous, p1, None, **opts)
    fb = np.linalg.norm(back.reshape(-1, 2) - p0.reshape(-1, 2), axis=1)
    good = ok.ravel().astype(bool) & ok_back.ravel().astype(bool) & (fb < 1.5) & (err.ravel() < 40)
    movement = (p1 - p0).reshape(-1, 2) / [sx, sy]
    owner = np.asarray(owner)
    for index in np.unique(owner):
        pick = good & (owner == index)
        if pick.sum() < 3:
            continue
        median = np.median(movement[pick], axis=0)
        residual = np.linalg.norm(movement[pick] - median, axis=1)
        inlier = residual < max(3.0, float(np.median(residual)) * 2.5)
        if inlier.sum() < 3:
            continue
        result[index, :2] = np.median(movement[pick][inlier], axis=0)
        result[index, 2] = pick.sum() / (owner == index).sum() * inlier.mean()
        result[index, 3] = np.median(residual[inlier])
    return result
