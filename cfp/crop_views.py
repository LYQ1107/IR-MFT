"""Declared crop-view repairs, preserving current-frame pairing and pixels."""
import cv2
import numpy as np

from .identity_data import crop_to_tensor


def long_axis_crop(image, xywh, size=(128, 256), margin=.05):
    x, y, w, h = map(float, xywh)
    if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0:
        raise ValueError('Invalid crop box')
    x1, y1 = max(0, int(np.floor(x - margin * w))), max(0, int(np.floor(y - margin * h)))
    x2, y2 = min(image.shape[1], int(np.ceil(x + (1 + margin) * w))), min(image.shape[0], int(np.ceil(y + (1 + margin) * h)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Crop lies outside image')
    region = image[y1:y2, x1:x2]
    if region.shape[0] > region.shape[1]:
        region = cv2.rotate(region, cv2.ROTATE_90_CLOCKWISE)
    return crop_to_tensor(region, [0, 0, region.shape[1], region.shape[0]], size=size, margin=0)
