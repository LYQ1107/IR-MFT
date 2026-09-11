#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Paired RGB-IR preprocessing for MFT26-IR."""

import random

import cv2
import numpy as np

from yolox.utils import xyxy2cxcywh

from .data_augment import preproc


class _PairedTransformBase:
    def __init__(
        self,
        rgb_means=None,
        std=None,
        ir_means=None,
        ir_std=None,
        swap=(2, 0, 1),
        max_labels=100,
    ):
        self.means = rgb_means
        self.std = std
        # The constructed IR images have a very different intensity
        # distribution from the visible images.  Keep the old arguments as
        # the default for V1 compatibility, while allowing V2+ to use
        # modality-specific calibration.
        self.ir_means = rgb_means if ir_means is None else ir_means
        self.ir_std = std if ir_std is None else ir_std
        self.swap = swap
        self.max_labels = max_labels

    def _preprocess_pair(self, rgb, ir, input_dim):
        rgb, rgb_ratio = preproc(rgb, input_dim, self.means, self.std, self.swap)
        ir, ir_ratio = preproc(
            ir, input_dim, self.ir_means, self.ir_std, self.swap
        )
        if not np.isclose(rgb_ratio, ir_ratio):
            raise AssertionError(
                "RGB and IR received different resize ratios: "
                f"{rgb_ratio} vs {ir_ratio}"
            )
        return np.ascontiguousarray(
            np.concatenate((rgb, ir), axis=0), dtype=np.float32
        ), rgb_ratio

    def _pad_labels(self, labels):
        padded = np.zeros((self.max_labels, 6), dtype=np.float32)
        if len(labels):
            count = min(len(labels), self.max_labels)
            padded[:count] = labels[:count]
        return np.ascontiguousarray(padded, dtype=np.float32)


class RGBTTrainTransform(_PairedTransformBase):
    """Training transform with shared geometry for both modalities.

    V1 intentionally uses only synchronized horizontal flipping, resizing and
    padding. Mosaic, MixUp, independent color distortion, crop and affine
    transforms are disabled until the paired pipeline has been validated.
    """

    def __init__(
        self,
        p=0.5,
        rgb_means=None,
        std=None,
        ir_means=None,
        ir_std=None,
        max_labels=100,
    ):
        super().__init__(
            rgb_means,
            std,
            ir_means=ir_means,
            ir_std=ir_std,
            max_labels=max_labels,
        )
        self.p = p

    def __call__(self, rgb, ir, targets, input_dim):
        if rgb is None or ir is None:
            raise ValueError("RGB and IR images must both be present")
        if rgb.shape[:2] != ir.shape[:2]:
            raise ValueError(
                "RGB and IR image sizes differ: "
                f"{rgb.shape[:2]} vs {ir.shape[:2]}"
            )

        targets = np.asarray(targets, dtype=np.float32).copy()
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        track_ids = targets[:, 5].copy()

        if random.random() < self.p:
            width = rgb.shape[1]
            rgb = np.ascontiguousarray(rgb[:, ::-1])
            ir = np.ascontiguousarray(ir[:, ::-1])
            if len(boxes):
                old_x1 = boxes[:, 0].copy()
                old_x2 = boxes[:, 2].copy()
                boxes[:, 0] = width - old_x2
                boxes[:, 2] = width - old_x1

        input_image, ratio = self._preprocess_pair(rgb, ir, input_dim)

        if len(boxes):
            boxes = xyxy2cxcywh(boxes)
            boxes *= ratio
            valid = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
            boxes = boxes[valid]
            labels = labels[valid]
            track_ids = track_ids[valid]
            labels = np.expand_dims(labels, 1)
            track_ids = np.expand_dims(track_ids, 1)
            targets = np.hstack((labels, boxes, track_ids))
        else:
            targets = np.zeros((0, 6), dtype=np.float32)

        return input_image, self._pad_labels(targets)


class RGBTValTransform(_PairedTransformBase):
    """Validation preprocessing without random operations."""

    def __init__(
        self, rgb_means=None, std=None, ir_means=None, ir_std=None
    ):
        super().__init__(
            rgb_means,
            std,
            ir_means=ir_means,
            ir_std=ir_std,
            max_labels=1,
        )

    def __call__(self, rgb, ir, targets, input_dim):
        del targets
        input_image, _ = self._preprocess_pair(rgb, ir, input_dim)
        return input_image, np.zeros((1, 5), dtype=np.float32)
