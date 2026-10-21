#!/usr/bin/env python3
"""Strictly causal RGB-derived auxiliary representation.

The historical ``infrared`` folders are retained as a legacy exploratory
input.  This module is the online-safe fallback used when the original
external generation scripts are unavailable: it only compares the current
RGB luminance with an exponential background formed from earlier frames.
The current frame is emitted before the background is updated.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class CausalAuxiliaryGenerator:
    """Generate a deterministic pseudo-colour auxiliary image from RGB history."""

    def __init__(self, alpha: float = 0.01, response_scale: float = 4.0):
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.response_scale = float(response_scale)
        self.background: Optional[np.ndarray] = None
        self.frames_seen = 0
        self.sequence_id = None
        self.last_frame_id = None

    def reset(self, sequence_id=None) -> None:
        self.background = None
        self.frames_seen = 0
        self.sequence_id = sequence_id
        self.last_frame_id = None

    def _ensure_sequence(self, sequence_id, frame_id=None) -> None:
        if self.sequence_id != sequence_id:
            self.reset(sequence_id)
        if frame_id is not None and self.last_frame_id is not None and frame_id <= self.last_frame_id:
            # A repeated or out-of-order frame cannot share causal state.
            self.reset(sequence_id)

    def __call__(self, frame_bgr: np.ndarray, sequence_id=None, frame_id=None) -> np.ndarray:
        if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(
                "expected uint8 BGR frame, got "
                f"dtype={frame_bgr.dtype}, shape={frame_bgr.shape}"
            )
        self._ensure_sequence(sequence_id, frame_id)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.background is None or self.background.shape != gray.shape:
            self.background = gray.astype(np.float32)
        background_u8 = np.clip(self.background, 0.0, 255.0).astype(np.uint8)
        response = cv2.absdiff(gray, background_u8).astype(np.float32)
        response = np.clip(response * self.response_scale, 0.0, 255.0).astype(np.uint8)
        # A fixed colormap preserves a three-channel interface without claiming
        # physical thermal measurements.
        auxiliary = cv2.applyColorMap(response, cv2.COLORMAP_INFERNO)
        cv2.accumulateWeighted(gray, self.background, self.alpha)
        self.frames_seen += 1
        self.last_frame_id = frame_id
        return auxiliary


def grayscale_three_channel(frame_bgr: np.ndarray) -> np.ndarray:
    """Return a three-channel grayscale control with the original shape."""

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

